"""
train.py — GCNDetector and GCNClassifier training for BoxingVI.

Input shape to both models: [N, M=1, T, V=12, C=2]  (xy normalised coords)
  Detector:   T=11 sliding windows, binary punch/no-punch, BCELoss
  Classifier: T=20 centred clips,   6 punch types, CrossEntropyLoss

Train/Val/Test split is video-level (V1-V7 / V8-V9 / V10) to prevent leakage.

Usage:
    # Detector — extract full-video poses on first run (cached afterwards):
    python train.py detector [--epochs 50] [--batch 64] [--lr 1e-3]

    # Classifier:
    python train.py classifier [--epochs 100] [--batch 32] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from GCN import GCNClassifier, GCNDetector, PUNCH_CLASSES
from preprocess import (
    _LANDMARK_IDX,
    _MODEL_PATH,
    _ensure_model,
    _find_video,
    prepare_windows,
)

# ── Paths ────────────────────────────────────────────────────────────────────

_REPO          = Path(__file__).resolve().parent
_POSE_SEQ_DIR  = _REPO / "Dataset" / "pose_sequences"
_DET_LABEL_DIR = _REPO / "Dataset" / "detection_frame_labels"
_LANDMARKS_NPZ = _REPO / "Dataset" / "landmarks.npz"
_CKPT_DIR      = _REPO / "checkpoints"

# ── 12-joint topology (matches MEDIAPIPE_TO_PAPER order) ─────────────────────
# 0=l_shoulder  1=r_shoulder  2=l_elbow  3=r_elbow  4=l_wrist  5=r_wrist
# 6=l_hip       7=r_hip       8=l_knee   9=r_knee  10=l_ankle 11=r_ankle

BONE_PAIRS_12 = [
    (0, 2), (2, 4),   # left arm
    (1, 3), (3, 5),   # right arm
    (0, 1),           # shoulders
    (0, 6), (1, 7),   # torso sides
    (6, 7),           # hips
    (6, 8), (8, 10),  # left leg
    (7, 9), (9, 11),  # right leg
]
CENTER_12 = 6  # left_hip

BACKBONE_12 = dict(
    edges=BONE_PAIRS_12,
    center=CENTER_12,
    num_joints=12,
    base_channels=32,
    inflate_stages=[4, 7],
    down_stages=[4, 7],
)

# ── Video split ───────────────────────────────────────────────────────────────

TRAIN_VERS = [f"V{i}" for i in range(1, 8)]   # V1–V7
VAL_VERS   = ["V8", "V9"]
TEST_VERS  = ["V10"]

# ── Label vocabulary ─────────────────────────────────────────────────────────

# Map annotation labels → PUNCH_CLASSES index (case-insensitive, space→underscore)
_LABEL_MAP = {lbl.lower().replace(" ", "_"): i for i, lbl in enumerate(PUNCH_CLASSES)}

def _label_to_idx(raw: str) -> int:
    return _LABEL_MAP[raw.strip().lower().replace(" ", "_")]


# ══════════════════════════════════════════════════════════════════════════════
# Full-video pose extraction (for detector dataset)
# ══════════════════════════════════════════════════════════════════════════════

def extract_full_video_poses(versions: list[str] | None = None) -> None:
    """
    Run MediaPipe on every frame of each video; save (F, 12, 2) arrays.

    Skips versions whose cache file already exists.
    Output: Dataset/pose_sequences/V{i}_pose.npz  with key 'pose' shape (F, 12, 2).
    """
    _POSE_SEQ_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_model()

    if versions is None:
        versions = [f"V{i}" for i in range(1, 11)]

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for ver in versions:
            out = _POSE_SEQ_DIR / f"{ver}_pose.npz"
            if out.exists():
                print(f"[{ver}] pose cache exists — skip")
                continue

            vid = _find_video(ver)
            if vid is None:
                print(f"[{ver}] no video found — skip", file=sys.stderr)
                continue

            cap = cv2.VideoCapture(str(vid))
            F = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[{ver}] extracting poses for {F} frames …")

            pose = np.zeros((F, 12, 2), dtype=np.float32)
            for f in range(F):
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )
                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    for j, mp_idx in enumerate(_LANDMARK_IDX):
                        lm = lms[mp_idx]
                        pose[f, j, 0] = lm.x
                        pose[f, j, 1] = lm.y
                if (f + 1) % 1000 == 0:
                    print(f"  {f + 1}/{F} frames done")

            cap.release()
            np.savez_compressed(out, pose=pose)
            print(f"[{ver}] saved → {out.name}  (shape {pose.shape})")


# ══════════════════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════════════════

class DetectionDataset(Dataset):
    """
    Sliding-window detection dataset.

    Each item: (x, y) where
      x : float32 tensor [1, T, 12, 2]  — [M, T, V, C]
      y : float32 scalar — 1.0 if any punch frame in window, else 0.0
    """

    def __init__(self, versions: list[str], window: int = 11):
        self.window = window
        self.samples: list[tuple[np.ndarray, float]] = []

        for ver in versions:
            pose_path  = _POSE_SEQ_DIR / f"{ver}_pose.npz"
            label_path = _DET_LABEL_DIR / f"{ver}_detection.npz"
            if not pose_path.exists() or not label_path.exists():
                print(f"[{ver}] missing pose or label file — skip", file=sys.stderr)
                continue

            pose   = np.load(pose_path)["pose"]             # (F, 12, 2)
            labels = np.load(label_path)

            starts    = labels["window_starts"]              # (W,) int64
            is_punch  = labels["window_is_punch"].astype(np.float32)  # (W,)

            for t, y in zip(starts, is_punch):
                end = t + window
                if end > len(pose):
                    continue
                clip = pose[t:end]                           # (T, 12, 2)
                self.samples.append((clip, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        clip, y = self.samples[idx]
        x = torch.from_numpy(clip).unsqueeze(0)             # [1, T, 12, 2]
        return x, torch.tensor(y, dtype=torch.float32)

    def pos_weight(self) -> torch.Tensor:
        """BCEWithLogitsLoss pos_weight = #neg / #pos."""
        labels = np.array([s[1] for s in self.samples])
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)


class ClassifierDataset(Dataset):
    """
    Per-clip punch-type classification dataset.

    Each item: (x, y) where
      x : float32 tensor [1, 20, 12, 2]  — [M, T, V, C]
      y : int64 class index in [0, 5]
    """

    def __init__(self, versions: list[str], window: int = 20):
        data = np.load(_LANDMARKS_NPZ, allow_pickle=True)
        seqs     = data["sequences"]   # object array of (T_i, 12, 2)
        raw_lbls = data["labels"]      # (N,) str
        vers_arr = data["versions"]    # (N,) str

        mask = np.isin(vers_arr, versions)
        seqs     = seqs[mask]
        raw_lbls = raw_lbls[mask]

        X, y_str = prepare_windows(seqs, raw_lbls, window=window)  # (N, 20, 12, 2)
        y = np.array([_label_to_idx(lbl) for lbl in y_str], dtype=np.int64)

        # replace any residual NaN with 0
        X = np.nan_to_num(X, nan=0.0)

        self.X = torch.from_numpy(X).unsqueeze(1)   # [N, 1, 20, 12, 2]
        self.y = torch.from_numpy(y)                # [N]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss."""
        counts = torch.zeros(len(PUNCH_CLASSES))
        for lbl in self.y:
            counts[lbl] += 1
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.sum() * len(PUNCH_CLASSES)


# ══════════════════════════════════════════════════════════════════════════════
# Training utilities
# ══════════════════════════════════════════════════════════════════════════════

def _make_balanced_sampler(dataset: DetectionDataset) -> WeightedRandomSampler:
    labels = np.array([s[1] for s in dataset.samples])
    n_pos  = labels.sum()
    n_neg  = len(labels) - n_pos
    w_pos  = 1.0 / max(n_pos, 1)
    w_neg  = 1.0 / max(n_neg, 1)
    sample_weights = torch.tensor(
        [w_pos if y == 1 else w_neg for y in labels], dtype=torch.float32
    )
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_detector: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _evaluate_detector(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs, all_true = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        probs = model(x)
        total_loss += criterion(probs, y).item() * len(y)
        all_probs.append(probs.cpu().numpy())
        all_true.append(y.cpu().numpy())

    probs = np.concatenate(all_probs)
    true  = np.concatenate(all_true)
    preds = (probs >= threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        true, preds, average="binary", zero_division=0
    )
    acc = (preds == true.astype(int)).mean()
    return {
        "loss": total_loss / len(loader.dataset),
        "acc":  acc,
        "prec": p,
        "rec":  r,
        "f1":   f1,
    }


@torch.no_grad()
def _evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_true.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    true  = np.concatenate(all_true)
    acc   = (preds == true).mean()
    f1    = f1_score(true, preds, average="macro", zero_division=0)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc":  acc,
        "f1":   f1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Top-level training routines
# ══════════════════════════════════════════════════════════════════════════════

def train_detector(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)

    all_vers = TRAIN_VERS + VAL_VERS + TEST_VERS
    if args.extract_poses:
        extract_full_video_poses(all_vers)
    else:
        missing = [
            v for v in all_vers
            if not (_POSE_SEQ_DIR / f"{v}_pose.npz").exists()
        ]
        if missing:
            print(
                f"Pose cache missing for {missing}.\n"
                "Re-run with --extract-poses to generate it.",
                file=sys.stderr,
            )
            sys.exit(1)

    print("\nBuilding datasets …")
    train_ds = DetectionDataset(TRAIN_VERS)
    val_ds   = DetectionDataset(VAL_VERS)
    test_ds  = DetectionDataset(TEST_VERS)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    sampler    = _make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch * 2, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch * 2, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = GCNDetector(
        in_channels=2,
        num_joints=12,
        feature_dim=64,
        dropout=args.dropout,
        backbone_kwargs=BACKBONE_12,
    ).to(device)

    pos_weight = train_ds.pos_weight().to(device)
    criterion  = nn.BCELoss()                       # model already applies sigmoid
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    print(f"\nTraining GCNDetector for {args.epochs} epochs on {device}\n")
    best_f1, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device, True)
        val_m      = _evaluate_detector(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_m['loss']:.4f}  acc={val_m['acc']:.3f}  "
            f"prec={val_m['prec']:.3f}  rec={val_m['rec']:.3f}  f1={val_m['f1']:.3f}"
        )

        if val_m["f1"] > best_f1:
            best_f1, best_epoch = val_m["f1"], epoch
            torch.save(model.state_dict(), _CKPT_DIR / "detector_best.pt")

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch} (best F1={best_f1:.3f} @ ep {best_epoch})")
            break

    print(f"\nBest val F1: {best_f1:.3f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(_CKPT_DIR / "detector_best.pt", map_location=device))
    test_m = _evaluate_detector(model, test_loader, criterion, device)
    print(
        f"\n── Test results ──\n"
        f"  loss={test_m['loss']:.4f}  acc={test_m['acc']:.3f}  "
        f"prec={test_m['prec']:.3f}  rec={test_m['rec']:.3f}  f1={test_m['f1']:.3f}"
    )


def train_classifier(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building datasets …")
    train_ds = ClassifierDataset(TRAIN_VERS)
    val_ds   = ClassifierDataset(VAL_VERS)
    test_ds  = ClassifierDataset(TEST_VERS)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch * 2, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch * 2, shuffle=False,
                              num_workers=4, pin_memory=True)

    class_w = train_ds.class_weights().to(device)
    model   = GCNClassifier(
        num_classes=len(PUNCH_CLASSES),
        in_channels=2,
        num_joints=12,
        feature_dim=128,
        dropout=args.dropout,
        backbone_kwargs=BACKBONE_12,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    print(f"\nTraining GCNClassifier for {args.epochs} epochs on {device}\n")
    best_f1, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device, False)
        val_m      = _evaluate_classifier(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_m['loss']:.4f}  acc={val_m['acc']:.3f}  f1={val_m['f1']:.3f}"
        )

        if val_m["f1"] > best_f1:
            best_f1, best_epoch = val_m["f1"], epoch
            torch.save(model.state_dict(), _CKPT_DIR / "classifier_best.pt")

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch} (best F1={best_f1:.3f} @ ep {best_epoch})")
            break

    print(f"\nBest val macro-F1: {best_f1:.3f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(_CKPT_DIR / "classifier_best.pt", map_location=device))

    # ── Full test report ──────────────────────────────────────────────────────
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y.numpy())

    preds = np.concatenate(all_preds)
    true  = np.concatenate(all_true)
    print("\n── Test results ──")
    print(classification_report(true, preds, target_names=PUNCH_CLASSES, zero_division=0))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GCNDetector or GCNClassifier")
    p.add_argument("mode", choices=["detector", "classifier"])
    p.add_argument("--extract-poses", action="store_true",
                   help="(detector only) run MediaPipe on all videos before training")
    p.add_argument("--epochs",   type=int,   default=60)
    p.add_argument("--batch",    type=int,   default=64)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--wd",       type=float, default=1e-4,  help="weight decay")
    p.add_argument("--dropout",  type=float, default=0.2)
    p.add_argument("--patience", type=int,   default=15,    help="early-stop patience (epochs)")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.mode == "detector":
        train_detector(args)
    else:
        train_classifier(args)
