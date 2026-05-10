# Train GCNClassifier over MotionBERT 3D skeletons (H36M-17, xyz)
# Data source: Dataset/MotionBERT_3d/V*/X3D.npy — each file is (total_frames, 17, 3)
# Punch labels: Annotation_files/*.xlsx (same frame-range convention as 2D pipeline).
# no_punch negatives: Dataset/gap_labels/gap_review.json (entries with label == "no_punch").
#
# Preprocessing matches punch_classifier.py:
#   1. Translate pelvis (joint 0) to origin
#   2. Rotate into boxer's body frame (+x right, +z up, +y forward) using frame 0
#   3. Scale by torso length (pelvis → thorax distance in frame 0)
#
# Augmentations (training only):
#   - Temporal jitter: random ±JITTER_RANGE frames on window centre
#   - Mirror flip (50%): negate x + swap L/R joints; labels unchanged (body-frame semantics)

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from GCN import (
    H36M_BONE_PAIRS,
    PUNCH_CLASSES,
    make_class_weights,
)
from punch_transformer import PunchTransformer
from preprocess import _load_annotations, _normalize_label

_REPO = Path.cwd().resolve()
_MOTIONBERT_DIR = _REPO / "Dataset" / "MotionBERT_3d"
_ANNOTATION_DIR = _REPO / "Dataset" / "Annotation_files"
_GAP_REVIEW_JSON = _REPO / "Dataset" / "gap_labels" / "gap_review.json"

CLASSIFIER_CLASSES: list[str] = [*PUNCH_CLASSES, "no_punch"]
NO_PUNCH_IDX = len(PUNCH_CLASSES)
CLF_WINDOW = 8
JITTER_RANGE = 2   # ± frames shifted from centre during training
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
LR_MIN = 1e-5
WARMUP_EPOCHS = 5
VAL_FRAC = 0.1
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# H36M-17 joint indices (matches punch_classifier._J)
_J_PELVIS    = 0
_J_THORAX    = 8
_J_L_SHOULDER = 11
_J_R_SHOULDER = 14

_LABEL_TO_IDX: dict[str, int] = {
    "Cross":         PUNCH_CLASSES.index("cross"),
    "Jab":           PUNCH_CLASSES.index("jab"),
    "Lead Hook":     PUNCH_CLASSES.index("lead_hook"),
    "Lead Uppercut": PUNCH_CLASSES.index("lead_uppercut"),
    "Rear Hook":     PUNCH_CLASSES.index("rear_hook"),
    "Rear Uppercut": PUNCH_CLASSES.index("rear_uppercut"),
}

# H36M-17: swap L/R when mirroring (pelvis(0) … l_wrist(13) r_shoulder(14)…)
_FLIP_JOINT_ORDER = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


def _no_punch_spans_from_gap_review() -> list[tuple[str, int, int]]:
    """``(workbook, start, end)`` half-open frame indices for each no_punch chunk."""
    if not _GAP_REVIEW_JSON.is_file():
        raise SystemExit(f"Missing {_GAP_REVIEW_JSON} — run label_punch_gaps.py")
    data = json.loads(_GAP_REVIEW_JSON.read_text(encoding="utf-8"))
    rows: list[tuple[str, int, int]] = []
    for e in data.get("entries", []):
        if e.get("label") != "no_punch":
            continue
        ver = str(e["workbook"]).strip().upper()
        s0, e0 = int(e["start"]), int(e["end"])
        if e0 > s0:
            rows.append((ver, s0, e0))
    return rows


# =============================================================================
# Preprocessing — body frame + torso scale (mirrors punch_classifier.py)
# =============================================================================


def _to_body_frame(poses: np.ndarray) -> np.ndarray:
    """
    Translate pelvis to origin, then rotate into the boxer's body frame:
        +x  boxer's right (R shoulder → L shoulder axis)
        +z  up            (pelvis → thorax axis)
        +y  forward       (ẑ × x̂)

    Rotation built from frame 0, same as punch_classifier.BoxingPunchClassifier._to_body_frame.
    """
    q = poses - poses[:, [_J_PELVIS], :]

    ref = q[0]
    x_raw = ref[_J_R_SHOULDER] - ref[_J_L_SHOULDER]
    x_norm = np.linalg.norm(x_raw)
    if x_norm < 1e-6:
        return q
    x_hat = x_raw / x_norm

    z_raw = ref[_J_THORAX] - ref[_J_PELVIS]
    z_norm = np.linalg.norm(z_raw)
    if z_norm < 1e-6:
        return q
    z_raw = z_raw / z_norm
    z_hat = z_raw - np.dot(z_raw, x_hat) * x_hat
    z_norm2 = np.linalg.norm(z_hat)
    if z_norm2 < 1e-6:
        return q
    z_hat = z_hat / z_norm2

    y_hat = np.cross(z_hat, x_hat)

    R = np.stack([x_hat, y_hat, z_hat], axis=0)
    return q @ R.T


def _scale_normalize(q: np.ndarray) -> np.ndarray:
    torso = float(np.linalg.norm(q[0, _J_THORAX] - q[0, _J_PELVIS]))
    return q / torso if torso > 1e-6 else q


def _preprocess_clip(clip: np.ndarray) -> np.ndarray:
    q = _to_body_frame(clip.astype(np.float64))
    q = _scale_normalize(q)
    q = q.astype(np.float32)
    vel = np.zeros_like(q)
    vel[1:] = q[1:] - q[:-1]          # frame-to-frame joint velocity; zero at t=0
    return np.nan_to_num(
        np.concatenate([q, vel], axis=-1),   # (T, 17, 6): xyz + dxyz
        nan=0.0, posinf=0.0, neginf=0.0,
    )


# =============================================================================
# Windowing
# =============================================================================


def _prepare_window_3d(seq: np.ndarray, window: int, jitter: int = 0) -> np.ndarray:
    """
    Center-crop seq (T, 17, C) to ``window`` frames.

    ``jitter`` shifts the crop centre (clamped in-bounds). Pads by repeating edge
    frames when the clip is shorter than ``window``.
    """
    T = seq.shape[0]
    half = window // 2

    if T < window:
        pad_pre  = (window - T) // 2
        pad_post = window - T - pad_pre
        seq = np.concatenate([
            np.tile(seq[[0]], (pad_pre, 1, 1)),
            seq,
            np.tile(seq[[-1]], (pad_post, 1, 1)),
        ], axis=0)
        T = window

    peak  = T // 2 + jitter
    peak  = max(half, min(T - (window - half), peak))   # clamp to valid range
    start = peak - half
    chunk = seq[start: start + window]

    if chunk.shape[0] < window:
        pad   = window - chunk.shape[0]
        chunk = np.concatenate([chunk, np.tile(chunk[[-1]], (pad, 1, 1))], axis=0)

    return chunk


# =============================================================================
# Data loading — returns raw (variable-length) preprocessed clips
# =============================================================================


def load_3d_clips() -> tuple[list[np.ndarray], np.ndarray]:
    """
    Punch clips from Annotation_files + ``no_punch`` clips from ``gap_review.json``.

    Returns
    -------
    clips : list of (T_i, 17, C) preprocessed tensors (xyz + velocities)
    y     : (N,) int64 — class indices into CLASSIFIER_CLASSES (index 6 = no_punch)
    """
    clips: list[np.ndarray] = []
    y_list: list[int] = []

    for ver_dir in sorted(_MOTIONBERT_DIR.iterdir()):
        npy_path = ver_dir / "X3D.npy"
        if not npy_path.exists():
            continue
        ver      = ver_dir.name.upper()
        ann_path = _ANNOTATION_DIR / f"{ver}.xlsx"
        if not ann_path.exists():
            print(f"[skip] no annotation file for {ver}")
            continue

        frames      = np.load(npy_path)
        annotations = _load_annotations(ann_path)
        n_total     = frames.shape[0]

        kept = 0
        for s, e, raw_label in annotations:
            label = _normalize_label(str(raw_label))
            if label not in _LABEL_TO_IDX:
                continue
            s0, e0 = s - 1, min(e, n_total)
            if e0 <= s0:
                continue
            clip = _preprocess_clip(frames[s0:e0].copy())
            clips.append(clip)
            y_list.append(_LABEL_TO_IDX[label])
            kept += 1

        print(f"{ver}: {kept}/{len(annotations)} punch clips loaded")

    if not clips:
        raise SystemExit("No 3D punch clips — check Dataset/MotionBERT_3d/ structure.")

    spans = _no_punch_spans_from_gap_review()
    if not spans:
        raise SystemExit(f"No no_punch entries in {_GAP_REVIEW_JSON} — run label_punch_gaps.py")

    cache: dict[str, np.ndarray] = {}
    n_np = 0
    skipped = 0
    for ver, s0, e0 in spans:
        npy_path = _MOTIONBERT_DIR / ver / "X3D.npy"
        if not npy_path.exists():
            skipped += 1
            continue
        if ver not in cache:
            cache[ver] = np.load(npy_path)
        frames = cache[ver]
        n_total = frames.shape[0]
        e0 = min(e0, n_total)
        if e0 <= s0:
            skipped += 1
            continue
        clip = _preprocess_clip(frames[s0:e0].copy())
        clips.append(clip)
        y_list.append(NO_PUNCH_IDX)
        n_np += 1

    print(
        f"gap_review.json: {n_np} no_punch clips "
        f"({skipped} spans skipped — missing X3D or out of range)"
    )
    if n_np == 0:
        raise SystemExit("Could not load any no_punch clips — check gap_review paths vs MotionBERT_3d.")

    return clips, np.array(y_list, dtype=np.int64)


# =============================================================================
# Dataset — augmentation in __getitem__ when augment=True (train only)
# =============================================================================


class Clf3DDataset(Dataset):
    def __init__(
        self,
        clips: list[np.ndarray],
        y: np.ndarray,
        window: int,
        augment: bool = False,
    ):
        self.clips   = clips
        self.y       = y.astype(np.int64)
        self.window  = window
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        clip  = self.clips[i]
        label = int(self.y[i])

        jitter = (
            int(np.random.randint(-JITTER_RANGE, JITTER_RANGE + 1))
            if self.augment
            else 0
        )
        win = _prepare_window_3d(clip, self.window, jitter=jitter)

        if self.augment and np.random.random() < 0.5:
            win = win[:, _FLIP_JOINT_ORDER, :].copy()
            win[:, :, 0] *= -1
            win[:, :, 3] *= -1

        # (T, V, C) — DataLoader stacks into (N, T, V, C)
        return torch.from_numpy(win), torch.tensor(label, dtype=torch.long)


def _display_class_name(c: str) -> str:
    if c == "no_punch":
        return "No Punch"
    return c.replace("_", " ").title()


# =============================================================================
# Training
# =============================================================================

if not _MOTIONBERT_DIR.is_dir():
    raise SystemExit(f"Missing: {_MOTIONBERT_DIR}")
if not _ANNOTATION_DIR.is_dir():
    raise SystemExit(f"Missing: {_ANNOTATION_DIR}")

print(f"Loading 3D clips from {_MOTIONBERT_DIR.relative_to(_REPO)} …")
print(f"No-punch source: {_GAP_REVIEW_JSON.relative_to(_REPO)}")
all_clips, all_y = load_3d_clips()
print(f"\nLoaded {len(all_y)} clips  device={DEVICE}")
print(
    "Class distribution:",
    {CLASSIFIER_CLASSES[k]: v for k, v in sorted(Counter(all_y.tolist()).items())},
)

uniq, counts = np.unique(all_y, return_counts=True)
can_stratify = bool(np.all(counts >= 2))

idx_train, idx_val = train_test_split(
    np.arange(len(all_y)),
    test_size=VAL_FRAC,
    random_state=SEED,
    stratify=all_y if can_stratify else None,
)

train_clips = [all_clips[i] for i in idx_train]
val_clips   = [all_clips[i] for i in idx_val]

train_ds = Clf3DDataset(train_clips, all_y[idx_train], window=CLF_WINDOW, augment=True)
val_ds   = Clf3DDataset(val_clips,   all_y[idx_val],   window=CLF_WINDOW, augment=False)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

train_counts = Counter(all_y[idx_train].tolist())
class_weight_dict = {
    CLASSIFIER_CLASSES[i]: max(train_counts.get(i, 0), 1)
    for i in range(len(CLASSIFIER_CLASSES))
}
class_weights = make_class_weights(class_weight_dict, classes=CLASSIFIER_CLASSES, device=DEVICE)
print(
    "Class weights:",
    {CLASSIFIER_CLASSES[k]: f"{v:.3f}" for k, v in sorted(
        enumerate(class_weights.tolist()), key=lambda x: x[0])},
)
crit = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

model = PunchTransformer(
    num_classes=len(CLASSIFIER_CLASSES),
    in_channels=6,
    edges=H36M_BONE_PAIRS,
    spatial_hidden=64,
    d_model=128,
    nhead=4,
    num_layers=4,
    dim_feedforward=256,
    dropout=0.1,
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

opt           = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
cosine_epochs = max(1, EPOCHS - WARMUP_EPOCHS)
sched = torch.optim.lr_scheduler.SequentialLR(
    opt,
    schedulers=[
        LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=min(WARMUP_EPOCHS, EPOCHS)),
        CosineAnnealingLR(opt, T_max=cosine_epochs, eta_min=LR_MIN),
    ],
    milestones=[min(WARMUP_EPOCHS, EPOCHS)],
)


@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    all_p, all_t = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss   = crit(logits, yb)
        tot   += loss.item() * yb.size(0)
        pred   = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n      += yb.size(0)
        all_p.append(pred.cpu())
        all_t.append(yb.cpu())
    return tot / max(n, 1), correct / max(n, 1), torch.cat(all_p).numpy(), torch.cat(all_t).numpy()


best_acc   = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss, run_ok, run_n = 0.0, 0, 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss   = crit(logits, yb)
        loss.backward()
        opt.step()
        run_loss += loss.item() * yb.size(0)
        run_ok   += (logits.argmax(1) == yb).sum().item()
        run_n    += yb.size(0)
    sched.step()
    lr_now = opt.param_groups[0]["lr"]

    va_loss, va_acc, _, _ = eval_epoch(val_dl)
    if va_acc > best_acc:
        best_acc   = va_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"epoch {epoch:02d}/{EPOCHS}  lr {lr_now:.2e}  "
        f"train loss {run_loss / max(run_n, 1):.4f} acc {run_ok / max(run_n, 1):.3f}  "
        f"val loss {va_loss:.4f} acc {va_acc:.3f}"
    )

if best_state is not None:
    model.load_state_dict(best_state)
_, _, vp, vt = eval_epoch(val_dl)
print(f"\nBest val acc: {best_acc:.3f}")
print("\nClassification report (val, best checkpoint):")
print(
    classification_report(
        vt, vp,
        target_names=[_display_class_name(c) for c in CLASSIFIER_CLASSES],
        zero_division=0,
    )
)
print(
    "Confusion matrix:\n",
    confusion_matrix(vt, vp, labels=list(range(len(CLASSIFIER_CLASSES)))),
)

vers_tag = "_".join(
    v.name.upper()
    for v in sorted(_MOTIONBERT_DIR.iterdir())
    if (v / "X3D.npy").exists()
)
ckpt = _REPO / "checkpoints" / f"punch_transformer_{vers_tag}_gap_review.pt"
ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state":   best_state,
        "model_class":   "PunchTransformer",
        "punch_classes": CLASSIFIER_CLASSES,
        "punch_classes_base": PUNCH_CLASSES,
        "no_punch_index": NO_PUNCH_IDX,
        "label_map":     _LABEL_TO_IDX,
        "window":        CLF_WINDOW,
        "in_channels":   6,
        "skeleton":      "H36M-17",
        "source":        "MotionBERT_3d",
        "preprocessing": "body_frame + torso_scale",
        "negatives":     f"gap_review.json @ {_GAP_REVIEW_JSON.relative_to(_REPO)}",
        "augmentation":  f"jitter±{JITTER_RANGE} + mirror_flip(50%, no label swap)",
        "lr_schedule": {
            "warmup_epochs": WARMUP_EPOCHS,
            "warmup":  "LinearLR 0.01→1.0 × base LR",
            "cosine":  f"CosineAnnealingLR T_max={cosine_epochs} eta_min={LR_MIN}",
        },
    },
    ckpt,
)
print(f"Checkpoint → {ckpt.relative_to(_REPO)}")
