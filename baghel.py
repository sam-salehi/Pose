"""
Train a punch-type classifier on the BoxingVI pose dataset.

Paradigms:
  gru (default): PunchNet on a chosen feature branch (see --features) or raw normalised skeleton
  fae: FAE 192-D + StandardScaler + KNN (classical; use --paradigm fae)

  Paper-style branches (pixel windows → encode → BiGRU):
    --features skeleton  hip-centred normalised (x,y) per joint (default)
    --features uae       UAE angles + forward Δ (§4.2.1)
    --features 2dmdd     joint Taylor stack [P; P_v; P_a] (Eq. 8)
    --features fae       fifth-order angular sequence (16×12 timesteps)

  Class imbalance: --class-weight inverse (or sqrt) upweights rare classes in the loss.

Usage:
    python train.py --features skeleton
    python train.py --features uae --epochs 40
    python train.py --features 2dmdd --split lovo
    python train.py --features fae
    python train.py --paradigm fae --split cv10   # FAE + KNN baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import prepare_windows
from fae import encode_fae
from paper_features import encode_branch

# ── Constants ─────────────────────────────────────────────────────────────────

_REPO     = Path(__file__).resolve().parent
_DATA_NPZ = _REPO / "Dataset" / "landmarks.npz"

# Canonical class order for BoxingVI (6 classes)
CLASSES = ["Jab", "Cross", "Lead Hook", "Rear Hook", "Lead Uppercut", "Rear Uppercut"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

_FIGURES_DIR = _REPO / "figures"


def _save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    out_path: Path,
    title: str,
) -> None:
    """Save a confusion matrix image with class names on both axes."""
    labels_idx = np.arange(len(CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Joint indices inside our 12-joint array (matches MEDIAPIPE_TO_PAPER order)
_J_LEFT_SHOULDER  = 0
_J_RIGHT_SHOULDER = 1
_J_LEFT_HIP       = 6
_J_RIGHT_HIP      = 7

# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset(
    npz_path: Path = _DATA_NPZ,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load landmarks.npz and apply windowing; optionally skeleton normalisation.

    Returns:
        X        : (N, 20, 12, 2) float32 — GRU: normalised; FAE: raw pixel-space windows
        y        : (N,) int64            — class indices 0-5
        versions : (N,) str array        — source video ('V1'…'V10')
    """
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found.\n"
            "Run  python preprocess.py extract  first to generate it."
        )

    data     = np.load(npz_path, allow_pickle=True)
    seqs     = data["sequences"]    # object array of (T_i, 12, 2)
    labels   = data["labels"]       # string array
    versions = data["versions"]     # 'V1'…'V10'

    # ── windowing ─────────────────────────────────────────────────────────────
    X_raw, y_str = prepare_windows(seqs, labels)   # (N, 20, 12, 2), (N,)
    X_raw = np.nan_to_num(X_raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # ── skeleton normalisation (FAE uses raw windows; math.md centres internally) ──
    if normalize:
        X = _normalise_skeleton(X_raw)
    else:
        X = X_raw.astype(np.float32)

    # ── label encoding ────────────────────────────────────────────────────────
    y = _encode_labels(y_str)

    return X, y, versions


def _normalise_skeleton(X: np.ndarray) -> np.ndarray:
    """
    Centre each frame on the hip midpoint and scale by torso length.

    X : (N, 20, 12, 2)
    Returns the same shape, float32.

    Torso length = distance from hip midpoint to shoulder midpoint,
    averaged over the 20 frames to give a per-clip scale.
    Clips where all frames are zero (full detection failure) are left as-is.
    """
    X = X.copy()
    hip_mid      = (X[:, :, _J_LEFT_HIP,  :] + X[:, :, _J_RIGHT_HIP,  :]) / 2  # (N,20,2)
    shoulder_mid = (X[:, :, _J_LEFT_SHOULDER, :] + X[:, :, _J_RIGHT_SHOULDER, :]) / 2

    torso_per_frame = np.linalg.norm(shoulder_mid - hip_mid, axis=-1, keepdims=True)  # (N,20,1)
    torso_scale     = torso_per_frame.mean(axis=1, keepdims=True)                     # (N,1,1)
    torso_scale     = np.where(torso_scale < 1e-6, 1.0, torso_scale)

    X -= hip_mid[:, :, np.newaxis, :]          # translate
    X /= torso_scale[:, :, np.newaxis, :]      # scale
    return X.astype(np.float32)


def _encode_labels(y_str: np.ndarray) -> np.ndarray:
    """Map label strings to class indices, raising on unknown labels."""
    out = np.empty(len(y_str), dtype=np.int64)
    for i, lbl in enumerate(y_str):
        if lbl not in CLASS_TO_IDX:
            raise ValueError(
                f"Unknown label '{lbl}'.  Known classes: {CLASSES}\n"
                "Check _normalize_label() in preprocess.py."
            )
        out[i] = CLASS_TO_IDX[lbl]
    return out


def _tensorize_for_gru(X_win: np.ndarray, features: str) -> tuple[np.ndarray, int, int]:
    """
    Build (N, T, F) tensors for PunchNet.

    ``skeleton``: hip-centred torso-scaled joints (original pipeline).
    Other branches apply ``paper_features`` encodings on raw pixel windows (paper Eq. 5
    centering is inside each encoder).
    """
    f = features.lower().strip()
    if f == "skeleton":
        Xn = _normalise_skeleton(X_win)
        X = Xn.reshape(len(Xn), 20, 24)
        return X, 20, 24
    return encode_branch(X_win, f)


def _make_class_weights(y_train: np.ndarray, mode: str) -> torch.Tensor | None:
    """
    Weights for CrossEntropyLoss to counter label frequency skew.

    ``inverse``: w_c ∝ N / (K * n_c), renormalised to sum to K (sklearn-style balance).
    ``sqrt``:    softer upweighting via sqrt of the inverse-frequency weights.
    """
    if mode == "none":
        return None
    counts = np.bincount(y_train, minlength=len(CLASSES)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n = len(y_train)
    k = float(len(CLASSES))
    w = n / (k * counts)
    if mode == "sqrt":
        w = np.sqrt(w)
    w = w * (k / w.sum())
    return torch.tensor(w, dtype=torch.float32)


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    """Wraps (N, T, F) float arrays for sequence models."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── Model ──────────────────────────────────────────────────────────────────────

class PunchNet(nn.Module):
    """
    Sequence classifier over per-frame feature vectors.

    Stack: per-frame linear embed + LayerNorm + GELU → stacked bidirectional GRU
    → learned attention pooling over time → two-hidden-layer MLP head.

    Input : (batch, T, F) — branch-dependent feature dim F and length T (20 or 16)
    Output: (batch, n_classes) — logits
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int = 160,
        num_layers:  int = 3,
        n_classes:   int = len(CLASSES),
        dropout:     float = 0.35,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.frame_embed = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_dim = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(gru_dim, gru_dim // 2),
            nn.Tanh(),
            nn.Linear(gru_dim // 2, 1),
        )
        mid = max(64, hidden_size // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(gru_dim),
            nn.Linear(gru_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.frame_embed(x)
        h, _ = self.gru(h)
        att = torch.softmax(self.attention(h), dim=1)
        pooled = (h * att).sum(dim=1)
        return self.head(pooled)

    def config_dict(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "n_classes": len(CLASSES),
        }


# ── Training utilities ─────────────────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        preds  = logits.argmax(1)
        total_loss += loss.item() * len(y_batch)
        correct    += (preds == y_batch).sum().item()
        n          += len(y_batch)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
    return (
        total_loss / n, correct / n,
        np.concatenate(all_preds),
        np.concatenate(all_targets),
    )


def _train_gru_fold(
    args: argparse.Namespace,
    device: torch.device,
    X_tr: np.ndarray,
    X_va: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    fold_name: str,
    features: str,
    seq_len: int,
    input_dim: int,
    class_weights: torch.Tensor | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one PunchNet branch for one fold; save checkpoint; return best-val predictions."""
    train_ds = SeqDataset(X_tr, y_tr)
    val_ds = SeqDataset(X_va, y_va)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = PunchNet(
        input_size=input_dim,
        hidden_size=args.hidden,
        num_layers=args.gru_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"   [{features}] PunchNet ~{n_params / 1e6:.2f}M params  in={input_dim}  T={seq_len}  "
        f"hidden={args.hidden}  BiGRU layers={args.gru_layers}"
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    w_device = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=w_device)

    best_val_acc = 0.0
    best_state = None
    best_preds = best_targets = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = _train_epoch(model, train_dl, optimiser, criterion, device)
        va_loss, va_acc, preds, targets = _eval_epoch(model, val_dl, criterion, device)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds, best_targets = preds, targets

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(
                f"   [{features}] ep {epoch:3d}/{args.epochs}  "
                f"train loss {tr_loss:.4f}  acc {tr_acc:.3f}  |  "
                f"val loss {va_loss:.4f}  acc {va_acc:.3f}  "
                f"{'*' if va_acc == best_val_acc else ''}"
            )

    assert best_state is not None and best_preds is not None and best_targets is not None
    print(f"\n   [{features}] Best val acc: {best_val_acc:.4f}")

    ckpt_dir = _REPO / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"punch_net_{features}_{fold_name}.pt"
    torch.save(
        {
            "model_state": best_state,
            "classes": CLASSES,
            "architecture": "PunchNet",
            "model_cfg": {
                "features": features,
                "seq_len": seq_len,
                "input_size": input_dim,
                "hidden_size": args.hidden,
                "num_layers": args.gru_layers,
                "dropout": args.dropout,
            },
        },
        ckpt_path,
    )
    print(f"   [{features}] Checkpoint → {ckpt_path}")

    return best_preds, best_targets


# ── Split helpers ──────────────────────────────────────────────────────────────

def _random_split(
    X: np.ndarray, y: np.ndarray, versions: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(sss.split(X, y))
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def _lovo_splits(
    X: np.ndarray, y: np.ndarray, versions: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Yield (X_train, X_val, y_train, y_val, held_out_version) for each video."""
    splits = []
    for ver in sorted(set(versions)):
        mask    = versions == ver
        splits.append((X[~mask], X[mask], y[~mask], y[mask], ver))
    return splits


def _make_fae_pipeline() -> Pipeline:
    """Standardise features then KNN K=4 with distance weighting (math.md Steps 11–12)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=4,
                    weights="distance",
                    metric="euclidean",
                ),
            ),
        ]
    )


def main_fae(args: argparse.Namespace, npz_path: Path) -> None:
    print("Loading dataset (pixel-space windows for FAE) …")
    X_win, y, versions = load_dataset(npz_path, normalize=False)
    X = encode_fae(X_win)
    print(f"  {len(X)} clips — FAE shape {X.shape[1:]} — classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    if args.split == "cv10":
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        fold_accs: list[float] = []
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            clf = _make_fae_pipeline()
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            acc = float((pred == y[test_idx]).mean())
            fold_accs.append(acc)
            all_preds.append(pred)
            all_targets.append(y[test_idx])
            print(f"  Fold {fold_idx}/10  accuracy = {acc:.4f}")

        mu = float(np.mean(fold_accs))
        sigma = float(np.std(fold_accs, ddof=0))
        print(f"\n10-fold stratified CV accuracy: μ = {mu:.4f}  σ = {sigma:.4f}  ({mu:.4f} ± {sigma:.4f})")

        all_preds_arr = np.concatenate(all_preds)
        all_targets_arr = np.concatenate(all_targets)
        print("\n" + "═" * 60)
        print("Classification report (out-of-fold predictions):")
        print(classification_report(all_targets_arr, all_preds_arr, target_names=CLASSES))
        print("Confusion matrix:")
        print(confusion_matrix(all_targets_arr, all_preds_arr))
        cm_path = _FIGURES_DIR / "confusion_fae_cv10.png"
        _save_confusion_matrix_figure(
            all_targets_arr,
            all_preds_arr,
            out_path=cm_path,
            title="FAE + KNN — 10-fold stratified CV (pooled out-of-fold predictions)",
        )
        print(f"Confusion matrix figure → {cm_path}")

    else:
        if args.split == "lovo":
            folds = _lovo_splits(X, y, versions)
        else:
            X_tr, X_va, y_tr, y_va = _random_split(X, y, versions, args.val_size, args.seed)
            folds = [(X_tr, X_va, y_tr, y_va, "random")]

        all_preds, all_targets = [], []

        for fold_idx, (X_tr, X_va, y_tr, y_va, fold_name) in enumerate(folds, 1):
            print(f"\n── Fold {fold_idx}/{len(folds)} (held-out: {fold_name}) ──")
            clf = _make_fae_pipeline()
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_va)
            acc = float((pred == y_va).mean())
            print(f"   train: {len(y_tr)}   val: {len(y_va)}   accuracy: {acc:.4f}")
            all_preds.append(pred)
            all_targets.append(y_va)

            ckpt_dir = _REPO / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            out_path = ckpt_dir / f"punch_fae_knn_{fold_name}.joblib"
            joblib.dump({"pipeline": clf, "classes": CLASSES}, out_path)
            print(f"  Checkpoint → {out_path}")

        all_preds_arr = np.concatenate(all_preds)
        all_targets_arr = np.concatenate(all_targets)
        print("\n" + "═" * 60)
        print("Classification report:")
        print(classification_report(all_targets_arr, all_preds_arr, target_names=CLASSES))
        print("Confusion matrix:")
        print(confusion_matrix(all_targets_arr, all_preds_arr))
        cm_path = _FIGURES_DIR / f"confusion_fae_{args.split}.png"
        _save_confusion_matrix_figure(
            all_targets_arr,
            all_preds_arr,
            out_path=cm_path,
            title=f"FAE + KNN — split={args.split}",
        )
        print(f"Confusion matrix figure → {cm_path}")

    if args.save_fae:
        clf_full = _make_fae_pipeline()
        clf_full.fit(X, y)
        out = Path(args.save_fae)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": clf_full, "classes": CLASSES}, out)
        print(f"\nFinal FAE+KNN model (trained on all {len(X)} clips) → {out}")


def main_gru(args: argparse.Namespace, npz_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataset …")
    X_win, y, versions = load_dataset(npz_path, normalize=False)
    X, seq_len, input_dim = _tensorize_for_gru(X_win, args.features)
    print(
        f"  {len(X)} clips — features={args.features}  shape (N,T,F)=({len(X)},{seq_len},{input_dim}) "
        f"— classes: {dict(zip(*np.unique(y, return_counts=True)))}"
    )

    if args.split == "lovo":
        folds = _lovo_splits(X, y, versions)
    else:
        X_tr, X_va, y_tr, y_va = _random_split(X, y, versions, args.val_size, args.seed)
        folds = [(X_tr, X_va, y_tr, y_va, "random")]

    all_preds, all_targets = [], []

    for fold_idx, (X_tr, X_va, y_tr, y_va, fold_name) in enumerate(folds, 1):
        print(f"\n── Fold {fold_idx}/{len(folds)} (held-out: {fold_name}) ──")
        print(f"   train: {len(y_tr)}   val: {len(y_va)}")
        cw = _make_class_weights(y_tr, args.class_weight)
        if cw is not None:
            print(f"   class-weight={args.class_weight} (loss rebalancing on train counts)")

        preds, targets = _train_gru_fold(
            args,
            device,
            X_tr,
            X_va,
            y_tr,
            y_va,
            fold_name,
            args.features,
            seq_len,
            input_dim,
            cw,
        )
        all_preds.append(preds)
        all_targets.append(targets)

    all_preds_arr = np.concatenate(all_preds)
    all_targets_arr = np.concatenate(all_targets)
    print("\n" + "═" * 60)
    print("Classification report (best-epoch predictions per fold):")
    print(classification_report(all_targets_arr, all_preds_arr, target_names=CLASSES))
    print("Confusion matrix:")
    print(confusion_matrix(all_targets_arr, all_preds_arr))
    cm_path = _FIGURES_DIR / f"confusion_gru_{args.features}_{args.split}.png"
    _save_confusion_matrix_figure(
        all_targets_arr,
        all_preds_arr,
        out_path=cm_path,
        title=f"PunchNet — features={args.features} — split={args.split}",
    )
    print(f"Confusion matrix figure → {cm_path}")


def main(args: argparse.Namespace) -> None:
    npz_path = Path(args.data)
    if args.paradigm == "fae":
        main_fae(args, npz_path)
    else:
        main_gru(args, npz_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paradigm",
        choices=["gru", "fae"],
        default="gru",
        help="gru: PunchNet; fae: FAE features + KNN (math.md)",
    )
    ap.add_argument("--data", default=str(_DATA_NPZ), help="path to landmarks.npz")
    ap.add_argument(
        "--features",
        choices=["skeleton", "uae", "2dmdd", "fae"],
        default="skeleton",
        help="GRU only: skeleton joints vs paper UAE / 2DMDD / FAE timestep encoding (ignored for --paradigm fae).",
    )
    ap.add_argument(
        "--class-weight",
        choices=["none", "inverse", "sqrt"],
        default="none",
        help="CrossEntropy loss weights by inverse train frequency (sqrt = softer); reduces majority-class bias",
    )
    ap.add_argument(
        "--split",
        choices=["random", "lovo", "cv10"],
        default="random",
        help="random: stratified 80/20; lovo: leave-one-video-out; cv10: 10-fold stratified (FAE only)",
    )
    ap.add_argument("--val-size", type=float, default=0.2, help="val fraction for random split")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=160, help="GRU hidden size (per direction)")
    ap.add_argument(
        "--gru-layers",
        type=int,
        default=3,
        metavar="N",
        help="number of stacked BiGRU layers",
    )
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save-fae",
        default="",
        help="If set (FAE only), save Pipeline trained on all clips to this path (joblib)",
    )
    args = ap.parse_args()

    if args.split == "cv10" and args.paradigm != "fae":
        raise SystemExit("Use --paradigm fae with --split cv10 (10-fold CV is for the FAE+KNN pipeline).")
    if args.save_fae and args.paradigm != "fae":
        raise SystemExit("--save-fae applies only to --paradigm fae.")

    main(args)
