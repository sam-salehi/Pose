# Train PunchDetectorTransformer on MotionBERT 3D skeletons (H36M-17, xyz)
#
# Data: Dataset/MotionBERT_3d/V*/X3D.npy — (total_frames, 17, 3) per video
# Labels: Annotation_files/*.xlsx — annotated punch intervals are "positive"
#
# Window labeling: a sliding window is positive if its centre frame falls inside
# any annotated punch interval (1-based inclusive → 0-based).
#
# Preprocessing per window (same as punch_classifier.py):
#   1. Translate pelvis (joint 0) to origin
#   2. Rotate into boxer's body frame (+x right, +z up, +y forward) from frame 0
#   3. Scale by torso length (pelvis → thorax in frame 0)
#
# Train/val split: video-level (whole videos held out) — no leakage from
# overlapping adjacent windows, same strategy as train_detection.py.

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from punch_transformer import PunchDetectorTransformer
from preprocess import _load_annotations, _normalize_label

_REPO           = Path.cwd().resolve()
_MOTIONBERT_DIR = _REPO / "Dataset" / "MotionBERT_3d"
_ANNOTATION_DIR = _REPO / "Dataset" / "Annotation_files"

DET_WINDOW    = 16     # frames per sliding window
STRIDE        = 2      # stride between window starts during dataset build
EPOCHS        = 60
BATCH_SIZE    = 128
LR            = 1e-3
LR_MIN        = 1e-5
WARMUP_EPOCHS = 5
VAL_FRAC      = 0.15   # fraction of *videos* held out (not windows)
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# H36M-17 joint indices
_J_PELVIS     = 0
_J_THORAX     = 8
_J_L_SHOULDER = 11
_J_R_SHOULDER = 14


# =============================================================================
# Preprocessing (mirrors punch_classifier.py / train_3d_classifier.py)
# =============================================================================


def _to_body_frame(poses: np.ndarray) -> np.ndarray:
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


def _preprocess_window(win: np.ndarray) -> np.ndarray:
    """Body-frame + torso-scale + velocity channels for a single (T, 17, 3) window."""
    q = _to_body_frame(win.astype(np.float64))
    q = _scale_normalize(q)
    q = q.astype(np.float32)
    vel = np.zeros_like(q)
    vel[1:] = q[1:] - q[:-1]          # frame-to-frame joint velocity; zero at t=0
    return np.nan_to_num(
        np.concatenate([q, vel], axis=-1),   # (T, 17, 6): xyz + dxyz
        nan=0.0, posinf=0.0, neginf=0.0,
    )


# =============================================================================
# Dataset
# =============================================================================


class Detection3DDataset(Dataset):
    """
    Pre-built detection windows for one video.

    windows : (N, T, 17, 3) float32 — preprocessed
    labels  : (N,)           float32 — 1.0 = punch, 0.0 = no-punch
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.windows = windows
        self.labels  = labels.astype(np.float32)

    @property
    def y(self) -> np.ndarray:
        return self.labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.windows[i]),
            torch.tensor(self.labels[i], dtype=torch.float32),
        )


# =============================================================================
# Window builder
# =============================================================================


def _build_video_dataset(
    video_frames: np.ndarray,
    annotations: list[tuple[int, int, str]],
    window: int,
    stride: int,
) -> Detection3DDataset:
    """
    Slide a window of length ``window`` over ``video_frames`` with step ``stride``.

    Label = 1 if the window's centre frame (0-based) lies inside any
    annotated punch interval.  Annotations use 1-based inclusive frame numbers.
    """
    n_frames = len(video_frames)

    # Build set of positive 0-based frame indices
    pos_frames: set[int] = set()
    for s, e, _ in annotations:
        pos_frames.update(range(s - 1, min(e, n_frames)))

    wins: list[np.ndarray] = []
    labels: list[float]    = []

    for w_start in range(0, n_frames - window + 1, stride):
        centre = w_start + window // 2
        label  = 1.0 if centre in pos_frames else 0.0
        win    = _preprocess_window(video_frames[w_start: w_start + window])
        wins.append(win)
        labels.append(label)

    if not wins:
        # Video shorter than one window — skip gracefully
        return Detection3DDataset(
            np.empty((0, window, 17, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return Detection3DDataset(
        np.stack(wins, axis=0),
        np.array(labels, dtype=np.float32),
    )


# =============================================================================
# Load all videos
# =============================================================================


def load_all_video_datasets() -> list[tuple[str, Detection3DDataset]]:
    """
    Returns list of (version_tag, Detection3DDataset) — one entry per video
    that has both X3D.npy and a matching annotation xlsx.
    """
    results: list[tuple[str, Detection3DDataset]] = []

    for ver_dir in sorted(_MOTIONBERT_DIR.iterdir()):
        npy_path = ver_dir / "X3D.npy"
        if not npy_path.exists():
            continue
        ver      = ver_dir.name.upper()
        ann_path = _ANNOTATION_DIR / f"{ver}.xlsx"
        if not ann_path.exists():
            print(f"[skip] no annotation file for {ver}")
            continue

        frames      = np.load(npy_path)             # (F, 17, 3)
        annotations = _load_annotations(ann_path)   # [(s, e, label), ...]

        ds = _build_video_dataset(frames, annotations, DET_WINDOW, STRIDE)

        n_pos = int(ds.labels.sum())
        n_neg = len(ds) - n_pos
        print(f"{ver}: {len(ds):6d} windows  pos={n_pos}  neg={n_neg}")
        results.append((ver, ds))

    if not results:
        raise SystemExit("No 3D data found — check Dataset/MotionBERT_3d/")
    return results


# =============================================================================
# Balanced sampling (mirrors train_detection.py)
# =============================================================================


def _balanced_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Equal pos/neg subsample; falls back to all indices if one class is missing."""
    pos = np.flatnonzero(labels > 0.5)
    neg = np.flatnonzero(labels <= 0.5)
    n   = min(len(pos), len(neg))
    if n == 0:
        return np.arange(len(labels), dtype=np.int64)
    pos = rng.permutation(pos)[:n]
    neg = rng.permutation(neg)[:n]
    return rng.permutation(np.concatenate([pos, neg])).astype(np.int64)


def _concat_labels(datasets: list[Detection3DDataset]) -> np.ndarray:
    return np.concatenate([ds.labels for ds in datasets])


# =============================================================================
# Training
# =============================================================================

if not _MOTIONBERT_DIR.is_dir():
    raise SystemExit(f"Missing: {_MOTIONBERT_DIR}")
if not _ANNOTATION_DIR.is_dir():
    raise SystemExit(f"Missing: {_ANNOTATION_DIR}")

print(f"Building detection windows (DET_WINDOW={DET_WINDOW}, stride={STRIDE}) …\n")
video_items = load_all_video_datasets()
n_v = len(video_items)

rng = np.random.default_rng(SEED)

if n_v < 2:
    raise SystemExit(
        "Need ≥2 videos for a video-level train/val split. "
        "Add more X3D.npy files to Dataset/MotionBERT_3d/."
    )

# Video-level train/val split — whole videos go to train or val, preventing
# leakage from adjacent overlapping windows (same approach as train_detection.py)
n_val_v    = max(1, int(round(VAL_FRAC * n_v)))
n_val_v    = min(n_val_v, n_v - 1)
perm       = rng.permutation(n_v)
val_idx    = set(perm[:n_val_v].tolist())

train_items = [(v, ds) for i, (v, ds) in enumerate(video_items) if i not in val_idx]
val_items   = [(v, ds) for i, (v, ds) in enumerate(video_items) if i in val_idx]

print(f"\nVideo-level split ({VAL_FRAC:.0%} of videos → val; no window leakage):")
print(f"  Train ({len(train_items)}): {[v for v, _ in train_items]}")
print(f"  Val   ({len(val_items)}):   {[v for v, _ in val_items]}")

# Balanced sampling within each split
train_parts = [ds for _, ds in train_items]
val_parts   = [ds for _, ds in val_items]

train_concat = ConcatDataset(train_parts)
val_concat   = ConcatDataset(val_parts)

y_train = _concat_labels(train_parts)
y_val   = _concat_labels(val_parts)

train_sel = _balanced_indices(y_train, rng)
val_sel   = _balanced_indices(y_val,   rng)

train_ds = Subset(train_concat, train_sel.tolist())
val_ds   = Subset(val_concat,   val_sel.tolist())

n_tr_pos  = int(np.sum(y_train[train_sel] > 0.5))
n_val_pos = int(np.sum(y_val[val_sel]     > 0.5))
print(f"\nTrain (balanced): punch {n_tr_pos}  no-punch {len(train_sel) - n_tr_pos}  total {len(train_ds)}")
print(f"Val   (balanced): punch {n_val_pos}  no-punch {len(val_sel) - n_val_pos}  total {len(val_ds)}")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = PunchDetectorTransformer(
    in_channels=6,
    spatial_hidden=64,
    d_model=128,
    nhead=4,
    num_layers=4,
    dim_feedforward=256,
    dropout=0.1,
).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

opt           = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
cosine_epochs = max(1, EPOCHS - WARMUP_EPOCHS)
sched = SequentialLR(
    opt,
    schedulers=[
        LinearLR(opt, start_factor=0.01, end_factor=1.0,
                 total_iters=min(WARMUP_EPOCHS, EPOCHS)),
        CosineAnnealingLR(opt, T_max=cosine_epochs, eta_min=LR_MIN),
    ],
    milestones=[min(WARMUP_EPOCHS, EPOCHS)],
)
loss_fn = torch.nn.BCELoss()


@torch.no_grad()
def evaluate(loader: DataLoader) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total, correct, n = 0.0, 0, 0
    all_p, all_t = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p    = model(x)
        total   += loss_fn(p, y).item() * y.size(0)
        pred     = (p >= 0.5).float()
        correct += (pred == y).sum().item()
        n       += y.size(0)
        all_p.append(pred.cpu())
        all_t.append(y.cpu())
    all_p = torch.cat(all_p).numpy().astype(int)
    all_t = torch.cat(all_t).numpy().astype(int)
    return total / max(n, 1), correct / max(n, 1), all_p, all_t


best_acc   = 0.0
best_state = None

print()
for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss, run_ok, run_n = 0.0, 0, 0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        p    = model(x)
        loss = loss_fn(p, y)
        loss.backward()
        opt.step()
        run_loss += loss.item() * y.size(0)
        run_ok   += ((p >= 0.5).float() == y).sum().item()
        run_n    += y.size(0)
    sched.step()
    lr_now = opt.param_groups[0]["lr"]

    va_loss, va_acc, _, _ = evaluate(val_dl)
    if va_acc > best_acc:
        best_acc   = va_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"epoch {epoch:02d}/{EPOCHS}  lr {lr_now:.2e}  "
        f"train loss {run_loss / max(run_n, 1):.4f} acc {run_ok / max(run_n, 1):.3f}  "
        f"val loss {va_loss:.4f} acc {va_acc:.3f}"
    )

# Final report on best checkpoint
if best_state is not None:
    model.load_state_dict(best_state)
_, _, vp, vt = evaluate(val_dl)

print(f"\nBest val acc: {best_acc:.3f}")
print("\nClassification report (val, best checkpoint):")
print(
    classification_report(
        vt, vp,
        target_names=["no-punch", "punch"],
        zero_division=0,
    )
)
print("Confusion matrix (rows=actual, cols=predicted):")
print("  [no-punch  punch]")
print(confusion_matrix(vt, vp))

# Save checkpoint
vers_tag = "_".join(v for v, _ in video_items)
ckpt = _REPO / "checkpoints" / f"punch_detector_3d_{vers_tag}.pt"
ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state":   best_state,
        "model_class":   "PunchDetectorTransformer",
        "window":        DET_WINDOW,
        "stride":        STRIDE,
        "in_channels":   6,
        "skeleton":      "H36M-17",
        "source":        "MotionBERT_3d",
        "preprocessing": "body_frame + torso_scale (per window)",
        "train_videos":  [v for v, _ in train_items],
        "val_videos":    [v for v, _ in val_items],
        "lr_schedule": {
            "warmup_epochs": WARMUP_EPOCHS,
            "cosine": f"CosineAnnealingLR T_max={cosine_epochs} eta_min={LR_MIN}",
        },
    },
    ckpt,
)
print(f"\nCheckpoint → {ckpt.relative_to(_REPO)}")
