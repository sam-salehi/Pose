# Clone of train_3d_classifier_no_punch.py with Neville polynomial interpolation added
# as a training augmentation.
#
# Neville upsampling (per-joint, local sliding window):
#   - 11th-order polynomial fit through 12 nearest original frames
#   - Factor-3 densification (inserts 2 synthetic frames between every real pair)
#   - Runge trim: drop NEVILLE_RUNGE_TRIM frames from each end of the upsampled
#     sequence to avoid polynomial oscillation near the boundaries
#   - Applied to raw (T, 17, 3) frames BEFORE body-frame rotation and torso scaling,
#     so velocities are computed on the smooth dense trajectory
#   - Applied stochastically (NEVILLE_PROB) during training; skipped at validation
#
# All other details (gap mining, augmentations, TTA, LR schedule) identical to
# train_3d_classifier_no_punch.py.

from __future__ import annotations

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

CLF_WINDOW = 48
JITTER_RANGE = 6
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
LR_MIN = 1e-5
WARMUP_EPOCHS = 5
VAL_FRAC = 0.1
SEED = 42
GRAD_CLIP_MAX_NORM = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neville interpolation hyperparameters
NEVILLE_ORDER = 3       # polynomial degree; requires ORDER+1 node frames
NEVILLE_FACTOR = 3       # upsample factor: inserts FACTOR-1 synthetic frames per gap
NEVILLE_RUNGE_TRIM = 2   # frames trimmed from each end after upsampling
NEVILLE_PROB = 1       # probability of applying Neville aug per training sample

CLASSIFIER_CLASSES: list[str] = [*PUNCH_CLASSES, "no_punch"]
NO_PUNCH_IDX = len(PUNCH_CLASSES)

NO_PUNCH_MIN_GAP_FRAMES = 16

_J_PELVIS = 0
_J_THORAX = 8
_J_R_SHOULDER = 14
_J_L_SHOULDER = 11

_LABEL_TO_IDX: dict[str, int] = {
    "Cross":         PUNCH_CLASSES.index("cross"),
    "Jab":           PUNCH_CLASSES.index("jab"),
    "Lead Hook":     PUNCH_CLASSES.index("lead_hook"),
    "Lead Uppercut": PUNCH_CLASSES.index("lead_uppercut"),
    "Rear Hook":     PUNCH_CLASSES.index("rear_hook"),
    "Rear Uppercut": PUNCH_CLASSES.index("rear_uppercut"),
}

_FLIP_JOINT_ORDER = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


# =============================================================================
# Interval helpers (unchanged)
# =============================================================================


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out: list[tuple[int, int]] = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def _gaps_from_busy(busy: list[tuple[int, int]], n_total: int) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    cur = 0
    for a, b in busy:
        a = max(0, min(a, n_total))
        b = max(0, min(b, n_total))
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if cur < n_total:
        gaps.append((cur, n_total))
    return gaps


def _no_punch_gap_raw_spans(g0: int, g1: int, window: int, stride: int) -> list[tuple[int, int]]:
    L = g1 - g0
    if L < window:
        return [(g0, g1)]
    spans: list[tuple[int, int]] = []
    s = g0
    while s + window <= g1:
        spans.append((s, s + window))
        s += stride
    tail_start = g1 - window
    if not spans:
        return [(tail_start, g1)]
    if spans[-1][0] < tail_start:
        spans.append((tail_start, g1))
    return spans


def _busy_intervals_from_xlsx(annotations: list[tuple[int, int, str]], n_total: int) -> list[tuple[int, int]]:
    raw: list[tuple[int, int]] = []
    for s, e, _raw_label in annotations:
        s0, e0 = s - 1, min(e, n_total)
        if e0 > s0:
            raw.append((s0, e0))
    return _merge_intervals(raw)


# =============================================================================
# Neville polynomial interpolation
# =============================================================================


def _neville_upsample(
    frames: np.ndarray,
    order: int = NEVILLE_ORDER,
    factor: int = NEVILLE_FACTOR,
    runge_trim: int = NEVILLE_RUNGE_TRIM,
) -> np.ndarray:
    """
    Upsample a (T, J, C) skeleton sequence using local Neville polynomial interpolation.

    For each query time a degree-`order` polynomial is fitted through the `order+1`
    nearest original frames (sliding window, centred on the query).  `runge_trim`
    frames are dropped from each end of the output to suppress Runge oscillation
    near the sequence boundaries.

    Returns (T', J, C) float32 where T' = factor*(T-1)+1 - 2*runge_trim  (≥1).
    If T < 2 the input is returned unchanged.
    """
    T, J, C = frames.shape
    if T < 2:
        return frames.astype(np.float32)

    n_nodes = order + 1
    orig_t = np.arange(T, dtype=np.float64)
    data = frames.astype(np.float64)

    # Build dense query times: factor-1 steps inside every consecutive pair.
    out_t: list[float] = []
    for i in range(T - 1):
        for k in range(factor):
            out_t.append(i + k / factor)
    out_t.append(float(T - 1))
    out_t_arr = np.array(out_t, dtype=np.float64)
    N = len(out_t_arr)

    result = np.empty((N, J, C), dtype=np.float64)

    for qi, t in enumerate(out_t_arr):
        # Select n_nodes nearest frames, centred on round(t).
        center = int(round(t))
        half = n_nodes // 2
        lo = max(0, center - half)
        hi = lo + n_nodes
        if hi > T:
            hi = T
            lo = max(0, hi - n_nodes)

        t_win = orig_t[lo:hi]       # (k,)
        Q = data[lo:hi].copy()      # (k, J, C) — Neville tableau, updated in-place
        k = len(t_win)

        # Neville recursion (vectorised over J and C simultaneously).
        for j in range(1, k):
            for i in range(k - 1, j - 1, -1):
                denom = t_win[i] - t_win[i - j]
                if abs(denom) < 1e-12:
                    continue
                Q[i] = (
                    (t - t_win[i - j]) * Q[i] - (t - t_win[i]) * Q[i - 1]
                ) / denom

        result[qi] = Q[-1]

    # Runge trim: discard boundary frames where the polynomial is least stable.
    if runge_trim > 0 and N > 2 * runge_trim:
        result = result[runge_trim: N - runge_trim]

    return result.astype(np.float32)


# =============================================================================
# Preprocessing — body frame + torso scale + velocities
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
    torso_lengths = np.linalg.norm(q[:, _J_THORAX] - q[:, _J_PELVIS], axis=-1)
    torso = float(np.median(torso_lengths))
    return q / torso if torso > 1e-6 else q


def _preprocess_clip(clip: np.ndarray) -> np.ndarray:
    """(T, 17, 3) raw xyz → (T, 17, 6) body-frame + torso-scaled xyz + velocities."""
    q = _to_body_frame(clip.astype(np.float64))
    q = _scale_normalize(q)
    q = q.astype(np.float32)
    vel = np.zeros_like(q)
    vel[1:] = q[1:] - q[:-1]
    return np.nan_to_num(
        np.concatenate([q, vel], axis=-1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def _prepare_window_3d(seq: np.ndarray, window: int, jitter: int = 0) -> np.ndarray:
    T = seq.shape[0]
    half = window // 2

    if T < window:
        pad_pre = (window - T) // 2
        pad_post = window - T - pad_pre
        seq = np.concatenate([
            np.tile(seq[[0]], (pad_pre, 1, 1)),
            seq,
            np.tile(seq[[-1]], (pad_post, 1, 1)),
        ], axis=0)
        T = window

    peak = T // 2 + jitter
    peak = max(half, min(T - (window - half), peak))
    start = peak - half
    chunk = seq[start: start + window]

    if chunk.shape[0] < window:
        pad = window - chunk.shape[0]
        chunk = np.concatenate([chunk, np.tile(chunk[[-1]], (pad, 1, 1))], axis=0)

    return chunk


# =============================================================================
# Data loading — returns RAW (T, 17, 3) clips so Neville runs before preprocessing
# =============================================================================


def load_3d_clips_raw() -> tuple[list[np.ndarray], np.ndarray]:
    """
    Returns raw (T, 17, 3) xyz clips (NOT preprocessed) so that Neville
    interpolation can be applied in __getitem__ before body-frame normalization.
    """
    clips: list[np.ndarray] = []
    y_list: list[int] = []
    n_punch, n_gap = 0, 0

    for ver_dir in sorted(_MOTIONBERT_DIR.iterdir()):
        npy_path = ver_dir / "X3D.npy"
        if not npy_path.exists():
            continue
        ver = ver_dir.name.upper()
        ann_path = _ANNOTATION_DIR / f"{ver}.xlsx"
        if not ann_path.exists():
            print(f"[skip] no annotation file for {ver}")
            continue

        frames = np.load(npy_path)           # (N, 17, 3)
        annotations = _load_annotations(ann_path)
        n_total = frames.shape[0]

        busy = _busy_intervals_from_xlsx(annotations, n_total)
        gaps = _gaps_from_busy(busy, n_total)

        kept_p = 0
        for s, e, raw_label in annotations:
            label = _normalize_label(str(raw_label))
            if label not in _LABEL_TO_IDX:
                continue
            s0, e0 = s - 1, min(e, n_total)
            if e0 <= s0:
                continue
            clips.append(frames[s0:e0].copy())   # raw (T, 17, 3)
            y_list.append(_LABEL_TO_IDX[label])
            kept_p += 1
            n_punch += 1

        kept_g = 0
        slide_stride = max(1, CLF_WINDOW // 2)
        for g0, g1 in gaps:
            if g1 - g0 < NO_PUNCH_MIN_GAP_FRAMES:
                continue
            for a, b in _no_punch_gap_raw_spans(g0, g1, CLF_WINDOW, slide_stride):
                clips.append(frames[a:b].copy())  # raw (T, 17, 3)
                y_list.append(NO_PUNCH_IDX)
                kept_g += 1
                n_gap += 1

        print(f"{ver}: punches {kept_p}/{len(annotations)}  no_punch windows {kept_g}")

    if not clips:
        raise SystemExit("No 3D clips found — check Dataset/MotionBERT_3d/ structure.")
    if n_gap == 0:
        raise SystemExit(
            "No no_punch gaps — widen annotations or lower NO_PUNCH_MIN_GAP_FRAMES."
        )

    print(f"\nTotal punch clips: {n_punch}  no_punch clips: {n_gap}")
    return clips, np.array(y_list, dtype=np.int64)


# =============================================================================
# Dataset — Neville applied in __getitem__ before preprocessing (train only)
# =============================================================================


class Clf3DDataset(Dataset):
    def __init__(
        self,
        clips: list[np.ndarray],   # raw (T, 17, 3)
        y: np.ndarray,
        window: int,
        augment: bool = False,
    ):
        self.clips = clips
        self.y = y.astype(np.int64)
        self.window = window
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        clip = self.clips[i]   # (T, 17, 3) raw
        label = int(self.y[i])

        # --- Neville upsampling (training only, stochastic) ---
        if self.augment and clip.shape[0] >= 2 and np.random.random() < NEVILLE_PROB:
            clip = _neville_upsample(clip)   # (T', 17, 3)

        # --- Preprocessing: body frame + torso scale + velocities ---
        clip = _preprocess_clip(clip)        # (T', 17, 6)

        # --- Temporal jitter + centre crop ---
        jitter = (
            int(np.random.randint(-JITTER_RANGE, JITTER_RANGE + 1))
            if self.augment
            else 0
        )
        win = _prepare_window_3d(clip, self.window, jitter=jitter)

        # --- Mirror flip (50%) ---
        if self.augment and np.random.random() < 0.5:
            win = win[:, _FLIP_JOINT_ORDER, :].copy()
            win[:, :, 0] *= -1   # negate x position
            win[:, :, 3] *= -1   # negate x velocity

        return torch.from_numpy(win), torch.tensor(label, dtype=torch.long)


def _display_class_name(c: str) -> str:
    if c == "no_punch":
        return "No Punch"
    return c.replace("_", " ").title()


# =============================================================================
# TTA helper
# =============================================================================


def _tta_mirror_batch(xb: torch.Tensor) -> torch.Tensor:
    idx = torch.tensor(_FLIP_JOINT_ORDER, device=xb.device, dtype=torch.long)
    flip = xb[:, :, idx, :].clone()
    flip[:, :, :, 0] *= -1
    flip[:, :, :, 3] *= -1
    return flip


@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    all_p, all_t = [], []
    disagree_n = 0
    sum_rel = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        flip = _tta_mirror_batch(xb)
        lo = model(xb)
        lf = model(flip)
        logits = (lo + lf) * 0.5
        loss = crit(logits, yb)
        tot += loss.item() * yb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n += yb.size(0)
        all_p.append(pred.cpu())
        all_t.append(yb.cpu())
        disagree_n += (lo.argmax(dim=1) != lf.argmax(dim=1)).sum().item()
        diff = lo - lf
        l2d = diff.flatten(1).norm(dim=1)
        l2o = lo.flatten(1).norm(dim=1)
        l2f = lf.flatten(1).norm(dim=1)
        rel = l2d / (0.5 * (l2o + l2f) + 1e-8)
        sum_rel += rel.sum().item()
    return (
        tot / max(n, 1),
        correct / max(n, 1),
        torch.cat(all_p).numpy(),
        torch.cat(all_t).numpy(),
        disagree_n / max(n, 1),
        sum_rel / max(n, 1),
    )


# =============================================================================
# Training
# =============================================================================

if not _MOTIONBERT_DIR.is_dir():
    raise SystemExit(f"Missing: {_MOTIONBERT_DIR}")
if not _ANNOTATION_DIR.is_dir():
    raise SystemExit(f"Missing: {_ANNOTATION_DIR}")

print(f"Loading raw 3D clips (+ no_punch gaps) from {_MOTIONBERT_DIR.relative_to(_REPO)} …")
print(
    f"Neville aug: order={NEVILLE_ORDER}  factor={NEVILLE_FACTOR}  "
    f"runge_trim={NEVILLE_RUNGE_TRIM}  prob={NEVILLE_PROB}"
)
all_clips, all_y = load_3d_clips_raw()
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
    dropout=0.2,
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Gradient clipping: max_norm={GRAD_CLIP_MAX_NORM}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
cosine_epochs = max(1, EPOCHS - WARMUP_EPOCHS)
sched = torch.optim.lr_scheduler.SequentialLR(
    opt,
    schedulers=[
        LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=min(WARMUP_EPOCHS, EPOCHS)),
        CosineAnnealingLR(opt, T_max=cosine_epochs, eta_min=LR_MIN),
    ],
    milestones=[min(WARMUP_EPOCHS, EPOCHS)],
)

best_acc   = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss, run_ok, run_n = 0.0, 0, 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        opt.step()
        run_loss += loss.item() * yb.size(0)
        run_ok   += (logits.argmax(1) == yb).sum().item()
        run_n    += yb.size(0)
    sched.step()
    lr_now = opt.param_groups[0]["lr"]

    va_loss, va_acc, _, _, tta_disagree, tta_rel = eval_epoch(val_dl)
    if va_acc > best_acc:
        best_acc   = va_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"epoch {epoch:02d}/{EPOCHS}  lr {lr_now:.2e}  "
        f"train loss {run_loss / max(run_n, 1):.4f} acc {run_ok / max(run_n, 1):.3f}  "
        f"val loss {va_loss:.4f} acc {va_acc:.3f}  "
        f"tta_mismatch={tta_disagree:.3f}  tta_rel={tta_rel:.4f}"
    )

if best_state is not None:
    model.load_state_dict(best_state)
_, _, vp, vt, tta_disagree_final, tta_rel_final = eval_epoch(val_dl)
print(f"\nBest val acc: {best_acc:.3f}")
print(
    f"Val TTA (best checkpoint): argmax mismatch rate={tta_disagree_final:.4f}  "
    f"mean rel ||Δlogit||={tta_rel_final:.4f}"
)
print("\nClassification report (val, best checkpoint):")
print(
    classification_report(
        vt, vp,
        target_names=[_display_class_name(c) for c in CLASSIFIER_CLASSES],
        zero_division=0,
    )
)
print("Confusion matrix:\n", confusion_matrix(vt, vp))

vers_tag = "_".join(
    v.name.upper()
    for v in sorted(_MOTIONBERT_DIR.iterdir())
    if (v / "X3D.npy").exists()
)
ckpt = _REPO / "checkpoints" / f"punch_transformer_7cls_neville_{vers_tag}.pt"
ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state":        best_state,
        "model_class":        "PunchTransformer",
        "punch_classes":      CLASSIFIER_CLASSES,
        "punch_classes_base": PUNCH_CLASSES,
        "no_punch_index":     NO_PUNCH_IDX,
        "label_map":          _LABEL_TO_IDX,
        "window":             CLF_WINDOW,
        "in_channels":        6,
        "skeleton":           "H36M-17",
        "source":             "MotionBERT_3d",
        "preprocessing":      "body_frame + torso_scale(median) + vel",
        "negatives": (
            f"gaps between xlsx intervals, min_gap={NO_PUNCH_MIN_GAP_FRAMES}, "
            f"sliding raw windows len={CLF_WINDOW} stride={max(1, CLF_WINDOW // 2)}"
        ),
        "augmentation": (
            f"Neville(order={NEVILLE_ORDER}, factor={NEVILLE_FACTOR}, "
            f"runge_trim={NEVILLE_RUNGE_TRIM}, prob={NEVILLE_PROB}) + "
            f"jitter±{JITTER_RANGE} + mirror_flip(50%)"
        ),
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "lr_schedule": {
            "warmup_epochs": WARMUP_EPOCHS,
            "warmup":  "LinearLR 0.01→1.0 × base LR",
            "cosine":  f"CosineAnnealingLR T_max={cosine_epochs} eta_min={LR_MIN}",
        },
    },
    ckpt,
)
print(f"Checkpoint → {ckpt.relative_to(_REPO)}")
