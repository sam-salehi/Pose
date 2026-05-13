# Binary punch-detector variant of train_3d_classifier_bio.py.
#
# Classes (2): punch (any type → 0)  |  no_punch (→ 1)
# No-punch windows are sampled from between-annotation gaps; gap_review.json
# is NOT used.  A CLF_WINDOW-frame buffer is applied around each punch boundary
# so that near-punch ambiguous frames are excluded from the no_punch class.
#
# Per-joint channels (15) identical to the bio variant:
#   pos(3) + vel(3) + acc(3) + l_elbow_angle + r_elbow_angle +
#   hip_yaw + shoulder_yaw + xfactor + com_z

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from skeleton_constants import H36M_BONE_PAIRS, make_class_weights
from punch_transformer import PunchTransformer
from preprocess import _load_annotations, _normalize_label

try:
    from scipy.signal import savgol_filter as _savgol_fn
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_REPO             = Path.cwd().resolve()
_MOTIONBERT_DIR   = _REPO / "Dataset" / "MotionBERT_3d"
_ANNOTATION_DIR   = _REPO / "Dataset" / "Annotation_files"

USE_VERSIONS: frozenset[str] = frozenset(f"V{i}" for i in range(2, 11))

CLF_WINDOW  = 16
JITTER_RANGE = 2
EPOCHS      = 200
BATCH_SIZE  = 64
LR          = 1e-3
LR_MIN      = 1e-5
WARMUP_EPOCHS       = 5
EARLY_STOP_PATIENCE = 100
VAL_FRAC    = 0.1
SEED        = 42
GRAD_CLIP_MAX_NORM  = 1.0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PunchTransformer backbone (identical to bio variant)
MODEL_SPATIAL_HIDDEN = 96
MODEL_D_MODEL        = 128
MODEL_NHEAD          = 4
MODEL_NUM_LAYERS     = 5
MODEL_DIM_FEEDFORWARD = 256
MODEL_DROPOUT        = 0.25

# ── Classes ──────────────────────────────────────────────────────────────────
CLASSIFIER_CLASSES: list[str] = ["punch", "no_punch"]
PUNCH_IDX    = 0
NO_PUNCH_IDX = 1

# All recognised punch labels collapse to PUNCH_IDX
_PUNCH_RAW_LABELS: set[str] = {
    "Cross", "Jab", "Lead Hook", "Lead Uppercut", "Rear Hook", "Rear Uppercut",
}

# Max no_punch windows sampled from each between-punch gap.
# Increase if the dataset is heavily punch-heavy.
NO_PUNCH_SAMPLES_PER_GAP = 3

# Buffer (frames) removed from each side of a punch boundary before no_punch
# sampling, to exclude ambiguous near-punch frames.
PUNCH_BOUNDARY_BUFFER = CLF_WINDOW

# ── H36M-17 joint indices ────────────────────────────────────────────────────
_J_PELVIS     = 0
_J_R_HIP      = 1
_J_L_HIP      = 4
_J_THORAX     = 8
_J_L_SHOULDER = 11
_J_L_ELBOW    = 12
_J_L_WRIST    = 13
_J_R_SHOULDER = 14
_J_R_ELBOW    = 15
_J_R_WRIST    = 16

# ── Feature layout ───────────────────────────────────────────────────────────
N_SCALARS   = 6
IN_CHANNELS = 9 + N_SCALARS   # 15 total

_SC_L_ELBOW = 9
_SC_R_ELBOW = 10
_SC_HIP_YAW = 11
_SC_SH_YAW  = 12
_SC_XFACTOR = 13
_SC_COM_Z   = 14

# Mirror: both classes are symmetric (punch↔punch, no_punch↔no_punch)
_MIRROR_LABEL_MAP: list[int] = [PUNCH_IDX, NO_PUNCH_IDX]

# H36M-17 L/R joint swap for sagittal-plane reflection
_FLIP_JOINT_ORDER = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


# =============================================================================
# Preprocessing (identical to bio variant)
# =============================================================================

def _to_body_frame(poses: np.ndarray) -> np.ndarray:
    q = poses - poses[:, [_J_PELVIS], :]
    ref = q[0]
    x_raw  = ref[_J_R_SHOULDER] - ref[_J_L_SHOULDER]
    x_norm = np.linalg.norm(x_raw)
    if x_norm < 1e-6:
        return q
    x_hat = x_raw / x_norm
    z_raw  = ref[_J_THORAX] - ref[_J_PELVIS]
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
    torso = float(np.median(np.linalg.norm(q[:, _J_THORAX] - q[:, _J_PELVIS], axis=-1)))
    return q / torso if torso > 1e-6 else q


def _smooth(q: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    T = q.shape[0]
    if not _HAS_SCIPY or T < window:
        return q
    flat = q.reshape(T, -1)
    return _savgol_fn(flat, window_length=window, polyorder=min(polyorder, window - 1), axis=0).reshape(q.shape)


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba   = a - b
    bc   = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    return np.arccos(np.clip(np.einsum("ti,ti->t", ba, bc) / denom, -1.0, 1.0))


def _preprocess_clip(clip: np.ndarray) -> np.ndarray:
    """(T, 17, 3) → (T, 17, 15) float32."""
    q    = _to_body_frame(clip.astype(np.float64))
    q    = _scale_normalize(q)
    q_sm = _smooth(q)

    T = q_sm.shape[0]
    if T > 1:
        vel = np.gradient(q_sm, axis=0)
        acc = np.gradient(vel,  axis=0)
    else:
        vel = np.zeros_like(q_sm)
        acc = np.zeros_like(q_sm)

    l_elbow = _angle3(q_sm[:, _J_L_SHOULDER], q_sm[:, _J_L_ELBOW], q_sm[:, _J_L_WRIST])
    r_elbow = _angle3(q_sm[:, _J_R_SHOULDER], q_sm[:, _J_R_ELBOW], q_sm[:, _J_R_WRIST])

    hip_vec      = q_sm[:, _J_R_HIP]      - q_sm[:, _J_L_HIP]
    sh_vec       = q_sm[:, _J_R_SHOULDER] - q_sm[:, _J_L_SHOULDER]
    hip_yaw      = np.arctan2(hip_vec[:, 1], hip_vec[:, 0])
    shoulder_yaw = np.arctan2(sh_vec[:, 1],  sh_vec[:, 0])
    xfactor      = shoulder_yaw - hip_yaw
    com_z        = q_sm[:, :, 2].mean(axis=1)

    scalars    = np.stack([l_elbow, r_elbow, hip_yaw, shoulder_yaw, xfactor, com_z], axis=-1)
    scalars_bc = np.broadcast_to(scalars[:, np.newaxis, :], (T, 17, N_SCALARS)).copy()

    out = np.concatenate([q_sm, vel, acc, scalars_bc], axis=-1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_window_3d(seq: np.ndarray, window: int, jitter: int = 0) -> np.ndarray:
    T    = seq.shape[0]
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
    peak  = max(half, min(T - (window - half), peak))
    start = peak - half
    chunk = seq[start: start + window]
    if chunk.shape[0] < window:
        pad   = window - chunk.shape[0]
        chunk = np.concatenate([chunk, np.tile(chunk[[-1]], (pad, 1, 1))], axis=0)
    return chunk


# =============================================================================
# No-punch sampling from between-annotation gaps
# =============================================================================

def _no_punch_clips_from_gaps(
    frames: np.ndarray,
    annotations: list,
    n_total: int,
) -> list[np.ndarray]:
    """
    Sample no_punch windows from frames not covered by any punch annotation.

    Merges all punch intervals, adds PUNCH_BOUNDARY_BUFFER on each side, then
    takes up to NO_PUNCH_SAMPLES_PER_GAP evenly-spaced windows from each
    remaining gap that is at least CLF_WINDOW frames long.
    """
    # Build merged punch intervals (0-indexed, half-open)
    raw_ivs = sorted(
        (max(0, s - 1), min(e, n_total))
        for s, e, label in annotations
        if _normalize_label(str(label)) in {l.lower().replace(" ", "_") for l in _PUNCH_RAW_LABELS}
        or str(label).strip() in _PUNCH_RAW_LABELS
    )
    # Merge overlapping intervals
    merged: list[list[int]] = []
    for s, e in raw_ivs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Expand by buffer
    buffered = [
        [max(0, s - PUNCH_BOUNDARY_BUFFER), min(n_total, e + PUNCH_BOUNDARY_BUFFER)]
        for s, e in merged
    ]

    # Compute complement gaps
    gaps: list[tuple[int, int]] = []
    prev_end = 0
    for s, e in buffered:
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < n_total:
        gaps.append((prev_end, n_total))

    clips: list[np.ndarray] = []
    for g_start, g_end in gaps:
        gap_len = g_end - g_start
        if gap_len < CLF_WINDOW:
            continue
        n_fit  = gap_len // CLF_WINDOW
        n_take = min(n_fit, NO_PUNCH_SAMPLES_PER_GAP)
        if n_take == 1:
            mid = (g_start + g_end) // 2
            s   = max(g_start, min(mid - CLF_WINDOW // 2, g_end - CLF_WINDOW))
            clips.append(_preprocess_clip(frames[s: s + CLF_WINDOW].copy()))
        else:
            for s in np.linspace(g_start, g_end - CLF_WINDOW, n_take, dtype=int):
                clips.append(_preprocess_clip(frames[int(s): int(s) + CLF_WINDOW].copy()))

    return clips


# =============================================================================
# Data loading
# =============================================================================

def load_binary_clips(
    *,
    versions: frozenset[str] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Load punch + gap-sampled no_punch clips. ``versions=None`` → ``USE_VERSIONS``."""
    use     = USE_VERSIONS if versions is None else versions
    clips:  list[np.ndarray] = []
    y_list: list[int]        = []
    n_punch = 0
    n_np    = 0
    used_versions: set[str] = set()

    for ver_dir in sorted(_MOTIONBERT_DIR.iterdir()):
        if not ver_dir.is_dir():
            continue
        ver = ver_dir.name.upper()
        if ver not in use:
            continue
        npy_path = ver_dir / "X3D.npy"
        if not npy_path.exists():
            continue
        ann_path = _ANNOTATION_DIR / f"{ver}.xlsx"
        if not ann_path.exists():
            print(f"[skip] no annotation file for {ver}")
            continue

        frames      = np.load(npy_path)
        annotations = _load_annotations(ann_path)
        n_total     = frames.shape[0]

        # ── Punch clips ──────────────────────────────────────────────────────
        kept_p = 0
        for s, e, raw_label in annotations:
            if str(raw_label).strip() not in _PUNCH_RAW_LABELS:
                if _normalize_label(str(raw_label)) not in {
                    l.lower().replace(" ", "_") for l in _PUNCH_RAW_LABELS
                }:
                    continue
            s0, e0 = s - 1, min(e, n_total)
            if e0 <= s0:
                continue
            clips.append(_preprocess_clip(frames[s0:e0].copy()))
            y_list.append(PUNCH_IDX)
            kept_p += 1
            n_punch += 1

        # ── No-punch clips from gaps ─────────────────────────────────────────
        np_clips = _no_punch_clips_from_gaps(frames, annotations, n_total)
        for c in np_clips:
            clips.append(c)
            y_list.append(NO_PUNCH_IDX)
            n_np += 1

        print(f"{ver}: {kept_p} punch clips, {len(np_clips)} no_punch clips from gaps")
        used_versions.add(ver)

    if not clips:
        raise SystemExit("No clips found — check Dataset/MotionBERT_3d/ structure.")
    if n_punch == 0:
        raise SystemExit("No punch clips found.")
    if n_np == 0:
        raise SystemExit(
            "No no_punch clips extracted — all frames are within punch boundaries + buffer. "
            "Reduce PUNCH_BOUNDARY_BUFFER or NO_PUNCH_SAMPLES_PER_GAP."
        )

    print(f"\nTotal punch: {n_punch}  no_punch: {n_np}  ratio: {n_np/n_punch:.2f}")
    return clips, np.array(y_list, dtype=np.int64), sorted(used_versions)


# =============================================================================
# Dataset
# =============================================================================

class BinaryClipDataset(Dataset):
    def __init__(
        self,
        clips:   list[np.ndarray],
        y:       np.ndarray,
        window:  int,
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
            if self.augment else 0
        )
        win = _prepare_window_3d(clip, self.window, jitter=jitter)

        if self.augment and np.random.random() < 0.5:
            win = win[:, _FLIP_JOINT_ORDER, :].copy()
            win[:, :, 0] *= -1   # pos_x
            win[:, :, 3] *= -1   # vel_x
            win[:, :, 6] *= -1   # acc_x
            win[:, :, [_SC_L_ELBOW, _SC_R_ELBOW]] = win[:, :, [_SC_R_ELBOW, _SC_L_ELBOW]]
            win[:, :, _SC_HIP_YAW] *= -1
            win[:, :, _SC_SH_YAW]  *= -1
            win[:, :, _SC_XFACTOR] *= -1
            label = _MIRROR_LABEL_MAP[label]   # punch→punch, no_punch→no_punch

        return torch.from_numpy(win), torch.tensor(label, dtype=torch.long)


# =============================================================================
# Metrics
# =============================================================================

def _val_summary(vt: np.ndarray, vp: np.ndarray) -> str:
    vt  = np.asarray(vt)
    vp  = np.asarray(vp)
    bal = balanced_accuracy_score(vt, vp)
    f1p = f1_score(vt, vp, pos_label=PUNCH_IDX,    average="binary", zero_division=0)
    f1n = f1_score(vt, vp, pos_label=NO_PUNCH_IDX, average="binary", zero_division=0)
    pm  = vt == PUNCH_IDX
    nm  = vt == NO_PUNCH_IDX
    acc_p = float((vp[pm] == PUNCH_IDX).mean())    if pm.any() else float("nan")
    acc_n = float((vp[nm] == NO_PUNCH_IDX).mean()) if nm.any() else float("nan")
    return (
        f"val_bal_acc={bal:.3f}  "
        f"val_F1_punch={f1p:.3f}  val_F1_no_punch={f1n:.3f}  "
        f"val_acc_punch={acc_p:.3f}  val_acc_no_punch={acc_n:.3f}"
    )


# =============================================================================
# TTA mirror
# =============================================================================

_MIRROR_PERM = torch.tensor(_MIRROR_LABEL_MAP, dtype=torch.long)


def _tta_mirror_batch(xb: torch.Tensor) -> torch.Tensor:
    idx  = torch.tensor(_FLIP_JOINT_ORDER, device=xb.device, dtype=torch.long)
    flip = xb[:, :, idx, :].clone()
    flip[:, :, :, 0] *= -1
    flip[:, :, :, 3] *= -1
    flip[:, :, :, 6] *= -1
    tmp = flip[:, :, :, _SC_L_ELBOW].clone()
    flip[:, :, :, _SC_L_ELBOW] = flip[:, :, :, _SC_R_ELBOW]
    flip[:, :, :, _SC_R_ELBOW] = tmp
    flip[:, :, :, _SC_HIP_YAW] *= -1
    flip[:, :, :, _SC_SH_YAW]  *= -1
    flip[:, :, :, _SC_XFACTOR] *= -1
    return flip


# =============================================================================
# Training loop helpers
# =============================================================================


@torch.no_grad()
def eval_epoch(loader, model, crit):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    all_p, all_t   = [], []
    disagree_n     = 0
    sum_rel        = 0.0
    perm           = _MIRROR_PERM.to(DEVICE)

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        flip   = _tta_mirror_batch(xb)
        lo     = model(xb)
        lf_raw = model(flip)
        lf     = lf_raw[:, perm]
        logits = (lo + lf) * 0.5
        loss   = crit(logits, yb)
        tot   += loss.item() * yb.size(0)
        pred   = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n      += yb.size(0)
        all_p.append(pred.cpu())
        all_t.append(yb.cpu())
        disagree_n += (lo.argmax(dim=1) != lf.argmax(dim=1)).sum().item()
        diff   = lo - lf
        l2d    = diff.flatten(1).norm(dim=1)
        l2o    = lo.flatten(1).norm(dim=1)
        l2f    = lf.flatten(1).norm(dim=1)
        sum_rel += (l2d / (0.5 * (l2o + l2f) + 1e-8)).sum().item()

    return (
        tot / max(n, 1),
        correct / max(n, 1),
        torch.cat(all_p).numpy(),
        torch.cat(all_t).numpy(),
        disagree_n / max(n, 1),
        sum_rel / max(n, 1),
    )


def main() -> None:
    if not _MOTIONBERT_DIR.is_dir():
        raise SystemExit(f"Missing: {_MOTIONBERT_DIR}")
    if not _ANNOTATION_DIR.is_dir():
        raise SystemExit(f"Missing: {_ANNOTATION_DIR}")

    print(f"USE_VERSIONS ({len(USE_VERSIONS)}): {', '.join(sorted(USE_VERSIONS))}")
    print(f"in_channels={IN_CHANNELS}  binary: punch vs no_punch")
    print(f"scipy SG smoothing: {'enabled' if _HAS_SCIPY else 'DISABLED (install scipy)'}")
    print(f"No-punch from gaps  buffer={PUNCH_BOUNDARY_BUFFER}f  max_per_gap={NO_PUNCH_SAMPLES_PER_GAP}")
    print("Loading clips …")

    all_clips, all_y, used_versions = load_binary_clips()
    print(f"\nLoaded {len(all_y)} clips  device={DEVICE}")
    print("Class distribution:", {CLASSIFIER_CLASSES[k]: v for k, v in sorted(Counter(all_y.tolist()).items())})

    uniq, counts = np.unique(all_y, return_counts=True)
    can_stratify  = bool(np.all(counts >= 2))

    idx_train, idx_val = train_test_split(
        np.arange(len(all_y)),
        test_size=VAL_FRAC,
        random_state=SEED,
        stratify=all_y if can_stratify else None,
    )

    train_clips = [all_clips[i] for i in idx_train]
    val_clips   = [all_clips[i] for i in idx_val]

    train_ds = BinaryClipDataset(train_clips, all_y[idx_train], window=CLF_WINDOW, augment=True)
    val_ds   = BinaryClipDataset(val_clips,   all_y[idx_val],   window=CLF_WINDOW, augment=False)

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
        in_channels=IN_CHANNELS,
        edges=H36M_BONE_PAIRS,
        spatial_hidden=MODEL_SPATIAL_HIDDEN,
        d_model=MODEL_D_MODEL,
        nhead=MODEL_NHEAD,
        num_layers=MODEL_NUM_LAYERS,
        dim_feedforward=MODEL_DIM_FEEDFORWARD,
        dropout=MODEL_DROPOUT,
    ).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt           = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    cosine_epochs = max(1, EPOCHS - WARMUP_EPOCHS)
    sched         = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=min(WARMUP_EPOCHS, EPOCHS)),
            CosineAnnealingLR(opt, T_max=cosine_epochs, eta_min=LR_MIN),
        ],
        milestones=[min(WARMUP_EPOCHS, EPOCHS)],
    )

    best_acc    = 0.0
    best_state  = None
    patience_ctr = 0
    epochs_ran  = 0

    for epoch in range(1, EPOCHS + 1):
        epochs_ran = epoch
        model.train()
        run_loss, run_ok, run_n = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            opt.step()
            run_loss += loss.item() * yb.size(0)
            run_ok   += (logits.argmax(1) == yb).sum().item()
            run_n    += yb.size(0)
        sched.step()
        lr_now = opt.param_groups[0]["lr"]

        va_loss, va_acc, vp, vt, tta_disagree, tta_rel = eval_epoch(val_dl, model, crit)
        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        print(
            f"epoch {epoch:02d}/{EPOCHS}  lr {lr_now:.2e}  "
            f"train loss {run_loss / max(run_n, 1):.4f} acc {run_ok / max(run_n, 1):.3f}  "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}  "
            f"tta_mismatch={tta_disagree:.3f}  tta_rel={tta_rel:.4f}\n"
            f"         {_val_summary(vt, vp)}  "
            f"no_improve={patience_ctr}/{EARLY_STOP_PATIENCE}"
        )

        if patience_ctr >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stop: val acc did not improve for {EARLY_STOP_PATIENCE} epochs "
                f"(best={best_acc:.3f})."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, _, vp, vt, tta_disagree_final, tta_rel_final = eval_epoch(val_dl, model, crit)
    print(f"\nBest val acc: {best_acc:.3f}")
    print(_val_summary(vt, vp))
    print(f"Val TTA (best): argmax mismatch={tta_disagree_final:.4f}  mean rel ||Δlogit||={tta_rel_final:.4f}")
    print("\nClassification report (val, best checkpoint):")
    print(classification_report(vt, vp, target_names=["Punch", "No Punch"], zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(vt, vp))

    vers_tag = "_".join(sorted(used_versions))
    ckpt     = _REPO / "checkpoints" / f"punch_transformer_binary_bio_{vers_tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state":        best_state,
            "model_class":        "PunchTransformer",
            "punch_classes":      CLASSIFIER_CLASSES,
            "punch_idx":          PUNCH_IDX,
            "no_punch_idx":       NO_PUNCH_IDX,
            "window":             CLF_WINDOW,
            "in_channels":        IN_CHANNELS,
            "spatial_hidden":     MODEL_SPATIAL_HIDDEN,
            "d_model":            MODEL_D_MODEL,
            "nhead":              MODEL_NHEAD,
            "num_layers":         MODEL_NUM_LAYERS,
            "dim_feedforward":    MODEL_DIM_FEEDFORWARD,
            "dropout":            MODEL_DROPOUT,
            "skeleton":           "H36M-17",
            "source":             "MotionBERT_3d",
            "preprocessing": (
                f"body_frame + torso_scale + SG_smooth(w=5,p=2) + "
                f"pos(3)+vel(3)+acc(3)+scalars({N_SCALARS} broadcast)"
            ),
            "scalar_features":    ["l_elbow_angle", "r_elbow_angle", "hip_yaw", "shoulder_yaw", "xfactor", "com_z"],
            "negatives": (
                f"between-annotation gaps  buffer={PUNCH_BOUNDARY_BUFFER}f  "
                f"max_per_gap={NO_PUNCH_SAMPLES_PER_GAP}"
            ),
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "epochs_ran":         epochs_ran,
            "augmentation":       f"jitter±{JITTER_RANGE} + mirror_flip(50%)",
            "lr_schedule": {
                "warmup_epochs": WARMUP_EPOCHS,
                "warmup":        "LinearLR 0.01→1.0 × base LR",
                "cosine":        f"CosineAnnealingLR T_max={cosine_epochs} eta_min={LR_MIN}",
            },
        },
        ckpt,
    )
    print(f"Checkpoint → {ckpt.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
