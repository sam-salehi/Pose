# 4-class biomechanics classifier: jab_punch | hook | uppercut | no_punch.
#
# Collapsed from the 7-class bio variant by merging lead/rear pairs:
#   jab_punch  = jab + cross
#   hook       = lead_hook + rear_hook
#   uppercut   = lead_uppercut + rear_uppercut
#
# Per-joint channels (9): pos(3) + vel(3) + acc(3)  [Savitzky-Golay smoothed]
# Scalar features (6) broadcast to every joint:
#   l_elbow_angle, r_elbow_angle  — elbow flexion
#   hip_yaw, shoulder_yaw         — girdle orientations in body frame
#   xfactor                       — shoulder_yaw − hip_yaw (kinetic-chain separation)
#   com_z                         — mean joint z (vertical loading proxy)
# Total in_channels = 15.
#
# Mirror augmentation flips the pose but since lead/rear are merged the label
# is unchanged — it acts as a pure pose-diversity augmentation (no permutation needed).
# TTA mirror averaging still reduces prediction variance.

from __future__ import annotations

from collections import Counter
import json
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

from skeleton_constants import (
    H36M_BONE_PAIRS,
    make_class_weights,
)
from punch_transformer import PunchTransformer
from preprocess import _load_annotations, _normalize_label

try:
    from scipy.signal import savgol_filter as _savgol_fn
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_REPO = Path.cwd().resolve()
_MOTIONBERT_DIR = _REPO / "Dataset" / "MotionBERT_3d"
_ANNOTATION_DIR = _REPO / "Dataset" / "Annotation_files"
_GAP_REVIEW_JSON = _REPO / "Dataset" / "gap_labels" / "gap_review.json"

TRAIN_VERSIONS: frozenset[str] = frozenset(f"V{i}" for i in range(4, 11))

CLF_WINDOW = 16
JITTER_RANGE = 2
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
LR_MIN = 1e-5
WARMUP_EPOCHS = 5
EARLY_STOP_PATIENCE = 20
VAL_FRAC = 0.1
SEED = 42
GRAD_CLIP_MAX_NORM = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SPATIAL_HIDDEN = 96
MODEL_D_MODEL = 128
MODEL_NHEAD = 4
MODEL_NUM_LAYERS = 5
MODEL_DIM_FEEDFORWARD = 256
MODEL_DROPOUT = 0.25

# 4 output classes
CLASSIFIER_CLASSES: list[str] = ["jab_punch", "hook", "uppercut", "no_punch"]
NO_PUNCH_IDX = 3

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
N_SCALARS   = 10
IN_CHANNELS = 9 + N_SCALARS  # 19 total

# Original 6 scalars
_SC_L_ELBOW = 9
_SC_R_ELBOW = 10
_SC_HIP_YAW = 11
_SC_SH_YAW  = 12
_SC_XFACTOR = 13
_SC_COM_Z   = 14
# 4 new ANOVA-motivated scalars
_SC_L_LAT   = 15   # normalised lateral vel direction of L wrist: v_x_L / |vel_L|
_SC_R_LAT   = 16   # normalised lateral vel direction of R wrist: v_x_R / |vel_R|
_SC_BILAT   = 17   # bilateral speed asymmetry: (sp_R - sp_L) / (sp_R + sp_L)
_SC_SH_AVEL = 18   # shoulder angular velocity: d(shoulder_yaw)/dt

# Map all 6 raw punch labels to the 3 merged punch classes
_LABEL_TO_IDX: dict[str, int] = {
    "Jab":           0,  # jab_punch
    "Cross":         0,  # jab_punch
    "Lead Hook":     1,  # hook
    "Rear Hook":     1,  # hook
    "Lead Uppercut": 2,  # uppercut
    "Rear Uppercut": 2,  # uppercut
}

# Mirror (sagittal-plane flip) keeps class identity: lead/rear already merged.
# Index i → class index after flipping.
_MIRROR_LABEL_MAP: list[int] = [0, 1, 2, 3]  # identity

# H36M-17 L/R joint swap for sagittal-plane reflection
_FLIP_JOINT_ORDER = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


# =============================================================================
# Preprocessing  (identical biomechanics pipeline to the 7-class variant)
# =============================================================================

def _to_body_frame(poses: np.ndarray) -> np.ndarray:
    """Translate to pelvis origin and rotate into first-frame body axes."""
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


def _smooth(q: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smooth over time. Falls back to unsmoothed if scipy absent or clip too short."""
    T = q.shape[0]
    if not _HAS_SCIPY or T < window:
        return q
    po = min(polyorder, window - 1)
    flat = q.reshape(T, -1)
    return _savgol_fn(flat, window_length=window, polyorder=po, axis=0).reshape(q.shape)


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle (radians) at b given points a, b, c. Shapes (T, 3) → (T,)."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    cos_a = np.clip(np.einsum("ti,ti->t", ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cos_a)


def _preprocess_clip(clip: np.ndarray) -> np.ndarray:
    """
    (T, 17, 3) → (T, 17, 19) float32.
    Channel layout per joint:
      0–2   pos          (body frame, torso-normalised)
      3–5   vel          (central finite diff on smoothed pos)
      6–8   acc          (central finite diff on vel)
      9     l_elbow_angle       (broadcast scalar)
      10    r_elbow_angle       (broadcast scalar)
      11    hip_yaw             (broadcast scalar)
      12    shoulder_yaw        (broadcast scalar)
      13    xfactor             (broadcast scalar)
      14    com_z               (broadcast scalar)
      15    l_wrist_lat_vel     v_x_L / |vel_L|  — normalised lateral direction
      16    r_wrist_lat_vel     v_x_R / |vel_R|  — normalised lateral direction
      17    bilateral_asymmetry (sp_R − sp_L) / (sp_R + sp_L)
      18    shoulder_ang_vel    d(shoulder_yaw)/dt
    """
    q = _to_body_frame(clip.astype(np.float64))
    q = _scale_normalize(q)
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

    # ── 4 ANOVA-motivated scalars ──────────────────────────────────────────
    vel_L = vel[:, _J_L_WRIST, :]                              # (T, 3)
    vel_R = vel[:, _J_R_WRIST, :]
    sp_L  = np.linalg.norm(vel_L, axis=-1) + 1e-8             # (T,)
    sp_R  = np.linalg.norm(vel_R, axis=-1) + 1e-8
    # Normalised lateral direction: +1 = pure rightward, −1 = pure leftward
    l_lat = vel_L[:, 0] / sp_L
    r_lat = vel_R[:, 0] / sp_R
    # Bilateral asymmetry: +1 = right hand punching, −1 = left hand punching
    bilat = (sp_R - sp_L) / (sp_R + sp_L)
    # Shoulder angular velocity (rad/frame) — high for cross/hook, low for jab
    if T < 2:
        sh_ang_vel = np.zeros(T, dtype=np.float64)
    else:
        sh_ang_vel = np.gradient(shoulder_yaw)

    scalars = np.stack(
        [l_elbow, r_elbow, hip_yaw, shoulder_yaw, xfactor, com_z,
         l_lat, r_lat, bilat, sh_ang_vel],
        axis=-1,
    )  # (T, 10)
    scalars_bc = np.broadcast_to(
        scalars[:, np.newaxis, :], (T, 17, N_SCALARS)
    ).copy()

    out = np.concatenate([q_sm, vel, acc, scalars_bc], axis=-1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_window_3d(seq: np.ndarray, window: int, jitter: int = 0) -> np.ndarray:
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
    peak  = max(half, min(T - (window - half), peak))
    start = peak - half
    chunk = seq[start: start + window]
    if chunk.shape[0] < window:
        pad   = window - chunk.shape[0]
        chunk = np.concatenate([chunk, np.tile(chunk[[-1]], (pad, 1, 1))], axis=0)
    return chunk


# =============================================================================
# Data loading
# =============================================================================

# Raw windows per gap_review.json no_punch span — flush-left + flush-right
_NO_PUNCH_SAMPLES_PER_GAP_SPAN = 2


def _no_punch_spans_from_gap_review(
    versions: frozenset[str],
) -> list[tuple[str, int, int]]:
    """(workbook, start, end) half-open frame indices for each no_punch chunk in ``versions``."""
    if not _GAP_REVIEW_JSON.is_file():
        raise SystemExit(f"Missing {_GAP_REVIEW_JSON} — run label_punch_gaps.py")
    data = json.loads(_GAP_REVIEW_JSON.read_text(encoding="utf-8"))
    rows: list[tuple[str, int, int]] = []
    for e in data.get("entries", []):
        if e.get("label") != "no_punch":
            continue
        ver = str(e["workbook"]).strip().upper()
        if ver not in versions:
            continue
        s0, e0 = int(e["start"]), int(e["end"])
        if e0 > s0:
            rows.append((ver, s0, e0))
    return rows


def _two_raw_spans_for_no_punch(g0: int, g1: int, window: int) -> list[tuple[int, int]]:
    """Exactly two half-open [a, b) slices within [g0, g1)."""
    L = g1 - g0
    if L <= 0:
        return []
    if L < window:
        sp = (g0, g1)
        return [sp, sp]
    last_start = g1 - window
    first_start = g0
    if last_start <= first_start:
        sp = (g0, g0 + window)
        return [sp, sp]
    return [(first_start, first_start + window), (last_start, g1)]


def load_3d_clips_with_negatives(
    versions: frozenset[str],
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """
    Punch clips from xlsx annotations merged into 3 punch classes,
    plus no_punch from gap_review.json, restricted to ``versions``.
    """
    clips: list[np.ndarray] = []
    y_list: list[int] = []
    n_punch, n_np = 0, 0
    trained_versions: set[str] = set()

    for ver_dir in sorted(_MOTIONBERT_DIR.iterdir()):
        if not ver_dir.is_dir():
            continue
        ver = ver_dir.name.upper()
        if ver not in versions:
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

        kept_p = 0
        for s, e, raw_label in annotations:
            label = _normalize_label(str(raw_label))
            if label not in _LABEL_TO_IDX:
                continue
            s0, e0 = s - 1, min(e, n_total)
            if e0 <= s0:
                continue
            clips.append(_preprocess_clip(frames[s0:e0].copy()))
            y_list.append(_LABEL_TO_IDX[label])
            kept_p += 1
            n_punch += 1

        trained_versions.add(ver)
        print(f"{ver}: punches {kept_p}/{len(annotations)} (xlsx)")

    if not clips:
        raise SystemExit("No 3D clips found — check Dataset/MotionBERT_3d/ structure.")

    spans = _no_punch_spans_from_gap_review(versions)
    if not spans:
        raise SystemExit(
            f"No no_punch entries for versions={versions} in {_GAP_REVIEW_JSON} — run label_punch_gaps.py"
        )

    cache: dict[str, np.ndarray] = {}
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
        s0 = max(0, s0)
        if e0 <= s0:
            skipped += 1
            continue
        raws = _two_raw_spans_for_no_punch(s0, e0, CLF_WINDOW)
        if len(raws) != _NO_PUNCH_SAMPLES_PER_GAP_SPAN:
            skipped += 1
            continue
        for a, b in raws:
            clips.append(_preprocess_clip(frames[a:b].copy()))
            y_list.append(NO_PUNCH_IDX)
            n_np += 1
        trained_versions.add(ver)

    print(
        f"gap_review.json: {n_np} no_punch clips from {len(spans)} labeled spans "
        f"({skipped} spans skipped — missing X3D or out of range)"
    )
    if n_np == 0:
        raise SystemExit(
            "Could not load any no_punch clips — check gap_review vs MotionBERT_3d and USE_VERSIONS."
        )

    print(f"\nTotal punch clips: {n_punch}  no_punch clips: {n_np}")
    return clips, np.array(y_list, dtype=np.int64), sorted(trained_versions)


# =============================================================================
# Dataset
# =============================================================================

class Clf3DDataset(Dataset):
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

        T_clip = clip.shape[0]
        if self.augment:
            if T_clip < self.window:
                # Randomly place the punch anywhere within the window so the model
                # sees it at all temporal positions, matching inference sliding-window.
                max_offset = self.window - T_clip
                offset = int(np.random.randint(0, max_offset + 1))
                pad_pre  = offset
                pad_post = self.window - T_clip - offset
                win = np.concatenate([
                    np.tile(clip[[0]], (pad_pre, 1, 1)),
                    clip,
                    np.tile(clip[[-1]], (pad_post, 1, 1)),
                ], axis=0)
            else:
                # Clip longer than window: jitter the center crop as before
                jitter = int(np.random.randint(-JITTER_RANGE, JITTER_RANGE + 1))
                win = _prepare_window_3d(clip, self.window, jitter=jitter)
        else:
            # Val/test: center the clip consistently
            win = _prepare_window_3d(clip, self.window, jitter=0)

        if self.augment and np.random.random() < 0.5:
            win = win[:, _FLIP_JOINT_ORDER, :].copy()
            # Negate x-components of pos, vel, acc
            win[:, :, 0] *= -1
            win[:, :, 3] *= -1
            win[:, :, 6] *= -1
            # Swap l/r elbow angles
            win[:, :, [_SC_L_ELBOW, _SC_R_ELBOW]] = win[:, :, [_SC_R_ELBOW, _SC_L_ELBOW]]
            # Negate yaw-based scalars (reflection flips chirality)
            win[:, :, _SC_HIP_YAW]  *= -1
            win[:, :, _SC_SH_YAW]   *= -1
            win[:, :, _SC_XFACTOR]  *= -1
            # com_z unsigned — unchanged
            # New scalars: swap L/R lateral vel dirs (and negate — x flips sign)
            tmp = win[:, :, _SC_L_LAT].copy()
            win[:, :, _SC_L_LAT]  = -win[:, :, _SC_R_LAT]
            win[:, :, _SC_R_LAT]  = -tmp
            win[:, :, _SC_BILAT]  *= -1   # (sp_R−sp_L) becomes (sp_L−sp_R)
            win[:, :, _SC_SH_AVEL] *= -1  # shoulder rotation reverses chirality
            # Label unchanged: lead/rear merged, mirroring stays in same class
            label = _MIRROR_LABEL_MAP[label]

        return torch.from_numpy(win), torch.tensor(label, dtype=torch.long)


# =============================================================================
# Metrics helpers
# =============================================================================

def _extended_val_summary(vt: np.ndarray, vp: np.ndarray) -> str:
    vt  = np.asarray(vt)
    vp  = np.asarray(vp)
    bal = balanced_accuracy_score(vt, vp)
    f4  = f1_score(vt, vp, average="macro", zero_division=0)
    f3  = f1_score(vt, vp, average="macro", labels=[0, 1, 2], zero_division=0)
    pm  = vt != NO_PUNCH_IDX
    acc_on_punch = float((vp[pm] == vt[pm]).mean()) if np.any(pm) else float("nan")
    return (
        f"val_bal_acc={bal:.3f}  val_macroF1_4cls={f4:.3f}  "
        f"val_macroF1_3punch={f3:.3f}  val_acc_true_punch={acc_on_punch:.3f}"
    )


def _display_class_name(c: str) -> str:
    return {
        "jab_punch": "Jab / Punch",
        "hook":      "Hook",
        "uppercut":  "Uppercut",
        "no_punch":  "No Punch",
    }.get(c, c.replace("_", " ").title())


# =============================================================================
# TTA mirror (label map is identity for 4 classes, but averaging still helps)
# =============================================================================

_MIRROR_PERM = torch.tensor(_MIRROR_LABEL_MAP, dtype=torch.long)


def _tta_mirror_batch(xb: torch.Tensor) -> torch.Tensor:
    """Return the sagittal-plane mirror of a batch (N, T, V, C)."""
    idx  = torch.tensor(_FLIP_JOINT_ORDER, device=xb.device, dtype=torch.long)
    flip = xb[:, :, idx, :].clone()
    flip[:, :, :, 0] *= -1   # pos_x
    flip[:, :, :, 3] *= -1   # vel_x
    flip[:, :, :, 6] *= -1   # acc_x
    tmp = flip[:, :, :, _SC_L_ELBOW].clone()
    flip[:, :, :, _SC_L_ELBOW] = flip[:, :, :, _SC_R_ELBOW]
    flip[:, :, :, _SC_R_ELBOW] = tmp
    flip[:, :, :, _SC_HIP_YAW]  *= -1
    flip[:, :, :, _SC_SH_YAW]   *= -1
    flip[:, :, :, _SC_XFACTOR]  *= -1
    tmp = flip[:, :, :, _SC_L_LAT].clone()
    flip[:, :, :, _SC_L_LAT]   = -flip[:, :, :, _SC_R_LAT]
    flip[:, :, :, _SC_R_LAT]   = -tmp
    flip[:, :, :, _SC_BILAT]   *= -1
    flip[:, :, :, _SC_SH_AVEL] *= -1
    return flip


# =============================================================================
# Training
# =============================================================================

if not _MOTIONBERT_DIR.is_dir():
    raise SystemExit(f"Missing: {_MOTIONBERT_DIR}")
if not _ANNOTATION_DIR.is_dir():
    raise SystemExit(f"Missing: {_ANNOTATION_DIR}")
if not TRAIN_VERSIONS:
    raise SystemExit("TRAIN_VERSIONS is empty.")

print(f"TRAIN_VERSIONS ({len(TRAIN_VERSIONS)}): {', '.join(sorted(TRAIN_VERSIONS))}")
print(f"in_channels={IN_CHANNELS} (pos+vel+acc=9 + {N_SCALARS} broadcast scalars: "
      f"elbow×2, hip_yaw, sh_yaw, xfactor, com_z, lat_vel×2, bilat_asym, sh_ang_vel)")
print(f"scipy SG smoothing: {'enabled' if _HAS_SCIPY else 'DISABLED (install scipy)'}")
print(f"Classes ({len(CLASSIFIER_CLASSES)}): {CLASSIFIER_CLASSES}")
print(
    f"Loading train/val punches (xlsx) + no_punch ({_GAP_REVIEW_JSON.relative_to(_REPO)}) …"
)
all_clips, all_y, used_versions = load_3d_clips_with_negatives(TRAIN_VERSIONS)
print(f"\nLoaded {len(all_y)} train/val clips  device={DEVICE}")
print(
    "Train/val class distribution:",
    {CLASSIFIER_CLASSES[k]: v for k, v in sorted(Counter(all_y.tolist()).items())},
)

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
print(f"Gradient clipping: max_norm={GRAD_CLIP_MAX_NORM}")

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


@torch.no_grad()
def eval_epoch(loader):
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


best_acc   = 0.0
best_state = None
patience_ctr = 0
epochs_ran = 0

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

    va_loss, va_acc, vp, vt, tta_disagree, tta_rel = eval_epoch(val_dl)
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
        f"tta_argmax_mismatch={tta_disagree:.3f}  tta_rel_logit_gap={tta_rel:.4f}\n"
        f"         {_extended_val_summary(vt, vp)}  "
        f"no_val_improve={patience_ctr}/{EARLY_STOP_PATIENCE}"
    )

    if patience_ctr >= EARLY_STOP_PATIENCE:
        print(
            f"\nEarly stop: val acc did not improve for {EARLY_STOP_PATIENCE} epochs "
            f"(best val acc={best_acc:.3f} at an earlier epoch)."
        )
        break

if best_state is not None:
    model.load_state_dict(best_state)

# ── Validation report (best checkpoint) ─────────────────────────────────────
_, _, vp, vt, tta_disagree_final, tta_rel_final = eval_epoch(val_dl)
print(f"\nBest val acc: {best_acc:.3f}")
print(_extended_val_summary(vt, vp))
print(
    "macroF1_3punch: macro-averaged F1 over three punch labels. "
    "acc_true_punch: accuracy restricted to ground-truth punch samples."
)
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
print("Confusion matrix (val):\n", confusion_matrix(vt, vp))

vers_tag = "_".join(sorted(used_versions))
ckpt     = _REPO / "checkpoints" / f"punch_transformer_4cls_bio_{vers_tag}.pt"
ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state":       best_state,
        "model_class":       "PunchTransformer",
        "punch_classes":     CLASSIFIER_CLASSES,
        "no_punch_index":    NO_PUNCH_IDX,
        "label_map":         _LABEL_TO_IDX,
        "window":            CLF_WINDOW,
        "in_channels":       IN_CHANNELS,
        "spatial_hidden":    MODEL_SPATIAL_HIDDEN,
        "d_model":           MODEL_D_MODEL,
        "nhead":             MODEL_NHEAD,
        "num_layers":        MODEL_NUM_LAYERS,
        "dim_feedforward":   MODEL_DIM_FEEDFORWARD,
        "dropout":           MODEL_DROPOUT,
        "skeleton":          "H36M-17",
        "source":            "MotionBERT_3d",
        "preprocessing": (
            f"body_frame + torso_scale + SG_smooth(w=5,p=2) + "
            f"pos(3)+vel(3)+acc(3)+scalars({N_SCALARS} broadcast)"
        ),
        "scalar_features": [
            "l_elbow_angle", "r_elbow_angle",
            "hip_yaw", "shoulder_yaw", "xfactor", "com_z",
            "l_wrist_lat_vel_norm", "r_wrist_lat_vel_norm",
            "bilateral_asymmetry", "shoulder_ang_vel",
        ],
        "negatives": (
            f"gap_review.json (label=no_punch) @ {_GAP_REVIEW_JSON.relative_to(_REPO)}; "
            f"{_NO_PUNCH_SAMPLES_PER_GAP_SPAN} raw windows per span (flush-left/right)"
        ),
        "train_versions":      sorted(TRAIN_VERSIONS),
        "grad_clip_max_norm":  GRAD_CLIP_MAX_NORM,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "epochs_ran":          epochs_ran,
        "augmentation": (
            f"random_temporal_placement(clip<window) or jitter±{JITTER_RANGE}(clip≥window) "
            f"+ mirror_flip(50%, pose diversity — label unchanged)"
        ),
        "lr_schedule": {
            "warmup_epochs": WARMUP_EPOCHS,
            "warmup":        "LinearLR 0.01→1.0 × base LR",
            "cosine":        f"CosineAnnealingLR T_max={cosine_epochs} eta_min={LR_MIN}",
        },
        "class_design": (
            "4-class: jab_punch=(jab+cross), hook=(lead+rear hook), "
            "uppercut=(lead+rear uppercut), no_punch"
        ),
    },
    ckpt,
)
print(f"Checkpoint → {ckpt.relative_to(_REPO)}")
