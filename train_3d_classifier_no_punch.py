# Same as train_3d_classifier.py but 7 classes: six punch types + no_punch.
# Positives: annotated punch clips (same as before).
# Negatives: entries with label == "no_punch" in Dataset/gap_labels/gap_review.json
#   (from label_punch_gaps.py). Per labeled span [start,end), take exactly two raw
#   windows (flush-left and flush-right when the span is longer than CLF_WINDOW).
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
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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

# Only these subdirs of MotionBERT_3d are loaded (name matched case-insensitively).
# Default is V1–V10; remove names from the set to skip sessions.
USE_VERSIONS: frozenset[str] = frozenset(f"V{i}" for i in range(3, 11))

CLF_WINDOW = 16
JITTER_RANGE = 2  # ± frames shifted from centre during training
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
LR_MIN = 1e-5
WARMUP_EPOCHS = 5
VAL_FRAC = 0.1
SEED = 42
GRAD_CLIP_MAX_NORM = 1.0  # global L2 clip after backward (transformer stability)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PunchTransformer backbone (larger than base 64/128/4×256 — tune here)
MODEL_SPATIAL_HIDDEN = 96
MODEL_D_MODEL = 256
MODEL_NHEAD = 8
MODEL_NUM_LAYERS = 5
MODEL_DIM_FEEDFORWARD = 512
MODEL_DROPOUT = 0.25

# Extra class index = 6
CLASSIFIER_CLASSES: list[str] = [*PUNCH_CLASSES, "no_punch"]
NO_PUNCH_IDX = len(PUNCH_CLASSES)

# Raw frame windows taken per gap_review.json ``no_punch`` span [start, end).
NO_PUNCH_SAMPLES_PER_GAP_SPAN = 2

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

# H36M-17: swap L/R when mirroring (pelvis(0) … l_wrist(13) r_shoulder(14)…)
_FLIP_JOINT_ORDER = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


def _no_punch_spans_from_gap_review() -> list[tuple[str, int, int]]:
    """``(workbook, start, end)`` half-open frame indices for each no_punch chunk (USE_VERSIONS only)."""
    if not _GAP_REVIEW_JSON.is_file():
        raise SystemExit(f"Missing {_GAP_REVIEW_JSON} — run label_punch_gaps.py")
    data = json.loads(_GAP_REVIEW_JSON.read_text(encoding="utf-8"))
    rows: list[tuple[str, int, int]] = []
    for e in data.get("entries", []):
        if e.get("label") != "no_punch":
            continue
        ver = str(e["workbook"]).strip().upper()
        if ver not in USE_VERSIONS:
            continue
        s0, e0 = int(e["start"]), int(e["end"])
        if e0 > s0:
            rows.append((ver, s0, e0))
    return rows


def _two_raw_spans_for_no_punch(g0: int, g1: int, window: int) -> list[tuple[int, int]]:
    """
    Exactly two half-open [a, b) slices into ``frames[a:b]`` within ``[g0, g1)``.

    Long spans: first window flush-left, second flush-right (same as two disjoint
    placements when ``g1 - g0 > window``). Short spans: the available span is used
    twice (padding happens in ``_prepare_window_3d``).
    """
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


# =============================================================================
# Preprocessing — body frame + torso scale (mirrors punch_classifier.py)
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
    """
    Centre-crop ``seq`` (T, 17, C) to ``window`` frames.

    ``jitter`` shifts the crop centre (clamped in-bounds). Pads by repeating edge
    frames when the clip is shorter than ``window``.
    """
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


def load_3d_clips_with_negatives() -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """
    Punch clips from annotated rows (known punch types) + ``no_punch`` from
    ``gap_review.json`` only.

    For each JSON span labeled ``no_punch``, add ``NO_PUNCH_SAMPLES_PER_GAP_SPAN``
    clips via ``_two_raw_spans_for_no_punch``.
    """
    clips: list[np.ndarray] = []
    y_list: list[int] = []

    n_punch, n_np = 0, 0
    trained_versions: set[str] = set()

    for ver_dir in sorted(_MOTIONBERT_DIR.iterdir()):
        if not ver_dir.is_dir():
            continue
        ver = ver_dir.name.upper()
        if ver not in USE_VERSIONS:
            continue
        npy_path = ver_dir / "X3D.npy"
        if not npy_path.exists():
            continue
        ann_path = _ANNOTATION_DIR / f"{ver}.xlsx"
        if not ann_path.exists():
            print(f"[skip] no annotation file for {ver}")
            continue

        frames = np.load(npy_path)
        annotations = _load_annotations(ann_path)
        n_total = frames.shape[0]

        kept_p = 0
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
            kept_p += 1
            n_punch += 1

        trained_versions.add(ver)
        print(f"{ver}: punches {kept_p}/{len(annotations)} (xlsx)")

    if not clips:
        raise SystemExit("No 3D punch clips — check Annotation_files and MotionBERT_3d.")

    spans = _no_punch_spans_from_gap_review()
    if not spans:
        raise SystemExit(
            f"No no_punch entries for USE_VERSIONS in {_GAP_REVIEW_JSON} — run label_punch_gaps.py"
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
        if len(raws) != NO_PUNCH_SAMPLES_PER_GAP_SPAN:
            skipped += 1
            continue
        for a, b in raws:
            clip = _preprocess_clip(frames[a:b].copy())
            clips.append(clip)
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


class Clf3DDataset(Dataset):
    def __init__(
        self,
        clips: list[np.ndarray],
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
        clip = self.clips[i]
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

        return torch.from_numpy(win), torch.tensor(label, dtype=torch.long)


def _extended_val_summary(vt: np.ndarray, vp: np.ndarray) -> str:
    """Metrics less dominated by majority ``no_punch`` than plain accuracy."""
    vt = np.asarray(vt)
    vp = np.asarray(vp)
    bal = balanced_accuracy_score(vt, vp)
    f7 = f1_score(vt, vp, average="macro", zero_division=0)
    f6 = f1_score(vt, vp, average="macro", labels=list(range(6)), zero_division=0)
    pm = vt != NO_PUNCH_IDX
    acc_on_punch = float((vp[pm] == vt[pm]).mean()) if np.any(pm) else float("nan")
    return (
        f"val_bal_acc={bal:.3f}  val_macroF1_7cls={f7:.3f}  "
        f"val_macroF1_6punch={f6:.3f}  val_acc_true_punch={acc_on_punch:.3f}"
    )


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
if not USE_VERSIONS:
    raise SystemExit("USE_VERSIONS is empty — add at least one session folder name.")

print(f"USE_VERSIONS ({len(USE_VERSIONS)}): {', '.join(sorted(USE_VERSIONS))}")
print(f"Loading punches (xlsx) + no_punch ({_GAP_REVIEW_JSON.relative_to(_REPO)}) …")
all_clips, all_y, trained_versions = load_3d_clips_with_negatives()
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
val_clips = [all_clips[i] for i in idx_val]

train_ds = Clf3DDataset(train_clips, all_y[idx_train], window=CLF_WINDOW, augment=True)
val_ds = Clf3DDataset(val_clips, all_y[idx_val], window=CLF_WINDOW, augment=False)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
    spatial_hidden=MODEL_SPATIAL_HIDDEN,
    d_model=MODEL_D_MODEL,
    nhead=MODEL_NHEAD,
    num_layers=MODEL_NUM_LAYERS,
    dim_feedforward=MODEL_DIM_FEEDFORWARD,
    dropout=MODEL_DROPOUT,
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


def _tta_mirror_batch(xb: torch.Tensor) -> torch.Tensor:
    """Mirror skeleton in body frame (same as train aug): swap L/R joints, negate x on pos and vel."""
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
    disagree_frac = disagree_n / max(n, 1)
    mean_rel_logit = sum_rel / max(n, 1)
    return (
        tot / max(n, 1),
        correct / max(n, 1),
        torch.cat(all_p).numpy(),
        torch.cat(all_t).numpy(),
        disagree_frac,
        mean_rel_logit,
    )


best_acc = 0.0
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
        run_ok += (logits.argmax(1) == yb).sum().item()
        run_n += yb.size(0)
    sched.step()
    lr_now = opt.param_groups[0]["lr"]

    va_loss, va_acc, vp, vt, tta_disagree, tta_rel = eval_epoch(val_dl)
    if va_acc > best_acc:
        best_acc = va_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"epoch {epoch:02d}/{EPOCHS}  lr {lr_now:.2e}  "
        f"train loss {run_loss / max(run_n, 1):.4f} acc {run_ok / max(run_n, 1):.3f}  "
        f"val loss {va_loss:.4f} acc {va_acc:.3f}  "
        f"tta_argmax_mismatch={tta_disagree:.3f}  tta_rel_logit_gap={tta_rel:.4f}\n"
        f"         {_extended_val_summary(vt, vp)}"
    )

if best_state is not None:
    model.load_state_dict(best_state)
_, _, vp, vt, tta_disagree_final, tta_rel_final = eval_epoch(val_dl)
print(f"\nBest val acc: {best_acc:.3f}")
print(_extended_val_summary(vt, vp))
print(
    "macroF1_6punch: macro-averaged F1 over the six punch labels (includes rows whose "
    "true label is no_punch when scoring punch-class predictions). "
    "acc_true_punch: accuracy restricted to val samples whose ground truth is a punch."
)
print(
    f"Val TTA (best checkpoint): argmax mismatch rate={tta_disagree_final:.4f}  "
    f"mean rel ||Δlogit||={tta_rel_final:.4f}  "
    "(rel = ||orig−mirror|| / mean(||orig||,||mirror||))"
)
print("\nClassification report (val, best checkpoint):")
print(
    classification_report(
        vt,
        vp,
        target_names=[_display_class_name(c) for c in CLASSIFIER_CLASSES],
        zero_division=0,
    )
)
print("Confusion matrix:\n", confusion_matrix(vt, vp))

vers_tag = "_".join(trained_versions)
ckpt = _REPO / "checkpoints" / f"punch_transformer_7cls_{vers_tag}.pt"
ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state": best_state,
        "model_class": "PunchTransformer",
        "punch_classes": CLASSIFIER_CLASSES,
        "punch_classes_base": PUNCH_CLASSES,
        "no_punch_index": NO_PUNCH_IDX,
        "label_map": _LABEL_TO_IDX,
        "window": CLF_WINDOW,
        "in_channels": 6,
        "spatial_hidden": MODEL_SPATIAL_HIDDEN,
        "d_model": MODEL_D_MODEL,
        "nhead": MODEL_NHEAD,
        "num_layers": MODEL_NUM_LAYERS,
        "dim_feedforward": MODEL_DIM_FEEDFORWARD,
        "dropout": MODEL_DROPOUT,
        "skeleton": "H36M-17",
        "source": "MotionBERT_3d",
        "preprocessing": "body_frame + torso_scale",
        "negatives": (
            f"gap_review.json (label=no_punch) @ {_GAP_REVIEW_JSON.relative_to(_REPO)}; "
            f"{NO_PUNCH_SAMPLES_PER_GAP_SPAN} raw windows per span; USE_VERSIONS filter"
        ),
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "augmentation": (
            f"jitter±{JITTER_RANGE} + mirror_flip(50%, no label swap)"
        ),
        "lr_schedule": {
            "warmup_epochs": WARMUP_EPOCHS,
            "warmup": "LinearLR 0.01→1.0 × base LR",
            "cosine": f"CosineAnnealingLR T_max={cosine_epochs} eta_min={LR_MIN}",
        },
    },
    ckpt,
)
print(f"Checkpoint → {ckpt.relative_to(_REPO)}")
