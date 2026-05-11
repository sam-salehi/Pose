"""
Soft-vote ensemble of three trained checkpoints:

  A  punch_transformer_7cls_bio_*.pt      in_channels=15  CLF_WINDOW=16
  B  punch_transformer_7cls_V*.pt         in_channels=6   CLF_WINDOW=16
  C  punch_transformer_7cls_neville_*.pt  in_channels=6   CLF_WINDOW=24

No-punch source: Dataset/gap_labels/gap_review.json (label == "no_punch"),
matching Model B's training data.  Versions with no gap_review.json entry
contribute punch clips only.

Val split: SEED=42, VAL_FRAC=0.1 — matches all three training scripts.

Outputs
-------
  • Per-model metrics (with each model's own TTA)
  • Equal-weight soft-vote ensemble
  • Bal-acc-weighted soft-vote ensemble
  • Indicative stacking (LR meta-learner fitted on the same val set — biased
    upper bound; quoted separately so it is not mistaken for a real estimate)

Usage
-----
  python ensemble_eval.py                    # auto-find latest checkpoints
  python ensemble_eval.py --bio path/to/a.pt --base path/to/b.pt --neville path/to/c.pt
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

try:
    from scipy.signal import savgol_filter as _savgol_fn
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from GCN import H36M_BONE_PAIRS, PUNCH_CLASSES, NUM_H36M_JOINTS
from punch_transformer import PunchTransformer
from preprocess import _load_annotations, _normalize_label

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_REPO             = Path.cwd().resolve()
_MOTIONBERT_DIR   = _REPO / "Dataset" / "MotionBERT_3d"
_ANNOTATION_DIR   = _REPO / "Dataset" / "Annotation_files"
_GAP_REVIEW_JSON  = _REPO / "Dataset" / "gap_labels" / "gap_review.json"
_CKPT_DIR         = _REPO / "checkpoints"

USE_VERSIONS: frozenset[str] = frozenset(f"V{i}" for i in range(1, 11))

SEED            = 42
VAL_FRAC        = 0.1
NO_PUNCH_MIN_GAP_FRAMES = 16
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFIER_CLASSES = [*PUNCH_CLASSES, "no_punch"]
NO_PUNCH_IDX       = len(PUNCH_CLASSES)
N_CLASSES          = len(CLASSIFIER_CLASSES)

# Bio model mirror-label permutation (cross↔jab, lead_hook↔rear_hook, …)
_MIRROR_LABEL_MAP = [
    PUNCH_CLASSES.index("jab"),
    PUNCH_CLASSES.index("cross"),
    PUNCH_CLASSES.index("rear_hook"),
    PUNCH_CLASSES.index("rear_uppercut"),
    PUNCH_CLASSES.index("lead_hook"),
    PUNCH_CLASSES.index("lead_uppercut"),
    NO_PUNCH_IDX,
]
_MIRROR_PERM = torch.tensor(_MIRROR_LABEL_MAP, dtype=torch.long)

# H36M-17 L/R joint swap
_FLIP_JOINT_ORDER = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

# H36M-17 joint indices used in biomechanical features
_J_PELVIS, _J_THORAX           = 0, 8
_J_R_HIP,  _J_L_HIP            = 1, 4
_J_L_SHOULDER, _J_L_ELBOW, _J_L_WRIST = 11, 12, 13
_J_R_SHOULDER, _J_R_ELBOW, _J_R_WRIST = 14, 15, 16

_LABEL_TO_IDX: dict[str, int] = {
    "Cross":         PUNCH_CLASSES.index("cross"),
    "Jab":           PUNCH_CLASSES.index("jab"),
    "Lead Hook":     PUNCH_CLASSES.index("lead_hook"),
    "Lead Uppercut": PUNCH_CLASSES.index("lead_uppercut"),
    "Rear Hook":     PUNCH_CLASSES.index("rear_hook"),
    "Rear Uppercut": PUNCH_CLASSES.index("rear_uppercut"),
}


# ---------------------------------------------------------------------------
# gap_review.json helpers (mirrors train_3d_classifier_no_punch.py)
# ---------------------------------------------------------------------------

def _load_gap_review_spans() -> list[tuple[str, int, int]]:
    """Return (workbook, start, end) for every no_punch entry in gap_review.json."""
    if not _GAP_REVIEW_JSON.is_file():
        raise SystemExit(
            f"Missing {_GAP_REVIEW_JSON} — run label_punch_gaps.py first, "
            f"or check the path."
        )
    data = json.loads(_GAP_REVIEW_JSON.read_text(encoding="utf-8"))
    spans: list[tuple[str, int, int]] = []
    for e in data.get("entries", []):
        if e.get("label") != "no_punch":
            continue
        ver = str(e["workbook"]).strip().upper()
        if ver not in USE_VERSIONS:
            continue
        s0, e0 = int(e["start"]), int(e["end"])
        if e0 > s0:
            spans.append((ver, s0, e0))
    return spans


def _two_spans(g0: int, g1: int, window: int) -> list[tuple[int, int]]:
    """Flush-left + flush-right raw windows inside [g0, g1)."""
    L = g1 - g0
    if L <= 0:
        return []
    if L < window:
        return [(g0, g1), (g0, g1)]
    last = g1 - window
    if last <= g0:
        return [(g0, g0 + window), (g0, g0 + window)]
    return [(g0, g0 + window), (last, g1)]


# ---------------------------------------------------------------------------
# Preprocessing — base (6-channel)
# ---------------------------------------------------------------------------

def _to_body_frame(poses: np.ndarray) -> np.ndarray:
    q   = poses - poses[:, [_J_PELVIS], :]
    ref = q[0]
    x_r = ref[_J_R_SHOULDER] - ref[_J_L_SHOULDER]
    xn  = np.linalg.norm(x_r)
    if xn < 1e-6: return q
    x_hat = x_r / xn
    z_r   = ref[_J_THORAX] - ref[_J_PELVIS]
    zn    = np.linalg.norm(z_r)
    if zn < 1e-6: return q
    z_r /= zn
    z_hat = z_r - np.dot(z_r, x_hat) * x_hat
    zn2   = np.linalg.norm(z_hat)
    if zn2 < 1e-6: return q
    z_hat /= zn2
    R = np.stack([x_hat, np.cross(z_hat, x_hat), z_hat], axis=0)
    return q @ R.T


def _scale_normalize(q: np.ndarray) -> np.ndarray:
    t = float(np.median(np.linalg.norm(q[:, _J_THORAX] - q[:, _J_PELVIS], axis=-1)))
    return q / t if t > 1e-6 else q


def _preprocess_base(clip: np.ndarray) -> np.ndarray:
    """(T,17,3) → (T,17,6): body-frame pos + forward-diff vel."""
    q   = _scale_normalize(_to_body_frame(clip.astype(np.float64))).astype(np.float32)
    vel = np.zeros_like(q)
    vel[1:] = q[1:] - q[:-1]
    return np.nan_to_num(np.concatenate([q, vel], axis=-1), nan=0., posinf=0., neginf=0.)


# ---------------------------------------------------------------------------
# Preprocessing — bio (15-channel)
# ---------------------------------------------------------------------------

def _smooth(q: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    T = q.shape[0]
    if not _HAS_SCIPY or T < window:
        return q
    flat = q.reshape(T, -1)
    return _savgol_fn(flat, window_length=window,
                      polyorder=min(polyorder, window-1), axis=0).reshape(q.shape)


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba, bc = a - b, c - b
    cos = np.clip(
        np.einsum("ti,ti->t", ba, bc) /
        (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8),
        -1., 1.
    )
    return np.arccos(cos)


def _preprocess_bio(clip: np.ndarray) -> np.ndarray:
    """(T,17,3) → (T,17,15): pos+vel+acc (SG-smoothed) + 6 broadcast scalars."""
    q    = _scale_normalize(_to_body_frame(clip.astype(np.float64)))
    q_sm = _smooth(q)
    T    = q_sm.shape[0]
    if T > 1:
        vel = np.gradient(q_sm, axis=0)
        acc = np.gradient(vel,  axis=0)
    else:
        vel = acc = np.zeros_like(q_sm)

    l_elbow = _angle3(q_sm[:, _J_L_SHOULDER], q_sm[:, _J_L_ELBOW], q_sm[:, _J_L_WRIST])
    r_elbow = _angle3(q_sm[:, _J_R_SHOULDER], q_sm[:, _J_R_ELBOW], q_sm[:, _J_R_WRIST])
    hip_vec = q_sm[:, _J_R_HIP]      - q_sm[:, _J_L_HIP]
    sh_vec  = q_sm[:, _J_R_SHOULDER] - q_sm[:, _J_L_SHOULDER]
    hip_yaw = np.arctan2(hip_vec[:, 1], hip_vec[:, 0])
    sh_yaw  = np.arctan2(sh_vec[:,  1], sh_vec[:,  0])
    scalars = np.stack([l_elbow, r_elbow, hip_yaw, sh_yaw,
                        sh_yaw - hip_yaw, q_sm[:,:,2].mean(1)], axis=-1)
    sc_bc   = np.broadcast_to(scalars[:, np.newaxis, :], (T, 17, 6)).copy()
    out = np.concatenate([q_sm, vel, acc, sc_bc], axis=-1).astype(np.float32)
    return np.nan_to_num(out, nan=0., posinf=0., neginf=0.)


# ---------------------------------------------------------------------------
# Window crop
# ---------------------------------------------------------------------------

def _window(seq: np.ndarray, window: int) -> np.ndarray:
    T = seq.shape[0]
    if T < window:
        pp, pq = (window - T) // 2, window - T - (window - T) // 2
        seq = np.concatenate([
            np.tile(seq[[0]], (pp, 1, 1)), seq, np.tile(seq[[-1]], (pq, 1, 1))
        ], axis=0)
        T = window
    s = T // 2 - window // 2
    chunk = seq[s: s + window]
    if chunk.shape[0] < window:
        chunk = np.concatenate(
            [chunk, np.tile(chunk[[-1]], (window - chunk.shape[0], 1, 1))], axis=0
        )
    return chunk


# ---------------------------------------------------------------------------
# Data loading — punch clips from xlsx, no_punch from gap_review.json
# ---------------------------------------------------------------------------

def load_raw_clips() -> tuple[list[np.ndarray], np.ndarray]:
    """
    Raw (T,17,3) clips.  Punch clips from annotation xlsx files.
    No-punch clips from gap_review.json (label == "no_punch") only —
    versions not present in that file contribute punch clips only.
    Window size for two-span sampling is 24 (largest model window) so every
    clip is long enough for all three models; _window() handles any cropping.
    """
    _NP_WINDOW = 24   # largest CLF_WINDOW across the three models

    clips: list[np.ndarray] = []
    y_list: list[int]       = []
    frame_cache: dict[str, np.ndarray] = {}

    def _frames(ver: str) -> np.ndarray | None:
        if ver not in frame_cache:
            npy = _MOTIONBERT_DIR / ver / "X3D.npy"
            if not npy.exists():
                return None
            frame_cache[ver] = np.load(npy)
        return frame_cache[ver]

    # ── Punch clips ───────────────────────────────────────────────────────
    n_punch = 0
    for ver_dir in sorted(_MOTIONBERT_DIR.iterdir()):
        if not ver_dir.is_dir():
            continue
        ver = ver_dir.name.upper()
        if ver not in USE_VERSIONS:
            continue
        ann = _ANNOTATION_DIR / f"{ver}.xlsx"
        if not ann.exists():
            continue
        fr = _frames(ver)
        if fr is None:
            continue
        n = fr.shape[0]
        for s, e, raw_lbl in _load_annotations(ann):
            lbl = _normalize_label(str(raw_lbl))
            if lbl not in _LABEL_TO_IDX:
                continue
            s0, e0 = s - 1, min(e, n)
            if e0 <= s0:
                continue
            clips.append(fr[s0:e0].copy())
            y_list.append(_LABEL_TO_IDX[lbl])
            n_punch += 1

    # ── No-punch clips from gap_review.json ───────────────────────────────
    gap_spans = _load_gap_review_spans()
    n_np, n_skip = 0, 0
    for ver, s0, e0 in gap_spans:
        fr = _frames(ver)
        if fr is None:
            n_skip += 1
            continue
        n  = fr.shape[0]
        e0 = min(e0, n)
        if e0 <= s0:
            n_skip += 1
            continue
        for a, b in _two_spans(s0, e0, _NP_WINDOW):
            clips.append(fr[a:b].copy())
            y_list.append(NO_PUNCH_IDX)
            n_np += 1

    print(f"Punch clips: {n_punch}  |  no_punch clips: {n_np} "
          f"({n_skip} gap_review spans skipped — missing X3D or out of range)")
    if not clips:
        raise SystemExit("No clips loaded — check paths and USE_VERSIONS.")
    if n_np == 0:
        raise SystemExit("No no_punch clips loaded — check gap_review.json.")
    return clips, np.array(y_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _latest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(_CKPT_DIR / pattern)))
    return Path(hits[-1]) if hits else None


def _find_checkpoints(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    bio     = Path(args.bio)     if args.bio     else _latest("punch_transformer_7cls_bio_*.pt")
    base    = Path(args.base)    if args.base    else _latest("punch_transformer_7cls_V*.pt")
    neville = Path(args.neville) if args.neville else _latest("punch_transformer_7cls_neville_*.pt")
    missing = [n for n, p in [("bio", bio), ("base", base), ("neville", neville)] if p is None]
    if missing:
        raise SystemExit(f"Checkpoint(s) not found: {missing}. Train them first, or pass --bio/--base/--neville.")
    return bio, base, neville


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path) -> tuple[PunchTransformer, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = {
        "in_channels":      ckpt.get("in_channels", 6),
        "spatial_hidden":   ckpt.get("spatial_hidden", 64),
        "d_model":          ckpt.get("d_model", 128),
        "nhead":            ckpt.get("nhead", 4),
        "num_layers":       ckpt.get("num_layers", 4),
        "dim_feedforward":  ckpt.get("dim_feedforward", 256),
        "dropout":          ckpt.get("dropout", 0.1),
        "window":           ckpt.get("window", 16),
    }
    m = PunchTransformer(
        num_classes     = N_CLASSES,
        in_channels     = cfg["in_channels"],
        edges           = H36M_BONE_PAIRS,
        spatial_hidden  = cfg["spatial_hidden"],
        d_model         = cfg["d_model"],
        nhead           = cfg["nhead"],
        num_layers      = cfg["num_layers"],
        dim_feedforward = cfg["dim_feedforward"],
        dropout         = cfg["dropout"],
    ).to(DEVICE)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, cfg


# ---------------------------------------------------------------------------
# TTA mirror helpers
# ---------------------------------------------------------------------------

def _mirror_base(xb: torch.Tensor) -> torch.Tensor:
    """Mirror for base / neville models (6-ch: pos+vel)."""
    idx  = torch.tensor(_FLIP_JOINT_ORDER, device=xb.device, dtype=torch.long)
    flip = xb[:, :, idx, :].clone()
    flip[:, :, :, 0] *= -1   # pos_x
    flip[:, :, :, 3] *= -1   # vel_x
    return flip


def _mirror_bio(xb: torch.Tensor) -> torch.Tensor:
    """Mirror for bio model (15-ch: pos+vel+acc+scalars)."""
    idx  = torch.tensor(_FLIP_JOINT_ORDER, device=xb.device, dtype=torch.long)
    flip = xb[:, :, idx, :].clone()
    flip[:, :, :, 0] *= -1   # pos_x
    flip[:, :, :, 3] *= -1   # vel_x
    flip[:, :, :, 6] *= -1   # acc_x
    # scalar channels 9,10 = l/r elbow angles → swap
    tmp = flip[:, :, :, 9].clone()
    flip[:, :, :, 9]  = flip[:, :, :, 10]
    flip[:, :, :, 10] = tmp
    flip[:, :, :, 11] *= -1  # hip_yaw
    flip[:, :, :, 12] *= -1  # shoulder_yaw
    flip[:, :, :, 13] *= -1  # xfactor
    return flip


# ---------------------------------------------------------------------------
# Inference — returns (N, 7) softmax probabilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def _infer(
    model:    PunchTransformer,
    clips:    list[np.ndarray],
    preproc,              # callable: raw (T,17,3) → (T,17,C)
    window:   int,
    mirror_fn,            # callable: (N,T,V,C) tensor → mirrored tensor
    perm:     torch.Tensor | None,  # if not None, permute mirrored logits before avg
    batch:    int = 128,
) -> np.ndarray:
    all_probs: list[np.ndarray] = []
    for start in range(0, len(clips), batch):
        batch_raw = clips[start: start + batch]
        wins = np.stack([_window(preproc(c), window) for c in batch_raw])  # (B,T,V,C)
        xb   = torch.from_numpy(wins).to(DEVICE)
        lo   = model(xb)
        lf   = model(mirror_fn(xb))
        if perm is not None:
            lf = lf[:, perm.to(DEVICE)]
        probs = (F.softmax(lo, dim=-1) + F.softmax(lf, dim=-1)) * 0.5
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)   # (N, 7)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _report(name: str, y_true: np.ndarray, probs: np.ndarray) -> float:
    y_pred = probs.argmax(axis=1)
    bal    = balanced_accuracy_score(y_true, y_pred)
    f7     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f6     = f1_score(y_true, y_pred, average="macro",
                      labels=list(range(6)), zero_division=0)
    pm     = y_true != NO_PUNCH_IDX
    pun_acc = float((y_pred[pm] == y_true[pm]).mean()) if pm.any() else float("nan")
    acc    = float((y_pred == y_true).mean())
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  acc={acc:.4f}  bal_acc={bal:.4f}  macroF1_7={f7:.4f}"
          f"  macroF1_6punch={f6:.4f}  punch_acc={pun_acc:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=[c.replace("_"," ").title() for c in CLASSIFIER_CLASSES],
        zero_division=0,
    ))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    return bal


def _display_class_name(c: str) -> str:
    return "No Punch" if c == "no_punch" else c.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Soft-vote ensemble of three punch classifiers.")
    ap.add_argument("--bio",     default=None, help="Path to bio checkpoint (.pt)")
    ap.add_argument("--base",    default=None, help="Path to base no-punch checkpoint (.pt)")
    ap.add_argument("--neville", default=None, help="Path to neville checkpoint (.pt)")
    ap.add_argument("--batch",   type=int, default=128)
    args = ap.parse_args()

    bio_path, base_path, nev_path = _find_checkpoints(args)
    print(f"Bio     checkpoint: {bio_path.name}")
    print(f"Base    checkpoint: {base_path.name}")
    print(f"Neville checkpoint: {nev_path.name}")
    print(f"No-punch source: {_GAP_REVIEW_JSON.relative_to(_REPO)}\n")

    print("Loading raw clips …")
    all_clips_raw, all_y = load_raw_clips()
    print(f"Loaded {len(all_y)} clips  "
          f"({Counter(all_y.tolist())[NO_PUNCH_IDX]} no_punch)")

    _, idx_val = train_test_split(
        np.arange(len(all_y)), test_size=VAL_FRAC,
        random_state=SEED,
        stratify=all_y if np.all(np.unique(all_y, return_counts=True)[1] >= 2) else None,
    )
    val_clips = [all_clips_raw[i] for i in idx_val]
    val_y     = all_y[idx_val]
    print(f"Val set: {len(val_y)} clips\n")

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading checkpoints …")
    m_bio,  cfg_bio  = _load_model(bio_path)
    m_base, cfg_base = _load_model(base_path)
    m_nev,  cfg_nev  = _load_model(nev_path)
    print(f"  Bio     params={sum(p.numel() for p in m_bio.parameters()):,}  "
          f"in_ch={cfg_bio['in_channels']}  window={cfg_bio['window']}")
    print(f"  Base    params={sum(p.numel() for p in m_base.parameters()):,}  "
          f"in_ch={cfg_base['in_channels']}  window={cfg_base['window']}")
    print(f"  Neville params={sum(p.numel() for p in m_nev.parameters()):,}  "
          f"in_ch={cfg_nev['in_channels']}  window={cfg_nev['window']}")

    # ── Per-model inference ───────────────────────────────────────────────────
    print("\nRunning inference …")
    p_bio  = _infer(m_bio,  val_clips, _preprocess_bio,  cfg_bio["window"],
                    _mirror_bio,  _MIRROR_PERM, args.batch)
    p_base = _infer(m_base, val_clips, _preprocess_base, cfg_base["window"],
                    _mirror_base, None, args.batch)
    p_nev  = _infer(m_nev,  val_clips, _preprocess_base, cfg_nev["window"],
                    _mirror_base, None, args.batch)

    # ── Individual model metrics ──────────────────────────────────────────────
    bal_bio  = _report("Model A — Bio (in_ch=15)",             val_y, p_bio)
    bal_base = _report("Model B — Base (in_ch=6)",             val_y, p_base)
    bal_nev  = _report("Model C — Neville (in_ch=6, win=24)",  val_y, p_nev)

    # ── Equal-weight soft vote ────────────────────────────────────────────────
    p_equal = (p_bio + p_base + p_nev) / 3.0
    bal_eq  = _report("Ensemble — equal-weight soft vote", val_y, p_equal)

    # ── Balanced-accuracy-weighted soft vote ─────────────────────────────────
    w = np.array([bal_bio, bal_base, bal_nev])
    w = w / w.sum()
    p_weighted = p_bio * w[0] + p_base * w[1] + p_nev * w[2]
    bal_wt = _report(
        f"Ensemble — bal-acc-weighted soft vote "
        f"(w={w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f})",
        val_y, p_weighted,
    )

    # ── Indicative stacking (biased — val used for both fit and eval) ─────────
    print("\n" + "="*60)
    print("  Indicative stacking  [BIASED — val fitted = val evaluated]")
    print("  Real stacking requires OOF predictions from training-time CV.")
    print("="*60)
    meta_X = np.concatenate([p_bio, p_base, p_nev], axis=1)   # (N, 21)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    lr.fit(meta_X, val_y)
    p_stack = lr.predict_proba(meta_X)
    _report("Indicative stacking (LR, in-sample)", val_y, p_stack)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY — balanced accuracy")
    print("="*60)
    for label, bal in [
        ("Bio",              bal_bio),
        ("Base",             bal_base),
        ("Neville",          bal_nev),
        ("Soft-vote equal",  bal_eq),
        ("Soft-vote weighted", bal_wt),
    ]:
        print(f"  {label:<24} {bal:.4f}")
    print(f"\n  Best soft-vote: {'equal' if bal_eq >= bal_wt else 'weighted'}"
          f"  ({max(bal_eq, bal_wt):.4f})")
    delta = max(bal_eq, bal_wt) - max(bal_bio, bal_base, bal_nev)
    print(f"  Ensemble gain over best single model: {delta:+.4f}")


if __name__ == "__main__":
    main()
