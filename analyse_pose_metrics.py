#!/usr/bin/env python3
"""
Biomechanical analysis of MotionBERT 3D pose data for boxing punch classification.

Loads all annotated punch clips (V4-V10), extracts per-clip metrics in body frame,
and writes diagnostic plots to analysis_output/:

  01_violin_scalars.png        — per-class distributions of all metrics (6-class)
  02_violin_4cls.png           — same, lead/rear merged (4-class view)
  03_wrist_speed_ts.png        — mean ± std wrist speed time-series
  04_elbow_angle_ts.png        — mean ± std elbow angle time-series
  05_rotation_ts.png           — mean ± std shoulder/hip/xfactor time-series
  06_wrist_height_ts.png       — mean ± std wrist height (z) time-series
  07_scatter_pairs.png         — pairwise scatter of top ANOVA-ranked metrics
  08_correlation.png           — metric inter-correlation heatmap
  09_version_stability.png     — per-version consistency check
  10_laterality.png            — lead vs rear wrist-speed separation

  metrics.csv                  — full per-clip metric table for further analysis

Usage:
  python analyse_pose_metrics.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

try:
    from scipy.signal import savgol_filter as _savgol_fn
    from scipy.interpolate import interp1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from preprocess import _load_annotations, _normalize_label

# =============================================================================
# Paths & constants
# =============================================================================

_REPO           = Path.cwd().resolve()
_MOTIONBERT_DIR = _REPO / "Dataset" / "MotionBERT_3d"
_ANNOTATION_DIR = _REPO / "Dataset" / "Annotation_files"
_OUT_DIR        = _REPO / "analysis_output"

VERSIONS = frozenset(f"V{i}" for i in range(4, 11))

# H36M-17 joint indices
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

ALL_CLASSES   = ["Jab", "Cross", "Lead Hook", "Rear Hook", "Lead Uppercut", "Rear Uppercut"]
LEAD_CLASSES  = {"Jab", "Lead Hook", "Lead Uppercut"}
REAR_CLASSES  = {"Cross", "Rear Hook", "Rear Uppercut"}

CLASS_4_MAP = {
    "Jab": "Jab/Cross", "Cross": "Jab/Cross",
    "Lead Hook": "Hook", "Rear Hook": "Hook",
    "Lead Uppercut": "Uppercut", "Rear Uppercut": "Uppercut",
}
CLASS_4_ORDER = ["Jab/Cross", "Hook", "Uppercut"]

PALETTE_6 = dict(zip(ALL_CLASSES, sns.color_palette("tab10", 6)))
PALETTE_4 = {"Jab/Cross": "#4C72B0", "Hook": "#55A868", "Uppercut": "#8172B3"}
PALETTE_SIDE = {"Lead": "#4C72B0", "Rear": "#DD8452"}

N_NORM = 40  # frames for time-series normalisation

# =============================================================================
# Preprocessing helpers  (median-frame body axes — more stable than first frame)
# =============================================================================

def _to_body_frame(poses: np.ndarray) -> np.ndarray:
    q = poses - poses[:, [_J_PELVIS], :]
    ref = np.median(q, axis=0)
    x_raw = ref[_J_R_SHOULDER] - ref[_J_L_SHOULDER]
    if (n := np.linalg.norm(x_raw)) < 1e-6:
        return q
    x_hat = x_raw / n
    z_raw = ref[_J_THORAX] - ref[_J_PELVIS]
    if (n := np.linalg.norm(z_raw)) < 1e-6:
        return q
    z_raw /= n
    z_hat = z_raw - np.dot(z_raw, x_hat) * x_hat
    if (n := np.linalg.norm(z_hat)) < 1e-6:
        return q
    z_hat /= n
    return q @ np.stack([x_hat, np.cross(z_hat, x_hat), z_hat]).T


def _scale_norm(q: np.ndarray) -> np.ndarray:
    torso = float(np.median(np.linalg.norm(q[:, _J_THORAX] - q[:, _J_PELVIS], axis=-1)))
    return q / torso if torso > 1e-6 else q


def _smooth(q: np.ndarray, win: int = 5, poly: int = 2) -> np.ndarray:
    if not _HAS_SCIPY or q.shape[0] < win:
        return q
    T = q.shape[0]
    flat = q.reshape(T, -1)
    return _savgol_fn(flat, window_length=win,
                      polyorder=min(poly, win - 1), axis=0).reshape(q.shape)


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at b (degrees). Shapes (T,3) each → (T,)."""
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    return np.degrees(np.arccos(np.clip(
        np.einsum("ti,ti->t", ba, bc) / denom, -1.0, 1.0)))


def _speed(pos: np.ndarray) -> np.ndarray:
    """Speed of a single joint (T,3) → (T,) via central-diff velocity magnitude."""
    if pos.shape[0] < 2:
        return np.zeros(pos.shape[0], dtype=np.float64)
    return np.linalg.norm(np.gradient(pos, axis=0), axis=-1)


def preprocess_clip(raw: np.ndarray) -> np.ndarray:
    """(T,17,3) raw → (T,17,3) body-frame, torso-normalised, SG-smoothed."""
    return _smooth(_scale_norm(_to_body_frame(raw.astype(np.float64))))


def resample(arr: np.ndarray, n: int = N_NORM) -> np.ndarray:
    """Resample 1-D array to length n via linear interpolation."""
    if len(arr) == n:
        return arr.copy()
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n)
    if _HAS_SCIPY:
        return interp1d(x_old, arr, kind="linear")(x_new)
    return np.interp(x_new, x_old, arr)


# =============================================================================
# Data loading
# =============================================================================

def load_punch_clips() -> list[dict]:
    """
    Returns list of dicts:
      clip      (T,17,3)  body-frame preprocessed poses
      label     str       display-form 6-class label
      class4    str       merged 4-class label
      side      str       "Lead" | "Rear"
      version   str       "V5" etc.
      n_frames  int       raw clip length
    """
    label_set = set(ALL_CLASSES)
    records   = []

    for ver_dir in sorted(_MOTIONBERT_DIR.iterdir()):
        if not ver_dir.is_dir():
            continue
        ver = ver_dir.name.upper()
        if ver not in VERSIONS:
            continue
        npy_path = ver_dir / "X3D.npy"
        ann_path = _ANNOTATION_DIR / f"{ver}.xlsx"
        if not npy_path.exists() or not ann_path.exists():
            print(f"  [skip] {ver}: missing X3D.npy or annotation")
            continue

        frames      = np.load(npy_path)
        annotations = _load_annotations(ann_path)
        n_total     = frames.shape[0]
        kept        = 0

        for s, e, raw_label in annotations:
            label = _normalize_label(str(raw_label))
            if label not in label_set:
                continue
            s0, e0 = s - 1, min(e, n_total)
            if e0 <= s0:
                continue
            raw_clip = frames[s0:e0].copy()
            records.append({
                "clip":    preprocess_clip(raw_clip),
                "label":   label,
                "class4":  CLASS_4_MAP[label],
                "side":    "Lead" if label in LEAD_CLASSES else "Rear",
                "version": ver,
                "n_frames": raw_clip.shape[0],
            })
            kept += 1

        print(f"  {ver}: {kept} punch clips")

    print(f"\nTotal: {len(records)} clips across {len(VERSIONS)} versions")
    return records


# =============================================================================
# Per-clip feature extraction
# =============================================================================

def extract_scalars(q: np.ndarray) -> dict:
    """
    (T,17,3) body-frame clip → dict of scalar metrics.

    Body-frame axes:
      x = lateral  (+ = right / rear side for orthodox)
      y = forward  (+ = towards opponent)
      z = vertical (+ = up)
    """
    T = q.shape[0]

    # ── Wrist speeds ──────────────────────────────────────────────────────────
    sp_L = _speed(q[:, _J_L_WRIST])
    sp_R = _speed(q[:, _J_R_WRIST])
    pk_L, pk_R = float(sp_L.max()), float(sp_R.max())
    denom = pk_L + pk_R + 1e-8

    # Punching arm = faster wrist
    punch_right = pk_R > pk_L
    pw_sp   = sp_R    if punch_right else sp_L
    pw_j    = _J_R_WRIST    if punch_right else _J_L_WRIST
    pw_sh   = _J_R_SHOULDER if punch_right else _J_L_SHOULDER
    pw_el   = _J_R_ELBOW    if punch_right else _J_L_ELBOW
    oth_el  = _J_L_ELBOW    if punch_right else _J_R_ELBOW
    oth_sh  = _J_L_SHOULDER if punch_right else _J_R_SHOULDER

    t_pk = int(np.argmax(pw_sp))

    # ── Wrist displacement (start → end) ──────────────────────────────────────
    # Use peak-speed frame vs first frame for robustness
    pw_pos  = q[:, pw_j]
    w_disp  = pw_pos[t_pk] - pw_pos[0]  # (3,)

    disp_fwd  = float(w_disp[1])   # y: forward
    disp_lat  = float(w_disp[0])   # x: lateral
    disp_vert = float(w_disp[2])   # z: vertical

    # Direction angle: 0° = pure forward, +90° = straight up, ± = hook territory
    punch_angle_yz = float(np.degrees(np.arctan2(w_disp[2], w_disp[1])))   # vert vs fwd
    punch_angle_xy = float(np.degrees(np.arctan2(abs(w_disp[0]), w_disp[1])))  # lat vs fwd

    # ── Wrist extension = |wrist − shoulder| ─────────────────────────────────
    pw_ext = np.linalg.norm(q[:, pw_j] - q[:, pw_sh], axis=-1)

    # ── Elbow angles ──────────────────────────────────────────────────────────
    ang_punch = _angle3(q[:, pw_sh], q[:, pw_el],
                        q[:, _J_R_WRIST if punch_right else _J_L_WRIST])
    ang_other = _angle3(q[:, oth_sh], q[:, oth_el],
                        q[:, _J_L_WRIST if punch_right else _J_R_WRIST])

    # ── Shoulder & hip yaw ────────────────────────────────────────────────────
    sh_vec  = q[:, _J_R_SHOULDER] - q[:, _J_L_SHOULDER]
    hip_vec = q[:, _J_R_HIP]      - q[:, _J_L_HIP]
    sh_yaw  = np.degrees(np.arctan2(sh_vec[:,  1], sh_vec[:,  0]))
    hip_yaw = np.degrees(np.arctan2(hip_vec[:, 1], hip_vec[:, 0]))
    xfactor = sh_yaw - hip_yaw

    # ── Hip–shoulder rotation lag ─────────────────────────────────────────────
    if T < 2:
        sh_rot_vel = np.zeros_like(sh_yaw, dtype=np.float64)
        hip_rot_vel = np.zeros_like(hip_yaw, dtype=np.float64)
    else:
        sh_rot_vel = np.abs(np.gradient(sh_yaw))
        hip_rot_vel = np.abs(np.gradient(hip_yaw))
    t_sh_peak   = int(np.argmax(sh_rot_vel))
    t_hip_peak  = int(np.argmax(hip_rot_vel))
    rot_lag     = float((t_sh_peak - t_hip_peak) / max(T - 1, 1))  # + = hip leads

    # ── COM forward displacement ──────────────────────────────────────────────
    com = q.mean(axis=1)   # (T, 3)
    com_fwd_disp  = float(com[-1, 1] - com[0, 1])
    com_vert_disp = float(com[t_pk, 2] - com[0, 2])

    return {
        # Wrist speed
        "peak_L_speed":         pk_L,
        "peak_R_speed":         pk_R,
        "laterality_index":     float((pk_R - pk_L) / denom),   # +1=right, -1=left
        "peak_punch_speed":     float(max(pk_L, pk_R)),
        "speed_ratio":          float(max(pk_L, pk_R) / (min(pk_L, pk_R) + 1e-8)),

        # Wrist displacement direction
        "punch_fwd_disp":       disp_fwd,
        "punch_lat_disp":       disp_lat,
        "punch_vert_disp":      disp_vert,
        "punch_angle_yz_deg":   punch_angle_yz,   # + = upward, 0 = horizontal
        "punch_angle_xy_deg":   punch_angle_xy,   # + = lateral, 0 = straight

        # Wrist extension
        "max_extension":        float(pw_ext.max()),
        "extension_at_peak":    float(pw_ext[t_pk]),

        # Elbow
        "punch_elbow_min":      float(ang_punch.min()),
        "punch_elbow_at_peak":  float(ang_punch[t_pk]),
        "punch_elbow_range":    float(ang_punch.max() - ang_punch.min()),
        "guard_elbow_min":      float(ang_other.min()),
        "guard_elbow_at_peak":  float(ang_other[t_pk]),

        # Shoulder & hip rotation
        "shoulder_yaw_range":   float(sh_yaw.max() - sh_yaw.min()),
        "hip_yaw_range":        float(hip_yaw.max() - hip_yaw.min()),
        "xfactor_max":          float(xfactor.max()),
        "xfactor_at_peak":      float(xfactor[t_pk]),
        "hip_shoulder_lag":     rot_lag,   # + = hip leads shoulder (expected)

        # Temporal
        "time_to_peak":         float(t_pk / max(T - 1, 1)),

        # Global body
        "com_fwd_disp":         com_fwd_disp,
        "com_vert_at_peak":     com_vert_disp,

        # Meta
        "clip_frames":          float(T),
    }


def extract_timeseries(q: np.ndarray) -> dict[str, np.ndarray]:
    """(T,17,3) → dict of per-frame metric arrays (variable length T)."""
    sh_vec  = q[:, _J_R_SHOULDER] - q[:, _J_L_SHOULDER]
    hip_vec = q[:, _J_R_HIP]      - q[:, _J_L_HIP]
    sh_yaw  = np.degrees(np.arctan2(sh_vec[:,  1], sh_vec[:,  0]))
    hip_yaw = np.degrees(np.arctan2(hip_vec[:, 1], hip_vec[:, 0]))
    return {
        "sp_L":     _speed(q[:, _J_L_WRIST]),
        "sp_R":     _speed(q[:, _J_R_WRIST]),
        "elbow_L":  _angle3(q[:, _J_L_SHOULDER], q[:, _J_L_ELBOW], q[:, _J_L_WRIST]),
        "elbow_R":  _angle3(q[:, _J_R_SHOULDER], q[:, _J_R_ELBOW], q[:, _J_R_WRIST]),
        "sh_yaw":   sh_yaw,
        "hip_yaw":  hip_yaw,
        "xfactor":  sh_yaw - hip_yaw,
        "wrist_L_z": q[:, _J_L_WRIST, 2],
        "wrist_R_z": q[:, _J_R_WRIST, 2],
    }


# =============================================================================
# Plotting
# =============================================================================

def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def plot_violins_6cls(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("peak_punch_speed",    "Peak punch wrist speed"),
        ("speed_ratio",         "Speed ratio  (punch / guard)"),
        ("laterality_index",    "Laterality index  (+1=R, −1=L)"),
        ("punch_fwd_disp",      "Wrist forward disp  (y)"),
        ("punch_lat_disp",      "Wrist lateral disp  (x)"),
        ("punch_vert_disp",     "Wrist vertical disp  (z)"),
        ("punch_angle_yz_deg",  "Punch angle vert/fwd  (°)"),
        ("punch_angle_xy_deg",  "Punch angle lat/fwd  (°)"),
        ("max_extension",       "Max wrist extension"),
        ("punch_elbow_min",     "Punch elbow min  (°)"),
        ("punch_elbow_at_peak", "Punch elbow @ peak speed  (°)"),
        ("guard_elbow_min",     "Guard elbow min  (°)"),
        ("shoulder_yaw_range",  "Shoulder yaw range  (°)"),
        ("hip_yaw_range",       "Hip yaw range  (°)"),
        ("xfactor_max",         "Max X-factor  (°)"),
        ("xfactor_at_peak",     "X-factor @ peak speed  (°)"),
        ("hip_shoulder_lag",    "Hip–shoulder rotation lag"),
        ("time_to_peak",        "Normalised time to peak"),
        ("com_fwd_disp",        "COM forward displacement"),
        ("clip_frames",         "Clip duration  (frames)"),
    ]
    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, nrows * 4.2))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        sns.violinplot(
            data=df, x="label", y=col, order=ALL_CLASSES,
            palette=PALETTE_6, inner="box", cut=0, ax=ax,
        )
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=30)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle("Per-class distributions — all scalar metrics  (6-class)", fontsize=14)
    fig.tight_layout()
    _save(fig, "01_violin_scalars.png", out_dir)


def plot_violins_4cls(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("punch_fwd_disp",      "Forward disp  (y)"),
        ("punch_lat_disp",      "Lateral disp  (x)"),
        ("punch_vert_disp",     "Vertical disp  (z)"),
        ("punch_angle_yz_deg",  "Angle vert/fwd  (°)"),
        ("punch_angle_xy_deg",  "Angle lat/fwd  (°)"),
        ("punch_elbow_min",     "Punch elbow min  (°)"),
        ("punch_elbow_at_peak", "Punch elbow @ peak  (°)"),
        ("guard_elbow_min",     "Guard elbow min  (°)"),
        ("shoulder_yaw_range",  "Shoulder yaw range  (°)"),
        ("xfactor_max",         "Max X-factor  (°)"),
        ("peak_punch_speed",    "Peak wrist speed"),
        ("time_to_peak",        "Time to peak  (norm.)"),
    ]
    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4.2))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics):
        sns.violinplot(
            data=df, x="class4", y=col, order=CLASS_4_ORDER,
            palette=PALETTE_4, inner="box", cut=0, ax=ax,
        )
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle("Per-class distributions — 4-class merged  (lead/rear collapsed)", fontsize=14)
    fig.tight_layout()
    _save(fig, "02_violin_4cls.png", out_dir)


def _ts_mean_std(ts_by_class: dict, cls: str, key: str):
    arrs = [resample(r[key]) for r in ts_by_class[cls] if len(r[key]) > 1]
    if not arrs:
        return None, None
    mat = np.stack(arrs)
    return mat.mean(0), mat.std(0)


def plot_timeseries(ts_by_class: dict, out_dir: Path) -> None:
    t = np.linspace(0, 1, N_NORM)

    # ── Wrist speed ──────────────────────────────────────────────────────────
    fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for cls in ALL_CLASSES:
        c = PALETTE_6[cls]
        for ax, key, side in [(ax_L, "sp_L", "Left"), (ax_R, "sp_R", "Right")]:
            m, s = _ts_mean_std(ts_by_class, cls, key)
            if m is None:
                continue
            ax.plot(t, m, label=cls, color=c, lw=1.8)
            ax.fill_between(t, m - s, m + s, alpha=0.12, color=c)
    for ax, side in [(ax_L, "Left"), (ax_R, "Right")]:
        ax.set_title(f"{side} wrist speed  (torso-units / frame)")
        ax.set_xlabel("Normalised clip time")
        ax.legend(fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
    fig.suptitle("Mean ± 1 std wrist speed profiles", fontsize=13)
    fig.tight_layout()
    _save(fig, "03_wrist_speed_ts.png", out_dir)

    # ── Elbow angle ──────────────────────────────────────────────────────────
    fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for cls in ALL_CLASSES:
        c = PALETTE_6[cls]
        for ax, key, side in [(ax_L, "elbow_L", "Left"), (ax_R, "elbow_R", "Right")]:
            m, s = _ts_mean_std(ts_by_class, cls, key)
            if m is None:
                continue
            ax.plot(t, m, label=cls, color=c, lw=1.8)
            ax.fill_between(t, m - s, m + s, alpha=0.12, color=c)
    for ax, side in [(ax_L, "Left"), (ax_R, "Right")]:
        ax.set_title(f"{side} elbow angle  (°)")
        ax.set_xlabel("Normalised clip time")
        ax.legend(fontsize=8)
    fig.suptitle("Mean ± 1 std elbow angle profiles", fontsize=13)
    fig.tight_layout()
    _save(fig, "04_elbow_angle_ts.png", out_dir)

    # ── Shoulder / hip yaw & x-factor ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for cls in ALL_CLASSES:
        c = PALETTE_6[cls]
        for ax, key, title in [
            (axes[0], "sh_yaw",  "Shoulder yaw  (°)"),
            (axes[1], "hip_yaw", "Hip yaw  (°)"),
            (axes[2], "xfactor", "X-factor: shoulder − hip  (°)"),
        ]:
            m, s = _ts_mean_std(ts_by_class, cls, key)
            if m is None:
                continue
            ax.plot(t, m, label=cls, color=c, lw=1.8)
            ax.fill_between(t, m - s, m + s, alpha=0.12, color=c)
    for ax, title in zip(axes, ["Shoulder yaw  (°)", "Hip yaw  (°)", "X-factor  (°)"]):
        ax.set_title(title)
        ax.set_xlabel("Normalised clip time")
        ax.legend(fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
    fig.suptitle("Mean ± 1 std shoulder / hip rotation profiles", fontsize=13)
    fig.tight_layout()
    _save(fig, "05_rotation_ts.png", out_dir)

    # ── Wrist height (z) ─────────────────────────────────────────────────────
    fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for cls in ALL_CLASSES:
        c = PALETTE_6[cls]
        for ax, key, side in [(ax_L, "wrist_L_z", "Left"), (ax_R, "wrist_R_z", "Right")]:
            m, s = _ts_mean_std(ts_by_class, cls, key)
            if m is None:
                continue
            ax.plot(t, m, label=cls, color=c, lw=1.8)
            ax.fill_between(t, m - s, m + s, alpha=0.12, color=c)
    for ax, side in [(ax_L, "Left"), (ax_R, "Right")]:
        ax.set_title(f"{side} wrist height  (body-frame z)")
        ax.set_xlabel("Normalised clip time")
        ax.legend(fontsize=8)
        ax.axhline(0, color="k", lw=0.5, ls="--")
    fig.suptitle("Mean ± 1 std wrist height profiles  (+ = above pelvis level)", fontsize=13)
    fig.tight_layout()
    _save(fig, "06_wrist_height_ts.png", out_dir)


def plot_scatter_pairs(df: pd.DataFrame, top_metrics: list[str], out_dir: Path) -> None:
    sub = df[top_metrics + ["label"]].copy()
    # Shorten column names for axis readability
    rename = {c: c.replace("_", " ") for c in top_metrics}
    sub = sub.rename(columns=rename)
    cols = list(rename.values())
    g = sns.pairplot(
        sub, hue="label", hue_order=ALL_CLASSES,
        palette=PALETTE_6,
        vars=cols,
        plot_kws={"alpha": 0.35, "s": 18, "edgecolors": "none"},
        diag_kind="kde",
    )
    g.figure.suptitle(
        "Pairwise scatter of top ANOVA-ranked metrics  (coloured by 6-class label)",
        y=1.01, fontsize=12,
    )
    _save(g.figure, "07_scatter_pairs.png", out_dir)


def plot_correlation(df: pd.DataFrame, metric_cols: list[str], out_dir: Path) -> None:
    corr = df[metric_cols].corr()
    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        linewidths=0.3, annot_kws={"size": 7}, ax=ax,
    )
    ax.set_title("Metric inter-correlation  (lower triangle)", fontsize=13)
    fig.tight_layout()
    _save(fig, "08_correlation.png", out_dir)


def plot_version_stability(df: pd.DataFrame, key_metrics: list[str], out_dir: Path) -> None:
    versions = sorted(df["version"].unique())
    nrows = len(key_metrics)
    fig, axes = plt.subplots(nrows, 1, figsize=(max(10, len(versions) * 1.5), 4.5 * nrows))
    if nrows == 1:
        axes = [axes]
    for ax, metric in zip(axes, key_metrics):
        sns.boxplot(
            data=df, x="version", y=metric, hue="class4",
            order=versions, hue_order=CLASS_4_ORDER,
            palette=PALETTE_4, ax=ax, linewidth=0.8,
            flierprops={"marker": ".", "markersize": 4},
        )
        ax.set_title(f"{metric}  by version & class", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(title="class", fontsize=8, loc="upper right")
    fig.suptitle("Per-version metric stability  (cross-session consistency check)", fontsize=13)
    fig.tight_layout()
    _save(fig, "09_version_stability.png", out_dir)


def plot_laterality(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Peak L vs R speed scatter — clearest lead/rear fingerprint
    ax = axes[0, 0]
    for cls in ALL_CLASSES:
        sub = df[df["label"] == cls]
        ax.scatter(sub["peak_L_speed"], sub["peak_R_speed"],
                   label=cls, color=PALETTE_6[cls], alpha=0.45, s=25, edgecolors="none")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="L = R")
    ax.set_xlabel("Peak L wrist speed")
    ax.set_ylabel("Peak R wrist speed")
    ax.set_title("Peak wrist speed: L vs R\n(above diagonal = right-hand punch)")
    ax.legend(fontsize=7, markerscale=1.5)

    # 2. Laterality index by class
    ax = axes[0, 1]
    sns.boxplot(
        data=df, x="label", y="laterality_index", order=ALL_CLASSES,
        palette=PALETTE_6, ax=ax,
        flierprops={"marker": ".", "markersize": 4},
    )
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_title("Laterality index by class\n(+1 = right, −1 = left)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=25)

    # 3. Laterality by Lead / Rear grouping (expected strong separation)
    ax = axes[1, 0]
    sns.violinplot(
        data=df, x="side", y="laterality_index",
        order=["Lead", "Rear"], palette=PALETTE_SIDE,
        inner="box", cut=0, ax=ax,
    )
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_title("Laterality index: Lead vs Rear\n(all punch types combined)")
    ax.set_xlabel("")

    # 4. Punch elbow angle at peak speed by Lead / Rear — should overlap (elbow not lateral-specific)
    ax = axes[1, 1]
    sns.violinplot(
        data=df, x="class4", y="punch_elbow_at_peak",
        order=CLASS_4_ORDER,
        hue="side", hue_order=["Lead", "Rear"],
        palette=PALETTE_SIDE,
        inner="box", cut=0, ax=ax, split=True,
    )
    ax.set_title("Punch elbow angle @ peak speed  (°)\nLead vs Rear within each type")
    ax.set_xlabel("")
    ax.legend(title="Side", fontsize=8)

    fig.suptitle("Lead vs Rear hand separation analysis", fontsize=13)
    fig.tight_layout()
    _save(fig, "10_laterality.png", out_dir)


# =============================================================================
# ANOVA separability ranking
# =============================================================================

def compute_anova(df: pd.DataFrame, metric_cols: list[str]) -> list[str]:
    """One-way ANOVA across 6 classes for every metric. Returns top-6 by F-stat."""
    print("\n── ANOVA separability  (6-class, one-way) ──────────────────────────")
    print(f"  {'Metric':<28}  {'F':>8}  {'p':>10}  {'η²':>6}  sig")
    print("  " + "─" * 60)

    results = []
    grand_mean_cache: dict[str, float] = {}

    for col in metric_cols:
        groups = [df.loc[df["label"] == c, col].dropna().values for c in ALL_CLASSES]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        gm = df[col].mean()
        ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
        ss_t = sum(((g - gm) ** 2).sum() for g in groups)
        eta2 = ss_b / (ss_t + 1e-10)
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        results.append((col, F, p, eta2))
        print(f"  {col:<28}  {F:>8.1f}  {p:>10.2e}  {eta2:>6.3f}  {sig}")

    results.sort(key=lambda x: x[1], reverse=True)
    top = [r[0] for r in results[:6]]
    print(f"\n  Top 6 by F-stat: {top}")
    return top


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if not _HAS_SCIPY:
        print("WARNING: scipy not found — smoothing and interpolation disabled")

    _OUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 100

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading clips from {_MOTIONBERT_DIR.relative_to(_REPO)} …")
    records = load_punch_clips()
    if not records:
        raise SystemExit("No clips loaded — check Dataset/ paths.")

    # ── Scalar DataFrame ─────────────────────────────────────────────────────
    print("\nExtracting scalar metrics …")
    rows = []
    for r in records:
        row = extract_scalars(r["clip"])
        row.update({"label": r["label"], "class4": r["class4"],
                    "side": r["side"], "version": r["version"]})
        rows.append(row)
    df = pd.DataFrame(rows)

    meta_cols   = ["label", "class4", "side", "version"]
    metric_cols = [c for c in df.columns if c not in meta_cols]
    print(f"DataFrame: {len(df)} clips  ×  {len(metric_cols)} metrics")
    print("\nClass counts:\n" +
          df["label"].value_counts().reindex(ALL_CLASSES).to_string())

    # ── Time-series dict ─────────────────────────────────────────────────────
    print("\nExtracting time-series …")
    ts_by_class: dict[str, list] = defaultdict(list)
    for r in records:
        ts_by_class[r["label"]].append(extract_timeseries(r["clip"]))

    # ── ANOVA ranking ─────────────────────────────────────────────────────────
    top_metrics = compute_anova(df, metric_cols)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_violins_6cls(df, _OUT_DIR)
    plot_violins_4cls(df, _OUT_DIR)
    plot_timeseries(ts_by_class, _OUT_DIR)
    plot_scatter_pairs(df, top_metrics, _OUT_DIR)
    plot_correlation(df, metric_cols, _OUT_DIR)
    plot_version_stability(
        df,
        key_metrics=["punch_elbow_min", "laterality_index",
                     "punch_angle_yz_deg", "xfactor_max"],
        out_dir=_OUT_DIR,
    )
    plot_laterality(df, _OUT_DIR)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = _OUT_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics table → {csv_path.relative_to(_REPO)}")
    print(f"All plots     → {_OUT_DIR.relative_to(_REPO)}/")
