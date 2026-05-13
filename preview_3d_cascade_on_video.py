#!/usr/bin/env python3
"""
Writes a composite MP4: RGB (cascade overlay) + side panel with MotionBERT X3D
skeleton (H36M-17, XY projection, pelvis-centered per frame).

Stage 1 — binary model (punch_transformer_binary_bio_*.pt, 2 classes):
  decides punch vs no_punch for every window.

Stage 2 — 7-class bio model (punch_transformer_7cls_bio_*.pt, 7 classes):
  runs only on windows Stage 1 called "punch" and returns the specific type.

Both models use 15-channel bio preprocessing (identical to train_3d_classifier_bio.py).
Windows where Stage 1 says no_punch are never passed to Stage 2.

Example::

    python preview_3d_cascade_on_video.py --ver V7
    python preview_3d_cascade_on_video.py --ver V7 --start-frac 0
    python preview_3d_cascade_on_video.py --ver V7 \\
        --binary  checkpoints/punch_transformer_binary_bio_V4_V5_V6_V7_V8_V9_V10.pt \\
        --seven   checkpoints/punch_transformer_7cls_bio_V4_V5_V6_V7_V8_V9_V10.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from skeleton_constants import H36M_BONE_PAIRS
from preprocess import _find_video
from punch_transformer import PunchTransformer

try:
    from scipy.signal import savgol_filter as _savgol_fn
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_REPO           = Path(__file__).resolve().parent
_MOTIONBERT_DIR = _REPO / "Dataset" / "MotionBERT_3d"
_FIGURES        = _REPO / "figures"
_CHECKPOINTS    = _REPO / "checkpoints"

# ── Joint indices (H36M-17) ────────────────────────────────────────────────
_J_PELVIS     = 0
_J_THORAX     = 8
_J_R_HIP      = 1
_J_L_HIP      = 4
_J_R_SHOULDER = 14
_J_L_SHOULDER = 11
_J_L_ELBOW    = 12
_J_L_WRIST    = 13
_J_R_ELBOW    = 15
_J_R_WRIST    = 16
_N_SCALARS    = 6


# =============================================================================
# Preprocessing (bio, 15 ch) — identical to train_3d_classifier_bio._preprocess_clip
# =============================================================================

def _to_body_frame(poses: np.ndarray) -> np.ndarray:
    q = poses - poses[:, [_J_PELVIS], :]
    ref = q[0]
    x_raw = ref[_J_R_SHOULDER] - ref[_J_L_SHOULDER]
    if np.linalg.norm(x_raw) < 1e-6:
        return q
    x_hat = x_raw / np.linalg.norm(x_raw)
    z_raw = ref[_J_THORAX] - ref[_J_PELVIS]
    if np.linalg.norm(z_raw) < 1e-6:
        return q
    z_raw = z_raw / np.linalg.norm(z_raw)
    z_hat = z_raw - np.dot(z_raw, x_hat) * x_hat
    if np.linalg.norm(z_hat) < 1e-6:
        return q
    z_hat = z_hat / np.linalg.norm(z_hat)
    y_hat = np.cross(z_hat, x_hat)
    return q @ np.stack([x_hat, y_hat, z_hat], axis=0).T


def _scale_normalize(q: np.ndarray) -> np.ndarray:
    torso = float(np.median(np.linalg.norm(q[:, _J_THORAX] - q[:, _J_PELVIS], axis=-1)))
    return q / torso if torso > 1e-6 else q


def _smooth(q: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    T = q.shape[0]
    if not _HAS_SCIPY or T < window:
        return q
    flat = q.reshape(T, -1)
    return _savgol_fn(flat, window_length=window, polyorder=min(polyorder, window - 1), axis=0).reshape(q.shape)


def _angle3(a, b, c) -> np.ndarray:
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    return np.arccos(np.clip(np.einsum("ti,ti->t", ba, bc) / denom, -1.0, 1.0))


def _preprocess_bio(clip_xyz: np.ndarray) -> np.ndarray:
    """(T, 17, 3) → (T, 17, 15) float32."""
    q    = _to_body_frame(clip_xyz.astype(np.float64))
    q    = _scale_normalize(q)
    q_sm = _smooth(q)
    T    = q_sm.shape[0]
    if T > 1:
        vel = np.gradient(q_sm, axis=0)
        acc = np.gradient(vel,  axis=0)
    else:
        vel = acc = np.zeros_like(q_sm)

    l_elbow      = _angle3(q_sm[:, _J_L_SHOULDER], q_sm[:, _J_L_ELBOW], q_sm[:, _J_L_WRIST])
    r_elbow      = _angle3(q_sm[:, _J_R_SHOULDER], q_sm[:, _J_R_ELBOW], q_sm[:, _J_R_WRIST])
    hip_vec      = q_sm[:, _J_R_HIP]      - q_sm[:, _J_L_HIP]
    sh_vec       = q_sm[:, _J_R_SHOULDER] - q_sm[:, _J_L_SHOULDER]
    hip_yaw      = np.arctan2(hip_vec[:, 1], hip_vec[:, 0])
    shoulder_yaw = np.arctan2(sh_vec[:, 1],  sh_vec[:, 0])
    xfactor      = shoulder_yaw - hip_yaw
    com_z        = q_sm[:, :, 2].mean(axis=1)

    scalars    = np.stack([l_elbow, r_elbow, hip_yaw, shoulder_yaw, xfactor, com_z], axis=-1)
    scalars_bc = np.broadcast_to(scalars[:, np.newaxis, :], (T, 17, _N_SCALARS)).copy()
    out = np.concatenate([q_sm, vel, acc, scalars_bc], axis=-1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _window_centered_at(seq: np.ndarray, window: int, center_idx: int) -> np.ndarray:
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
        T          = window
        center_idx = center_idx + pad_pre
    peak  = max(half, min(T - (window - half), int(center_idx)))
    start = peak - half
    chunk = seq[start: start + window]
    if chunk.shape[0] < window:
        pad   = window - chunk.shape[0]
        chunk = np.concatenate([chunk, np.tile(chunk[[-1]], (pad, 1, 1))], axis=0)
    return chunk


# =============================================================================
# Model loading
# =============================================================================

def _load_model(ckpt_path: Path, device: torch.device) -> tuple[PunchTransformer, list[str], int, int]:
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state")
    if state is None:
        raise SystemExit(f"Checkpoint missing model_state: {ckpt_path}")
    classes = list(ckpt.get("punch_classes", []))
    window  = int(ckpt.get("window", 16))
    in_ch   = int(ckpt.get("in_channels", 15))
    model   = PunchTransformer(
        num_classes=len(classes),
        in_channels=in_ch,
        edges=H36M_BONE_PAIRS,
        spatial_hidden=int(ckpt.get("spatial_hidden", 96)),
        d_model=int(ckpt.get("d_model", 128)),
        nhead=int(ckpt.get("nhead", 4)),
        num_layers=int(ckpt.get("num_layers", 5)),
        dim_feedforward=int(ckpt.get("dim_feedforward", 256)),
        dropout=float(ckpt.get("dropout", 0.25)),
    )
    model.load_state_dict(state)
    model.eval().to(device)
    return model, classes, window, in_ch


def _newest(pattern: str) -> Path | None:
    if not _CHECKPOINTS.is_dir():
        return None
    cands = list(_CHECKPOINTS.glob(pattern))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


# =============================================================================
# Batched two-stage inference
# =============================================================================

@torch.no_grad()
def _run_batched(
    model: PunchTransformer,
    tensors: list[torch.Tensor],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on a subset of pre-built tensors selected by ``indices``."""
    n    = len(indices)
    pred = np.empty(n, dtype=np.int64)
    prob = np.empty(n, dtype=np.float32)
    k = 0
    while k < n:
        batch_idx = indices[k: k + batch_size]
        xb        = torch.stack([tensors[i] for i in batch_idx]).to(device)
        pr        = torch.softmax(model(xb), dim=-1).cpu().numpy()
        pi        = pr.argmax(axis=-1)
        pred[k: k + len(batch_idx)] = pi
        prob[k: k + len(batch_idx)] = pr[np.arange(len(batch_idx)), pi].astype(np.float32)
        k += len(batch_idx)
    return pred, prob


def cascade_predict(
    binary_model:  PunchTransformer,
    binary_classes: list[str],
    seven_model:   PunchTransformer,
    seven_classes: list[str],
    raw_xyz:       np.ndarray,
    window:        int,
    centers:       np.ndarray,
    device:        torch.device,
    batch_size:    int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays aligned to ``centers``:
      label_names  — display string per window
      final_prob   — confidence of the displayed prediction
      binary_pred  — 0 = punch, 1 = no_punch (from Stage 1)
      binary_prob  — Stage 1 confidence
    """
    n = len(centers)

    # ── pre-build all window tensors (preprocess once) ──────────────────────
    print(f"Preprocessing {n} windows …", end=" ", flush=True)
    tensors: list[torch.Tensor] = [
        torch.from_numpy(_preprocess_bio(_window_centered_at(raw_xyz, window, int(c))))
        for c in centers
    ]
    print("done")

    all_indices = np.arange(n)

    # ── Stage 1: binary ──────────────────────────────────────────────────────
    print(f"Stage 1 (binary) over {n} windows …", end=" ", flush=True)
    bin_pred, bin_prob = _run_batched(binary_model, tensors, all_indices, device, batch_size)
    print("done")

    binary_punch_idx = int(binary_classes.index("punch")) if "punch" in binary_classes else 0
    punch_mask = bin_pred == binary_punch_idx
    punch_indices = np.where(punch_mask)[0]

    # ── Stage 2: 7-class (only on punch windows) ─────────────────────────────
    type_pred = np.full(n, -1, dtype=np.int64)
    type_prob = np.zeros(n, dtype=np.float32)
    if len(punch_indices) > 0:
        print(f"Stage 2 (7-class) over {len(punch_indices)} punch windows …", end=" ", flush=True)
        tp, tpb = _run_batched(seven_model, tensors, punch_indices, device, batch_size)
        type_pred[punch_indices] = tp
        type_prob[punch_indices] = tpb
        print("done")
    else:
        print("Stage 2: skipped (no punch windows detected)")

    # ── Compose final labels ──────────────────────────────────────────────────
    label_names = np.empty(n, dtype=object)
    final_prob  = np.empty(n, dtype=np.float32)

    for i in range(n):
        if not punch_mask[i]:
            label_names[i] = "No Punch"
            final_prob[i]  = bin_prob[i]
        else:
            cls_idx = int(type_pred[i])
            label_names[i] = seven_classes[cls_idx].replace("_", " ").title()
            final_prob[i]  = type_prob[i]

    return label_names, final_prob, bin_pred, bin_prob


# =============================================================================
# Overlay
# =============================================================================

def _overlay_banner(
    frame: np.ndarray,
    line1: str,
    line2: str = "",
    line3: str = "",
) -> None:
    h, w = frame.shape[:2]
    bar_h = 44 + (22 if line2 else 0) + (20 if line3 else 0)
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
    y = 28
    cv2.putText(frame, line1[:220], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if line2:
        y += 22
        cv2.putText(frame, line2[:220], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2, cv2.LINE_AA)
    if line3:
        y += 20
        cv2.putText(frame, line3[:220], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)


def _make_pose_panel(xyz: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Draw H36M-17 skeleton from MotionBERT ``X3D[t]`` (17,3) on a BGR canvas (XY view)."""
    panel = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(17, 3)
    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
    jc = xyz - xyz[0:1]  # pelvis-centered
    u = jc[:, 0]
    v = -jc[:, 1]  # image Y downward
    span = float(max(np.ptp(u), np.ptp(v), 1e-4))
    pad = span * 0.12
    u_min, u_max = float(u.min() - pad), float(u.max() + pad)
    v_min, v_max = float(v.min() - pad), float(v.max() + pad)
    du = u_max - u_min + 1e-9
    dv = v_max - v_min + 1e-9

    def to_pt(ui: float, vi: float) -> tuple[int, int]:
        px = int((ui - u_min) / du * (out_w - 1))
        py = int((vi - v_min) / dv * (out_h - 1))
        return int(np.clip(px, 0, out_w - 1)), int(np.clip(py, 0, out_h - 1))

    pts = [to_pt(float(u[i]), float(v[i])) for i in range(17)]
    col_bone = (72, 200, 72)
    col_joint = (90, 210, 255)
    thick = max(2, min(out_h, out_w) // 200 + 1)
    for a, b in H36M_BONE_PAIRS:
        cv2.line(panel, pts[a], pts[b], col_bone, thick, cv2.LINE_AA)
    for i in range(17):
        cv2.circle(panel, pts[i], max(2, thick), col_joint, -1, cv2.LINE_AA)

    cv2.putText(
        panel,
        "MotionBERT X3D (XY, pelvis origin)",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return panel


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Two-stage cascade: binary punch detector → 7-class type classifier."
    )
    ap.add_argument("--ver",     type=str,  default="V7")
    ap.add_argument("--video",   type=Path, default=None)
    ap.add_argument("--x3d",     type=Path, default=None)
    ap.add_argument(
        "--binary", type=Path, default=None, metavar="PT",
        help="Binary checkpoint (default: newest punch_transformer_binary_bio_*.pt)",
    )
    ap.add_argument(
        "--seven", type=Path, default=None, metavar="PT",
        help="7-class checkpoint (default: newest punch_transformer_7cls_bio_*.pt)",
    )
    ap.add_argument("--start-frac", type=float, default=0.5)
    ap.add_argument("--end-frac",   type=float, default=1.0)
    ap.add_argument("--max-frames", type=int,   default=None)
    ap.add_argument("--stride",     type=int,   default=1)
    ap.add_argument("--batch-size", type=int,   default=256)
    ap.add_argument(
        "--slow-motion", type=float, default=1.0, metavar="FACTOR",
        help="Output FPS = source FPS / FACTOR (≥1).",
    )
    ap.add_argument(
        "--pose-panel-width",
        type=int,
        default=None,
        metavar="PX",
        help="Width of skeleton panel (default: ~38%% of video width, clamped 260–520).",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ver = args.ver.strip().upper()
    if not ver.startswith("V"):
        raise SystemExit("--ver should look like V7")
    if not (0 <= args.start_frac < args.end_frac <= 1):
        raise SystemExit("Need 0 ≤ --start-frac < --end-frac ≤ 1")
    sm = float(args.slow_motion)
    if sm < 1.0 or not np.isfinite(sm):
        raise SystemExit("--slow-motion must be ≥ 1")

    # ── Resolve video + pose paths ────────────────────────────────────────────
    video_path = args.video
    if video_path is None:
        video_path = _find_video(ver)
        if video_path is None:
            raise SystemExit(f"No video for {ver} in Dataset/RGB_videos/")
    else:
        video_path = video_path.resolve()
        if not video_path.is_file():
            raise SystemExit(f"Not found: {video_path}")

    x3d_path = (args.x3d or (_MOTIONBERT_DIR / ver / "X3D.npy")).resolve()
    if not x3d_path.is_file():
        raise SystemExit(f"Missing: {x3d_path}")

    # ── Resolve checkpoints ────────────────────────────────────────────────────
    bin_path = args.binary
    if bin_path is None:
        bin_path = _newest("punch_transformer_binary_bio_*.pt")
        if bin_path is None:
            raise SystemExit(
                "No punch_transformer_binary_bio_*.pt in checkpoints/ — "
                "train train_3d_classifier_binary_bio.py or pass --binary PATH"
            )
    else:
        bin_path = bin_path.resolve()
        if not bin_path.is_file():
            raise SystemExit(f"Not found: {bin_path}")

    seven_path = args.seven
    if seven_path is None:
        seven_path = _newest("punch_transformer_7cls_bio_*.pt")
        if seven_path is None:
            raise SystemExit(
                "No punch_transformer_7cls_bio_*.pt in checkpoints/ — "
                "train train_3d_classifier_bio.py or pass --seven PATH"
            )
    else:
        seven_path = seven_path.resolve()
        if not seven_path.is_file():
            raise SystemExit(f"Not found: {seven_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Binary ckpt : {bin_path.relative_to(_REPO)}")
    print(f"7-class ckpt: {seven_path.relative_to(_REPO)}")

    binary_model,  binary_classes,  bin_window,  bin_ch   = _load_model(bin_path,   device)
    seven_model,   seven_classes,   seven_window, seven_ch = _load_model(seven_path, device)

    if bin_ch != 15 or seven_ch != 15:
        raise SystemExit(
            f"Both models must use 15-ch bio preprocessing "
            f"(binary has {bin_ch}, 7-class has {seven_ch})"
        )
    if bin_window != seven_window:
        print(
            f"Warning: binary window={bin_window} ≠ 7-class window={seven_window}; "
            "using binary window for all predictions."
        )
    window = bin_window

    print(f"Binary classes : {binary_classes}")
    print(f"7-class classes: {seven_classes}")
    print(f"Window={window}  scipy_sg={'on' if _HAS_SCIPY else 'off'}")

    # ── Load pose + video metadata ─────────────────────────────────────────────
    raw = np.load(x3d_path)
    if raw.ndim != 3 or raw.shape[1] != 17:
        raise SystemExit(f"Expected X3D (T,17,3), got {raw.shape}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    n_total = min(raw.shape[0], n_vid)
    if n_total < window:
        raise SystemExit(f"Need ≥{window} frames; got {n_total}")

    start_i  = int(np.floor(args.start_frac * n_total))
    end_i    = min(int(np.ceil(args.end_frac * n_total)), n_total)
    encode_n = end_i - start_i
    if args.max_frames is not None:
        encode_n = min(encode_n, int(args.max_frames))

    out_fps = fps / sm
    print(
        f"\nVideo: {video_path.name}  n_total={n_total}"
        f"  encoding [{start_i}, {start_i + encode_n})  stride={args.stride}"
        f"  src_fps={fps:.2f}  out_fps={out_fps:.2f}"
    )

    # ── Run cascade ────────────────────────────────────────────────────────────
    stride  = max(1, int(args.stride))
    sampled = np.arange(0, n_total, stride, dtype=np.int64)

    label_names, final_prob, bin_pred, bin_prob = cascade_predict(
        binary_model, binary_classes,
        seven_model,  seven_classes,
        raw, window, sampled, device, args.batch_size,
    )

    # Forward-fill strided predictions to every frame
    labels_all = np.empty(n_total, dtype=object)
    fprob_all  = np.empty(n_total, dtype=np.float32)
    bprob_all  = np.empty(n_total, dtype=np.float32)
    bpred_all  = np.empty(n_total, dtype=np.int64)
    si = 0
    for t in range(n_total):
        while si + 1 < len(sampled) and sampled[si + 1] <= t:
            si += 1
        labels_all[t] = label_names[si]
        fprob_all[t]  = final_prob[si]
        bprob_all[t]  = bin_prob[si]
        bpred_all[t]  = bin_pred[si]

    # ── Write output video ─────────────────────────────────────────────────────
    out_path = (
        args.out.resolve()
        if args.out is not None
        else (_FIGURES / f"{ver}_cascade_preview.mp4").resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel_w = args.pose_panel_width
    if panel_w is None:
        panel_w = int(np.clip(round(0.38 * fw), 260, 520))
    panel_w = max(120, panel_w)
    out_fw = fw + panel_w

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_fw, fh))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter failed: {out_path}")
    print(f"Output layout: {fw}x{fh} video + {panel_w}x{fh} pose panel → {out_fw}x{fh}")

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_i))

    binary_punch_idx = int(binary_classes.index("punch")) if "punch" in binary_classes else 0

    try:
        for j in range(encode_n):
            ok, frame = cap.read()
            if not ok:
                print(f"warning: short read at frame {j}", file=sys.stderr)
                break
            t   = start_i + j
            lbl = str(labels_all[t])
            fp  = float(fprob_all[t])
            bp  = float(bprob_all[t])
            is_punch = int(bpred_all[t]) == binary_punch_idx

            line1 = f"{ver} cascade  |  frame {t + 1}/{n_total}  |  stride={stride}"
            if is_punch:
                line2 = f"{lbl}   p={fp:.2f}  (det={bp:.2f})"
            else:
                line2 = f"No Punch   p={bp:.2f}"
            line3 = (
                f"binary: {bin_path.name}   7cls: {seven_path.name}"
                + (f"  slow={sm:g}x" if sm > 1.0 else "")
            )
            _overlay_banner(frame, line1, line2, line3)
            ti = min(t, raw.shape[0] - 1)
            pose_panel = _make_pose_panel(raw[ti], fh, panel_w)
            combo = np.hstack([frame, pose_panel])
            combo[:, fw - 1 : fw + 1] = (55, 55, 55)
            writer.write(combo)
    finally:
        writer.release()
        cap.release()

    try:
        rel = out_path.relative_to(_REPO)
    except ValueError:
        rel = out_path
    print(f"\nWrote → {rel}")


if __name__ == "__main__":
    main()
