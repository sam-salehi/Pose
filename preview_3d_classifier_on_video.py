#!/usr/bin/env python3
"""
Run the MotionBERT 3D PunchTransformer on a workbook video: load
``Dataset/MotionBERT_3d/{VER}/X3D.npy``, preprocess like training, sliding-window
classification, overlay predicted class + probability on the RGB MP4.

**Six punch types vs seven (including no punch)**

- By default, preview picks the newest ``punch_transformer_*.pt`` that is **not**
  ``*_gap_review.pt`` or ``*7cls*`` (intended as the 6 punch-type-only head). If you only
  trained the 7-class pipeline (e.g. ``*_gap_review.pt``), preview falls back to that
  model automatically.
- Pass ``--no-punch`` to **prefer** the newest 7-class checkpoint among
  ``*_gap_review.pt`` and ``punch_transformer_7cls_*.pt``.
- Use ``--no-punch`` to select a **7-class** checkpoint: newest of
  ``punch_transformer_*_gap_review.pt`` (from ``train_3d_classifier.py`` with
  ``Dataset/gap_labels/gap_review.json`` negatives) or ``punch_transformer_7cls_*.pt``
  (from ``train_3d_classifier_no_punch.py``).

Default: encode **from the middle** (``--start-frac 0.5``); use ``--start-frac 0`` for full video.

Preprocessing matches ``train_3d_classifier_no_punch``: each sliding window uses **only**
that window's raw ``X3D`` frames, then body frame + median torso scale + velocity with
**reference = first frame of the window** (not global video frame 0).

Example::

    python preview_3d_classifier_on_video.py --ver V7
    python preview_3d_classifier_on_video.py --ver V7 --no-punch
    python preview_3d_classifier_on_video.py --ver V7 --checkpoint checkpoints/punch_transformer_7cls_V1.pt
    python preview_3d_classifier_on_video.py --ver V7 --slow-motion 2

Biomechanics 7-class model (``train_3d_classifier_bio.py``, 15 input channels)::

    python preview_3d_classifier_on_video.py --ver V7 \\
        --checkpoint checkpoints/punch_transformer_7cls_bio_<versions>.pt

4-class biomechanics (``train_4cls_bio.py``) on a held-out workbook, **live window** (no MP4 unless ``--out``)::

    python preview_3d_classifier_on_video.py --ver V4 --live --start-frac 0 \\
        --checkpoint checkpoints/punch_transformer_4cls_bio_V10_V5_V6_V7_V8_V9.pt
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from skeleton_constants import H36M_BONE_PAIRS, PUNCH_CLASSES
from preprocess import _find_video
from punch_transformer import PunchTransformer

try:
    from scipy.signal import savgol_filter as _savgol_fn
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_REPO = Path(__file__).resolve().parent
_MOTIONBERT_DIR = _REPO / "Dataset" / "MotionBERT_3d"
_FIGURES = _REPO / "figures"
_CHECKPOINTS = _REPO / "checkpoints"

_J_PELVIS = 0
_J_THORAX = 8
_J_R_HIP = 1
_J_L_HIP = 4
_J_R_SHOULDER = 14
_J_L_SHOULDER = 11
_J_L_ELBOW = 12
_J_L_WRIST = 13
_J_R_ELBOW = 15
_J_R_WRIST = 16

_N_SCALARS_BIO = 6


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


def _preprocess_clip(clip_xyz: np.ndarray) -> np.ndarray:
    """Raw window ``(T, 17, 3)`` → ``(T, 17, 6)`` — same contract as training ``_preprocess_clip``."""
    q = _to_body_frame(clip_xyz.astype(np.float64))
    q = _scale_normalize(q)
    q = q.astype(np.float32)
    vel = np.zeros_like(q)
    vel[1:] = q[1:] - q[:-1]
    return np.nan_to_num(
        np.concatenate([q, vel], axis=-1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def _smooth_bio(q: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    T = q.shape[0]
    if not _HAS_SCIPY or T < window:
        return q
    po = min(polyorder, window - 1)
    flat = q.reshape(T, -1)
    return _savgol_fn(flat, window_length=window, polyorder=po, axis=0).reshape(q.shape)


def _angle3_bio(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    cos_a = np.clip(np.einsum("ti,ti->t", ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cos_a)


def _preprocess_clip_bio(clip_xyz: np.ndarray) -> np.ndarray:
    """``(T, 17, 3)`` → ``(T, 17, 15)`` — matches ``train_3d_classifier_bio._preprocess_clip``."""
    q = _to_body_frame(clip_xyz.astype(np.float64))
    q = _scale_normalize(q)
    q_sm = _smooth_bio(q)

    T = q_sm.shape[0]
    if T > 1:
        vel = np.gradient(q_sm, axis=0)
        acc = np.gradient(vel, axis=0)
    else:
        vel = np.zeros_like(q_sm)
        acc = np.zeros_like(q_sm)

    l_elbow = _angle3_bio(q_sm[:, _J_L_SHOULDER], q_sm[:, _J_L_ELBOW], q_sm[:, _J_L_WRIST])
    r_elbow = _angle3_bio(q_sm[:, _J_R_SHOULDER], q_sm[:, _J_R_ELBOW], q_sm[:, _J_R_WRIST])
    hip_vec = q_sm[:, _J_R_HIP] - q_sm[:, _J_L_HIP]
    sh_vec = q_sm[:, _J_R_SHOULDER] - q_sm[:, _J_L_SHOULDER]
    hip_yaw = np.arctan2(hip_vec[:, 1], hip_vec[:, 0])
    shoulder_yaw = np.arctan2(sh_vec[:, 1], sh_vec[:, 0])
    xfactor = shoulder_yaw - hip_yaw
    com_z = q_sm[:, :, 2].mean(axis=1)

    scalars = np.stack(
        [l_elbow, r_elbow, hip_yaw, shoulder_yaw, xfactor, com_z], axis=-1
    )
    scalars_bc = np.broadcast_to(
        scalars[:, np.newaxis, :], (T, 17, _N_SCALARS_BIO)
    ).copy()
    out = np.concatenate([q_sm, vel, acc, scalars_bc], axis=-1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _window_centered_at(seq: np.ndarray, window: int, center_idx: int) -> np.ndarray:
    """Extract ``window`` frames centred on ``center_idx`` (pad edges like training)."""
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
        center_idx = center_idx + pad_pre

    peak = int(center_idx)
    peak = max(half, min(T - (window - half), peak))
    start = peak - half
    chunk = seq[start: start + window]
    if chunk.shape[0] < window:
        pad = window - chunk.shape[0]
        chunk = np.concatenate([chunk, np.tile(chunk[[-1]], (pad, 1, 1))], axis=0)
    return chunk


def _overlay_banner(frame: np.ndarray, line1: str, line2: str = "", line3: str = "") -> None:
    h, w = frame.shape[:2]
    bar_h = 44 + (20 if line2 else 0) + (20 if line3 else 0)
    bar_h = max(bar_h, 44)
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
    y = 28
    cv2.putText(
        frame, line1[:220], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
    )
    if line2:
        y += 22
        cv2.putText(
            frame, line2[:220], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 2, cv2.LINE_AA,
        )
    if line3:
        y += 22
        cv2.putText(
            frame, line3[:220], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA,
        )


def _default_checkpoint_six_class() -> Path | None:
    """Newest ``punch_transformer_*.pt`` excluding 7-class no_punch checkpoints."""
    if not _CHECKPOINTS.is_dir():
        return None
    cands = [
        p
        for p in _CHECKPOINTS.glob("punch_transformer_*.pt")
        if "7cls" not in p.name and "gap_review" not in p.name
    ]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _default_checkpoint_seven_class() -> Path | None:
    """Newest 7-class checkpoint: ``*_gap_review.pt`` or ``punch_transformer_7cls_*.pt``."""
    if not _CHECKPOINTS.is_dir():
        return None
    cands = list(_CHECKPOINTS.glob("punch_transformer_*_gap_review.pt")) + list(
        _CHECKPOINTS.glob("punch_transformer_7cls_*.pt")
    )
    # Exclude bio filenames — they need 15-ch preprocessing.
    cands = [p for p in cands if "_bio_" not in p.name]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _default_checkpoint_bio() -> Path | None:
    """Newest ``punch_transformer_7cls_bio_*.pt`` (15-channel biomechanics model)."""
    if not _CHECKPOINTS.is_dir():
        return None
    cands = list(_CHECKPOINTS.glob("punch_transformer_7cls_bio_*.pt"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[PunchTransformer, list[str], int, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state")
    if state is None:
        raise SystemExit(f"Checkpoint missing model_state: {ckpt_path}")
    classes: list[str] = list(ckpt.get("punch_classes", PUNCH_CLASSES))
    window = int(ckpt.get("window", 8))
    # Architecture must match training; older checkpoints omit these keys (default = small base model).
    sh = int(ckpt.get("spatial_hidden", 64))
    dm = int(ckpt.get("d_model", 128))
    nh = int(ckpt.get("nhead", 4))
    nl = int(ckpt.get("num_layers", 4))
    df = int(ckpt.get("dim_feedforward", 256))
    do = float(ckpt.get("dropout", 0.2))
    in_ch = int(ckpt.get("in_channels", 6))
    model = PunchTransformer(
        num_classes=len(classes),
        in_channels=in_ch,
        edges=H36M_BONE_PAIRS,
        spatial_hidden=sh,
        d_model=dm,
        nhead=nh,
        num_layers=nl,
        dim_feedforward=df,
        dropout=do,
    )
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, classes, window, in_ch


@torch.no_grad()
def _predict_windows_batched(
    model: PunchTransformer,
    raw_xyz: np.ndarray,
    window: int,
    centers: np.ndarray,
    device: torch.device,
    batch_size: int,
    preprocess: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_idx [N], max_prob [N]) for each center index.

    ``raw_xyz`` is ``(T, 17, 3)`` MotionBERT output. Each window is preprocessed
    **locally** (body frame from window start) like training.
    """
    n = centers.shape[0]
    preds = np.empty(n, dtype=np.int64)
    probs = np.empty(n, dtype=np.float32)
    k = 0
    while k < n:
        batch_c = centers[k: k + batch_size]
        chunks = [
            preprocess(_window_centered_at(raw_xyz, window, int(c)))
            for c in batch_c
        ]
        xb = torch.stack([torch.from_numpy(c) for c in chunks]).to(device)
        logits = model(xb)
        pr = torch.softmax(logits, dim=-1).cpu().numpy()
        pi = pr.argmax(axis=-1)
        pc = pr[np.arange(pr.shape[0]), pi]
        preds[k: k + len(batch_c)] = pi
        probs[k: k + len(batch_c)] = pc.astype(np.float32)
        k += len(batch_c)
    return preds, probs


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay 3D punch-transformer predictions on RGB video.")
    ap.add_argument("--ver", type=str, default="V7", help="Workbook tag (default V7)")
    ap.add_argument("--video", type=Path, default=None, help="Override RGB path (default: _find_video)")
    ap.add_argument("--x3d", type=Path, default=None, help="Override X3D.npy path")
    ap.add_argument(
        "--no-punch",
        action="store_true",
        help="Use 7-class checkpoint (includes no_punch): gap_review or 7cls .pt",
    )
    ap.add_argument(
        "--bio",
        action="store_true",
        help="Use newest checkpoints/punch_transformer_7cls_bio_*.pt (15-ch; matches train_3d_classifier_bio.py)",
    )
    ap.add_argument("--checkpoint", type=Path, default=None, help="Explicit .pt (overrides --no-punch default)")
    ap.add_argument(
        "--start-frac",
        type=float,
        default=0.5,
        help="Start encoding from this fraction of the timeline (0–1). Default 0.5 = middle.",
    )
    ap.add_argument("--end-frac", type=float, default=1.0, help="End fraction (default 1 = end)")
    ap.add_argument("--max-frames", type=int, default=None, help="Cap number of output frames")
    ap.add_argument("--stride", type=int, default=1, help="Classifier stride ≥1 (2 = faster, labels forward-filled)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--slow-motion",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Slow playback: output FPS = source FPS / FACTOR (FACTOR≥1). "
        "Example: 2 → half speed, wall-clock duration doubles.",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--live",
        action="store_true",
        help="Show playback in an OpenCV window (press q or Esc to stop). "
        "If --out is omitted, no MP4 is written (encode-only then display).",
    )
    args = ap.parse_args()

    ver = args.ver.strip().upper()
    if not ver.startswith("V"):
        raise SystemExit("--ver should look like V7")

    if args.start_frac < 0 or args.start_frac > 1 or args.end_frac <= args.start_frac:
        raise SystemExit("Need 0 ≤ --start-frac < --end-frac ≤ 1")

    sm = float(args.slow_motion)
    if sm < 1.0 or not np.isfinite(sm):
        raise SystemExit("--slow-motion must be a finite number ≥ 1 (1 = real-time)")

    video_path = args.video
    if video_path is None:
        vp = _find_video(ver)
        if vp is None:
            raise SystemExit(f"No video Dataset/RGB_videos/source_youtube/{ver}_*.mp4")
        video_path = vp
    else:
        video_path = video_path.resolve()
        if not video_path.is_file():
            raise SystemExit(f"Not found: {video_path}")

    x3d_path = args.x3d
    if x3d_path is None:
        x3d_path = (_MOTIONBERT_DIR / ver / "X3D.npy").resolve()
    else:
        x3d_path = x3d_path.resolve()
    if not x3d_path.is_file():
        raise SystemExit(f"Missing MotionBERT array: {x3d_path}")

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        if args.bio:
            d = _default_checkpoint_bio()
            if d is None:
                raise SystemExit(
                    "No punch_transformer_7cls_bio_*.pt in checkpoints/ — train "
                    "train_3d_classifier_bio.py or pass --checkpoint PATH"
                )
            ckpt_path = d
        elif args.no_punch:
            d = _default_checkpoint_seven_class()
            if d is None:
                raise SystemExit(
                    "No 7-class checkpoint — train train_3d_classifier.py (saves "
                    "*_gap_review.pt) or train_3d_classifier_no_punch.py (7cls), "
                    "or pass --checkpoint PATH"
                )
            ckpt_path = d
        else:
            d = _default_checkpoint_six_class()
            if d is None:
                d7 = _default_checkpoint_seven_class()
                if d7 is not None:
                    print(
                        "No 6-class-only checkpoint; using 7-class model:\n  "
                        f"{d7.relative_to(_REPO)}"
                    )
                    d = d7
            if d is None:
                raise SystemExit(
                    "No checkpoints/punch_transformer_*.pt — train (e.g. train_3d_classifier.py) "
                    "or pass --checkpoint PATH. Use --no-punch to prefer 7-class / no_punch."
                )
            ckpt_path = d
    else:
        ckpt_path = ckpt_path.resolve()
        if not ckpt_path.is_file():
            raise SystemExit(f"Not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, punch_classes, clf_window, in_ch = _load_model(ckpt_path, device)

    if in_ch == 15:
        preprocess = _preprocess_clip_bio
        print(f"Preprocessing: bio (15 ch)  scipy_sg={'on' if _HAS_SCIPY else 'off'}")
    elif in_ch == 6:
        preprocess = _preprocess_clip
    else:
        raise SystemExit(
            f"Preview supports in_channels 6 or 15 only; checkpoint has in_channels={in_ch}"
        )

    print(f"Checkpoint: {ckpt_path.relative_to(_REPO)}")
    print(f"Classes ({len(punch_classes)}): {punch_classes}  window={clf_window}  in_channels={in_ch}")

    raw = np.load(x3d_path)
    if raw.ndim != 3 or raw.shape[1] != 17:
        raise SystemExit(f"Expected X3D (T,17,3), got {raw.shape}")

    n_pose = raw.shape[0]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    n_total = min(n_pose, n_vid)
    if n_total < clf_window:
        raise SystemExit(f"Need at least {clf_window} frames; got min(pose,video)={n_total}")

    start_i = int(np.floor(args.start_frac * n_total))
    end_i = int(np.ceil(args.end_frac * n_total))
    end_i = max(start_i + 1, min(end_i, n_total))

    encode_n = end_i - start_i
    if args.max_frames is not None:
        encode_n = min(encode_n, int(args.max_frames))

    print(f"Video: {video_path.name}  pose_rows={n_pose}  video_frames={n_vid}  using_n={n_total}")
    out_fps = fps / sm
    print(
        f"Encoding frames [{start_i}, {start_i + encode_n})  "
        f"(stride={args.stride})  source_fps={fps:.3f}  output_fps={out_fps:.3f}  slow_motion={sm:g}x"
    )

    stride = max(1, int(args.stride))
    sampled = np.arange(0, n_total, stride, dtype=np.int64)
    pr_idx, pr_pb = _predict_windows_batched(
        model, raw, clf_window, sampled, device, args.batch_size, preprocess,
    )

    pred_s = np.empty(n_total, dtype=np.int64)
    prob_s = np.empty(n_total, dtype=np.float32)
    si = 0
    for t in range(n_total):
        while si + 1 < len(sampled) and sampled[si + 1] <= t:
            si += 1
        pred_s[t] = pr_idx[si]
        prob_s[t] = pr_pb[si]

    if args.live and args.out is None:
        out_path = None
    else:
        out_path = (
            args.out.resolve()
            if args.out is not None
            else (
                _FIGURES
                / f"{ver}_3d_punch_transformer{'_bio' if in_ch == 15 else ''}{'_7cls' if len(punch_classes) > 6 else ''}_preview.mp4"
            ).resolve()
        )

    writer: cv2.VideoWriter | None = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (fw, fh))
        if not writer.isOpened():
            raise SystemExit(f"VideoWriter failed: {out_path}")

    if args.live:
        delay_ms = max(1, int(round(1000.0 * sm / max(fps, 1e-6))))
        print(f"Live display: delay_ms≈{delay_ms} (q or Esc to quit)")

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_i))

    win = "3d_punch_classifier_preview"
    try:
        for j in range(encode_n):
            ok, frame = cap.read()
            if not ok:
                print(f"warning: short read at output frame {j}", file=sys.stderr)
                break
            t = start_i + j
            pi = int(pred_s[t])
            pc = float(prob_s[t])
            name = punch_classes[pi].replace("_", " ").title()
            line1 = f"{ver} 3D clf  |  video frame {t + 1}/{n_total}  |  ckpt {ckpt_path.name}"
            line2 = f"{name}   p={pc:.2f}"
            line3 = (
                f"window={clf_window}  stride={stride}  body_frame=window_local"
                + (f"  slow={sm:g}x" if sm > 1.0 else "")
            )
            _overlay_banner(frame, line1, line2, line3)
            if writer is not None:
                writer.write(frame)
            if args.live:
                cv2.imshow(win, frame)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    print("Stopped by user (q/Esc).")
                    break
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        if args.live:
            try:
                cv2.destroyWindow(win)
            except Exception:
                cv2.destroyAllWindows()

    if out_path is not None:
        try:
            print_rel = out_path.relative_to(_REPO)
        except ValueError:
            print_rel = out_path
        print(f"\nWrote → {print_rel}")
    elif args.live:
        print("\nLive session finished (no file written; pass --out PATH.mp4 to save).")


if __name__ == "__main__":
    main()
