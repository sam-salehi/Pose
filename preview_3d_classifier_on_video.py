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

Preprocessing: full sequence, body frame from video frame 0, torso scale, velocities —
then each frame uses a window centered on that frame (inference; no train-time jitter).

Example::

    python preview_3d_classifier_on_video.py --ver V7
    python preview_3d_classifier_on_video.py --ver V7 --no-punch
    python preview_3d_classifier_on_video.py --ver V7 --checkpoint checkpoints/punch_transformer_7cls_V1.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from GCN import H36M_BONE_PAIRS, PUNCH_CLASSES
from preprocess import _find_video
from punch_transformer import PunchTransformer

_REPO = Path(__file__).resolve().parent
_MOTIONBERT_DIR = _REPO / "Dataset" / "MotionBERT_3d"
_FIGURES = _REPO / "figures"
_CHECKPOINTS = _REPO / "checkpoints"

_J_PELVIS = 0
_J_THORAX = 8
_J_R_SHOULDER = 14
_J_L_SHOULDER = 11


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


def _preprocess_full_sequence(poses_xyz: np.ndarray) -> np.ndarray:
    """``poses_xyz`` (T, 17, 3) raw MotionBERT → (T, 17, 6); scaling matches ``train_3d_classifier_no_punch``."""
    q = _to_body_frame(poses_xyz.astype(np.float64))
    q = _scale_normalize(q)
    q = q.astype(np.float32)
    vel = np.zeros_like(q)
    vel[1:] = q[1:] - q[:-1]
    return np.nan_to_num(
        np.concatenate([q, vel], axis=-1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


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
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[PunchTransformer, list[str], int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state")
    if state is None:
        raise SystemExit(f"Checkpoint missing model_state: {ckpt_path}")
    classes: list[str] = list(ckpt.get("punch_classes", PUNCH_CLASSES))
    window = int(ckpt.get("window", 8))
    model = PunchTransformer(
        num_classes=len(classes),
        in_channels=6,
        edges=H36M_BONE_PAIRS,
        spatial_hidden=64,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    )
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, classes, window


@torch.no_grad()
def _predict_windows_batched(
    model: PunchTransformer,
    seq: np.ndarray,
    window: int,
    centers: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_idx [N], max_prob [N]) for each center index."""
    n = centers.shape[0]
    preds = np.empty(n, dtype=np.int64)
    probs = np.empty(n, dtype=np.float32)
    k = 0
    while k < n:
        batch_c = centers[k: k + batch_size]
        chunks = [_window_centered_at(seq, window, int(c)) for c in batch_c]
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
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ver = args.ver.strip().upper()
    if not ver.startswith("V"):
        raise SystemExit("--ver should look like V7")

    if args.start_frac < 0 or args.start_frac > 1 or args.end_frac <= args.start_frac:
        raise SystemExit("Need 0 ≤ --start-frac < --end-frac ≤ 1")

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
        if args.no_punch:
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
    model, punch_classes, clf_window = _load_model(ckpt_path, device)

    print(f"Checkpoint: {ckpt_path.relative_to(_REPO)}")
    print(f"Classes ({len(punch_classes)}): {punch_classes}  window={clf_window}")

    raw = np.load(x3d_path)
    if raw.ndim != 3 or raw.shape[1] != 17:
        raise SystemExit(f"Expected X3D (T,17,3), got {raw.shape}")

    n_pose = raw.shape[0]
    seq = _preprocess_full_sequence(raw)

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
    print(f"Encoding frames [{start_i}, {start_i + encode_n})  (stride={args.stride})")

    stride = max(1, int(args.stride))
    sampled = np.arange(0, n_total, stride, dtype=np.int64)
    pr_idx, pr_pb = _predict_windows_batched(
        model, seq, clf_window, sampled, device, args.batch_size,
    )

    pred_s = np.empty(n_total, dtype=np.int64)
    prob_s = np.empty(n_total, dtype=np.float32)
    si = 0
    for t in range(n_total):
        while si + 1 < len(sampled) and sampled[si + 1] <= t:
            si += 1
        pred_s[t] = pr_idx[si]
        prob_s[t] = pr_pb[si]

    out_path = (
        args.out.resolve()
        if args.out is not None
        else (_FIGURES / f"{ver}_3d_punch_transformer{'_7cls' if len(punch_classes) > 6 else ''}_preview.mp4").resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter failed: {out_path}")

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_i))

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
            line3 = f"window={clf_window}  stride={stride}  body_frame=video_frame0"
            _overlay_banner(frame, line1, line2, line3)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()

    try:
        print_rel = out_path.relative_to(_REPO)
    except ValueError:
        print_rel = out_path
    print(f"\nWrote → {print_rel}")


if __name__ == "__main__":
    main()
