#!/usr/bin/env python3
"""
Visual audit for detector preprocessing (train.ipynb detection npz + cached poses).

Reads ``Dataset/detection_frame_labels/{ver}_detection.npz`` and pre-extracted
``Dataset/pose_sequences/{ver}_pose.npz`` (key ``pose``), draws the 12-joint
skeleton on each RGB frame, and overlays frame-level punch labels and sliding-window
coverage.

Writes ``figures/{ver}-detection_preview.mp4`` by default.

Usage:
    python preview_detection_on_video.py --ver V7
    python preview_detection_on_video.py --ver v7 --max-frames 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from detector_data import load_pose_sequence_npz
from preprocess import _find_video

_REPO = Path(__file__).resolve().parent
_DEFAULT_DET_DIR = _REPO / "Dataset" / "detection_frame_labels"
_POSE_SEQ_DIR = _REPO / "Dataset" / "pose_sequences"
_FIGURES = _REPO / "figures"

_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (0, 6),
    (1, 7),
    (6, 7),
    (6, 8),
    (8, 10),
    (7, 9),
    (9, 11),
]


def _draw_pose_bgr(frame: np.ndarray, xy: np.ndarray) -> None:
    h, w = frame.shape[:2]
    xy = np.asarray(xy, dtype=np.float64)
    pts: list[tuple[int, int] | None] = []
    for j in range(12):
        if np.all(np.isfinite(xy[j])):
            px = int(np.clip(xy[j, 0], 0.0, 1.0) * (w - 1))
            py = int(np.clip(xy[j, 1], 0.0, 1.0) * (h - 1))
            pts.append((px, py))
        else:
            pts.append(None)

    for i, j in _EDGES:
        a, b = pts[i], pts[j]
        if a is None or b is None:
            continue
        cv2.line(frame, a, b, (99, 67, 234), 2, cv2.LINE_AA)

    for j in range(12):
        p = pts[j]
        if p is None:
            continue
        cv2.circle(frame, p, 5, (38, 38, 220), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 5, (255, 255, 255), 1, cv2.LINE_AA)


def _overlay_banner(frame: np.ndarray, line1: str, line2: str = "", line3: str = "") -> None:
    h, w = frame.shape[:2]
    bar_h = 44 + (18 if line2 else 0) + (18 if line3 else 0)
    bar_h = max(bar_h, 44)
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
    y = 28
    cv2.putText(frame, line1[:160], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    if line2:
        y += 22
        cv2.putText(frame, line2[:160], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    if line3:
        y += 22
        cv2.putText(frame, line3[:160], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 180), 1, cv2.LINE_AA)


def _windows_covering_frame(
    t: int,
    window_starts: np.ndarray,
    window_length: int,
    window_is_punch: np.ndarray,
) -> tuple[int, int]:
    inside = (window_starts <= t) & (t < window_starts + window_length)
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return 0, 0
    return int(idx.size), int(window_is_punch[idx].max())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render detection labels + pose on source MP4 (audit train.ipynb preprocessing)."
    )
    ap.add_argument("--ver", type=str, required=True, help="Version tag, e.g. V7 or v7")
    ap.add_argument(
        "--detection",
        type=Path,
        default=None,
        help="Path to {ver}_detection.npz (default: Dataset/detection_frame_labels/)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output MP4 (default: {_FIGURES}/{{ver}}-detection_preview.mp4)",
    )
    ap.add_argument(
        "--poses",
        type=Path,
        default=None,
        help=f"Cached pose npz (default: {_POSE_SEQ_DIR}/{{ver}}_pose.npz, key 'pose')",
    )
    ap.add_argument("--max-frames", type=int, default=None, help="Only encode first N frames (quick check).")
    args = ap.parse_args()

    ver_raw = args.ver.strip()
    if not ver_raw.upper().startswith("V"):
        raise SystemExit("--ver should look like V7")
    ver = ver_raw.upper()

    det_path = args.detection if args.detection is not None else _DEFAULT_DET_DIR / f"{ver}_detection.npz"
    if not det_path.exists():
        raise SystemExit(f"Not found: {det_path}\nRun the train.ipynb cell that writes detection_frame_labels.")

    data = np.load(det_path, allow_pickle=True)
    for k in ("punch_frame_binary", "num_frames", "window_length", "window_starts", "window_is_punch"):
        if k not in data:
            raise SystemExit(f"Missing {k!r} in {det_path}")

    punch_frame = np.asarray(data["punch_frame_binary"], dtype=np.uint8)
    nf = int(data["num_frames"])
    wl = int(data["window_length"])
    w_starts = np.asarray(data["window_starts"])
    w_punch = np.asarray(data["window_is_punch"])

    if punch_frame.shape[0] != nf:
        raise SystemExit(f"Length mismatch: punch_frame_binary {punch_frame.shape[0]} vs num_frames {nf}")

    video_path = _find_video(ver)
    if video_path is None:
        raise SystemExit(
            f"No video under Dataset/RGB_videos/source_youtube/{ver}_*.mp4 — place the source MP4 there."
        )

    out_path = args.out if args.out is not None else (_FIGURES / f"{ver}-detection_preview.mp4").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encode_n = nf if args.max_frames is None else min(nf, int(args.max_frames))

    pose_path = args.poses if args.poses is not None else _POSE_SEQ_DIR / f"{ver}_pose.npz"
    if not pose_path.exists():
        raise SystemExit(
            f"Missing pose cache: {pose_path}\n"
            "Save MediaPipe extract as Dataset/pose_sequences/{ver}_pose.npz (numpy key 'pose')."
        )

    print(f"Detection npz: {det_path.relative_to(_REPO)}")
    print(f"Pose cache: {pose_path.relative_to(_REPO)}")
    print(f"Video: {video_path.name}")
    print(f"Frames in npz: {nf}  |  encoding: {encode_n}  |  window_length={wl}  |  windows={len(w_punch)}")
    if "num_frames_full_video" in data:
        print(f"  (truncated from full MP4 {int(data['num_frames_full_video'])} frames)")

    poses = load_pose_sequence_npz(pose_path, max_frames=nf)
    if poses.shape[0] < encode_n:
        print(
            f"warning: pose rows={poses.shape[0]} < encode_n={encode_n} — video may be shorter than npz.",
            file=sys.stderr,
        )
        encode_n = min(encode_n, poses.shape[0])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"VideoWriter failed for {out_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        for t in range(encode_n):
            ok, frame = cap.read()
            if not ok:
                print(f"warning: could not read frame {t}", file=sys.stderr)
                break

            _draw_pose_bgr(frame, poses[t])

            pf = int(punch_frame[t])
            frame_word = "PUNCH frame" if pf else "no punch (frame)"

            n_cov, win_mx = _windows_covering_frame(t, w_starts, wl, w_punch)
            win_word = "yes" if win_mx else "no"
            line1 = f"{ver}  |  frame {t + 1}/{nf}  |  {frame_word}"
            line2 = f"sliding windows covering this frame: {n_cov}  |  any punch-window: {win_word}"
            line3 = (
                "Excel timeline → punch_frame_binary  |  joints NaN: "
                f"{int(np.sum(~np.all(np.isfinite(poses[t]), axis=1)))}/12"
            )

            _overlay_banner(frame, line1, line2, line3)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()

    print(f"\nDone. Wrote → {out_path}")


if __name__ == "__main__":
    main()
