#!/usr/bin/env python3
"""
End-to-end audit: every clip for a workbook’s version in landmarks.npz, drawn on the
source RGB video, concatenated into one MP4: figures/Vx-all_clips.mp4

Usage:
    python preview_npz_on_video.py --xlsx V1.xlsx
    python preview_npz_on_video.py --xlsx Dataset/Annotation_files/V3.xlsx --npz Dataset/landmarks.npz
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent
_DEFAULT_NPZ = _REPO / "Dataset" / "landmarks.npz"
_VIDEOS = _REPO / "Dataset" / "RGB_videos" / "source_youtube"
_FIGURES = _REPO / "figures"

# Same edges as preview_npz_labels.py (12-joint paper order)
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


def _version_from_xlsx_arg(xlsx_arg: str) -> str:
    p = Path(xlsx_arg)
    stem = p.stem if p.suffix.lower() == ".xlsx" else Path(xlsx_arg).name
    if not stem:
        raise ValueError(f"Could not parse version from {xlsx_arg!r}")
    if not stem.upper().startswith("V"):
        raise ValueError(f"Expected version like V1 (got stem {stem!r}); pass V1.xlsx or V1")
    return stem


def _find_video(version: str) -> Path | None:
    matches = sorted(glob.glob(str(_VIDEOS / f"{version}_*.mp4")))
    return Path(matches[0]) if matches else None


def _draw_pose_bgr(frame: np.ndarray, xy: np.ndarray) -> None:
    """Draw skeleton from normalised (0–1) x,y onto BGR frame in-place."""
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


def _overlay_banner(frame: np.ndarray, line1: str, line2: str = "") -> None:
    h, w = frame.shape[:2]
    bar_h = 56 if line2 else 44
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(frame, line1[:140], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if line2:
        cv2.putText(frame, line2[:140], (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)


def _write_clip_frames(
    cap: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    seq: np.ndarray,
    *,
    label: str,
    ver: str,
    start_1based: int,
    end_1based: int,
) -> None:
    T = int(seq.shape[0])
    i0 = start_1based - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)

    for t in range(T):
        ok, frame = cap.read()
        if not ok:
            print(f"  warning: could not read frame at index {i0 + t}", file=sys.stderr)
            break

        _draw_pose_bgr(frame, seq[t])
        g1 = start_1based + t
        line1 = f"{label}  |  {ver}  NPZ frames [{start_1based},{end_1based}]  current {g1}/{end_1based}"
        nan_j = int(np.sum(~np.all(np.isfinite(seq[t]), axis=1)))
        line2 = f"clip frame {t + 1}/{T}  |  joints NaN: {nan_j}/12"
        _overlay_banner(frame, line1, line2)
        writer.write(frame)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Write figures/Vx-all_clips.mp4 — all NPZ clips for that version on source video, concatenated."
    )
    ap.add_argument("--npz", type=Path, default=_DEFAULT_NPZ, help="path to landmarks.npz")
    ap.add_argument(
        "--xlsx",
        type=str,
        required=True,
        help="workbook name or path, e.g. V1.xlsx or Dataset/Annotation_files/V1.xlsx",
    )
    args = ap.parse_args()

    ver = _version_from_xlsx_arg(args.xlsx)
    npz_path = args.npz
    if not npz_path.exists():
        raise SystemExit(f"Not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    for k in ("sequences", "labels", "versions", "start_frames", "end_frames"):
        if k not in data:
            raise SystemExit(f"Missing {k!r} in {npz_path}")

    seqs = data["sequences"]
    labels = data["labels"]
    versions = np.asarray(data["versions"])
    starts = data["start_frames"]
    ends = data["end_frames"]

    mask = np.array([str(v) == ver for v in versions], dtype=bool)
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        raise SystemExit(f"No clips in {npz_path} with versions=={ver!r}")

    video_path = _find_video(ver)
    if video_path is None:
        raise SystemExit(
            f"No video found under {_VIDEOS} matching {ver}_*.mp4\n"
            "Place the source RGB MP4 there (same layout as preprocess.extract_landmarks)."
        )

    out_path = (_FIGURES / f"{ver}-all_clips.mp4").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"Video: {video_path.name}  |  {ver}: {len(idxs)} clip(s) → {out_path}\n")

    try:
        for local_i, si in enumerate(idxs):
            si = int(si)
            seq = np.asarray(seqs[si], dtype=np.float32)
            lab = str(labels[si])
            sf, ef = int(starts[si]), int(ends[si])
            T = seq.shape[0]
            expected = ef - sf + 1
            if T != expected:
                print(
                    f"  warning: idx={si} stored T={T} but annotation span suggests {expected} frames",
                    file=sys.stderr,
                )
            if local_i % 100 == 0:
                print(f"  idx={si}  label={lab!r}  frames [{sf},{ef}]")
            _write_clip_frames(
                cap,
                writer,
                seq,
                label=lab,
                ver=ver,
                start_1based=sf,
                end_1based=ef,
            )
    finally:
        writer.release()
        cap.release()

    print(f"\nDone. Wrote → {out_path}")


if __name__ == "__main__":
    main()
