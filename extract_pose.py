#!/usr/bin/env python3
"""
Extract full-video pose sequences with MediaPipe (BoxingVI 12 joints).

Writes ``Dataset/pose_sequences/{V*}_pose.npz`` with array ``pose`` of shape
``(F, 12, 2)`` float32 — same layout as consumed by ``train.py`` / ``test_detection.py``.

Examples:

    python extract_pose.py V7
    python extract_pose.py V7 V8 --force
    python extract_pose.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from preprocess import _LANDMARK_IDX, _MODEL_PATH, _ensure_model, _find_video

_REPO = Path(__file__).resolve().parent
_POSE_SEQ_DIR = _REPO / "Dataset" / "pose_sequences"


def extract_full_video_poses(
    versions: list[str] | None = None,
    *,
    force: bool = False,
    out_dir: Path | None = None,
    min_confidence: float = 0.4,
) -> None:
    """
    Run MediaPipe on every frame of each video; save ``(F, 12, 2)`` arrays.

    Unless ``force``, skips versions whose ``*_pose.npz`` already exists.

    Output: ``{out_dir or pose_sequences}/{ver}_pose.npz`` with key ``pose``.
    """
    out_root = out_dir if out_dir is not None else _POSE_SEQ_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    _ensure_model()

    if versions is None:
        versions = [f"V{i}" for i in range(1, 11)]

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for ver in versions:
            out = out_root / f"{ver}_pose.npz"
            if out.exists() and not force:
                print(f"[{ver}] pose cache exists — skip (use --force to overwrite)")
                continue

            vid = _find_video(ver)
            if vid is None:
                print(f"[{ver}] no video found — skip", file=sys.stderr)
                continue

            cap = cv2.VideoCapture(str(vid))
            F = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[{ver}] extracting poses for {F} frames …")

            pose = np.zeros((F, 12, 2), dtype=np.float32)
            for fi in range(F):
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )
                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    for j, mp_idx in enumerate(_LANDMARK_IDX):
                        lm = lms[mp_idx]
                        pose[fi, j, 0] = lm.x
                        pose[fi, j, 1] = lm.y
                if (fi + 1) % 1000 == 0:
                    print(f"  {fi + 1}/{F} frames done")

            cap.release()
            np.savez_compressed(out, pose=pose)
            print(f"[{ver}] saved → {out.relative_to(_REPO)}  (shape {pose.shape})")


def _normalize_ver(s: str) -> str:
    t = s.strip().upper()
    if not t.startswith("V") or len(t) < 2:
        raise argparse.ArgumentTypeError(f"expected version like V7, got {s!r}")
    return t


def _parse_vers() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MediaPipe full-video pose extract → Dataset/pose_sequences/V*_pose.npz"
    )
    p.add_argument(
        "versions",
        nargs="*",
        type=_normalize_ver,
        help="Workbook tags, e.g. V7 V8 (omit with --all)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Extract V1 through V10 (same as passing all ten).",
    )
    p.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing npz files.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {_POSE_SEQ_DIR.relative_to(_REPO)})",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.4,
        help="MediaPipe detection / presence / tracking confidence (default 0.4).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_vers()
    if args.all and args.versions:
        print("Do not pass both --all and explicit versions.", file=sys.stderr)
        sys.exit(2)
    if not args.all and not args.versions:
        print("Pass one or more versions (e.g. V7) or use --all.", file=sys.stderr)
        sys.exit(2)

    if args.all:
        vers = [f"V{i}" for i in range(1, 11)]
    else:
        vers = list(args.versions)

    extract_full_video_poses(
        vers,
        force=args.force,
        out_dir=args.out_dir,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
