#!/usr/bin/env python3
"""
Run MotionBERT 3D lifting (MotionBERT/infer_wild.py) for **one** workbook video.

Resolves the RGB clip::

    Dataset/RGB_videos/source_youtube/{VER}_*.mp4

Writes **both** (from infer_wild)::

    Dataset/MotionBERT_3d/{VER}/X3D.mp4
    Dataset/MotionBERT_3d/{VER}/X3D.npy

Use ``--skip-render`` on this launcher (or ``--skip_render`` on ``infer_wild.py``) to write **only** ``X3D.npy`` and skip the slow MP4.

Run from the pose repo root with the same Python env you use for MotionBERT.

Example::

    conda activate motionbert-gpu
    cd /path/to/pose
    python run_motionbert_3d_batch.py V1
    python run_motionbert_3d_batch.py --ver V7
    python run_motionbert_3d_batch.py V1 --pose-backend yolo   #Ultralytics COCO17→H36M (needs pip install ultralytics)
    python run_motionbert_3d_batch.py V7 --skip-render        # npy only, no matplotlib export
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    motionbert_dir = repo_root / "MotionBERT"
    rgb_dir = repo_root / "Dataset" / "RGB_videos" / "source_youtube"
    data_out_root = repo_root / "Dataset" / "MotionBERT_3d"

    ap = argparse.ArgumentParser(
        description="Run MotionBERT 3D pose on one workbook (V1…V10) and save mp4 + npy."
    )
    ap.add_argument(
        "ver_pos",
        nargs="?",
        default=None,
        help="Workbook tag, e.g. V1 (optional if --ver is used)",
    )
    ap.add_argument(
        "--ver",
        type=str,
        default=None,
        help="Workbook tag, e.g. V7 (alternative to positional)",
    )
    ap.add_argument(
        "--pose-backend",
        type=str,
        choices=("mediapipe", "yolo"),
        default="mediapipe",
        help="2D pose source when calling infer_wild without JSON (default: mediapipe)",
    )
    ap.add_argument(
        "--yolo-weights",
        type=str,
        default="yolo11n-pose.pt",
        help="Ultralytics weights when --pose-backend yolo",
    )
    ap.add_argument(
        "--yolo-device",
        type=str,
        default=None,
        help="Device for YOLO, e.g. 0 or cpu (optional)",
    )
    ap.add_argument(
        "--skip-render",
        dest="skip_render",
        action="store_true",
        help="Pass --skip_render to infer_wild: only X3D.npy, no X3D.mp4",
    )
    args = ap.parse_args()

    raw = args.ver if args.ver is not None else args.ver_pos
    if raw is None:
        ap.error("pass a workbook tag, e.g. V1 or --ver V1")
    ver = raw.strip().upper()
    if not ver.startswith("V") or not ver[1:].isdigit():
        raise SystemExit(f"Expected tag like V1, got {raw!r}")

    if not motionbert_dir.is_dir():
        raise SystemExit(f"MotionBERT not found: {motionbert_dir}")
    infer_script = motionbert_dir / "infer_wild.py"
    if not infer_script.is_file():
        raise SystemExit(f"Missing {infer_script}")

    matches = sorted(glob.glob(str(rgb_dir / f"{ver}_*.mp4")))
    if not matches:
        raise SystemExit(f"No MP4 found for {ver} under {rgb_dir}/")

    vid_path = Path(matches[0])
    out_path = data_out_root / ver
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(infer_script),
        "--vid_path",
        str(vid_path),
        "--out_path",
        str(out_path),
        "--pose_backend",
        args.pose_backend,
        "--yolo_weights",
        args.yolo_weights,
    ]
    if args.yolo_device:
        cmd.extend(["--yolo_device", args.yolo_device])
    if args.skip_render:
        cmd.append("--skip_render")
    env = os.environ.copy()

    print(f"Video:     {vid_path.relative_to(repo_root)}")
    print(f"Output:    {out_path.relative_to(repo_root)}/")
    print(f"Command:   {' '.join(cmd)}")
    print(f"cwd:       {motionbert_dir}")
    print()

    proc = subprocess.run(cmd, cwd=str(motionbert_dir), env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    npy = out_path / "X3D.npy"
    if args.skip_render:
        print(f"Done. Wrote {npy.relative_to(repo_root)} (skipped MP4).")
    else:
        mp4 = out_path / "X3D.mp4"
        print(f"Done. Wrote {mp4.relative_to(repo_root)} and {npy.relative_to(repo_root)}.")


if __name__ == "__main__":
    main()
