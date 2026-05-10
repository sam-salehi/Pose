#!/usr/bin/env python3
"""
Play a workbook RGB clip next to its MotionBERT ``X3D.mp4`` output.

Resolves (same layout as ``run_motionbert_3d_batch.py``)::

    Dataset/RGB_videos/source_youtube/{VER}_*.mp4   (left)
    Dataset/MotionBERT_3d/{VER}/X3D.mp4            (right)

Example::

    python compare_workbook_x3d.py V7
    python compare_workbook_x3d.py V7 --height 720 --fps 30

Controls (in the viewer): q / ESC quit, SPACE pause.

See also ``MotionBERT/compare_videos_side_by_side.py`` for generic paths.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Side-by-side playback: workbook RGB vs MotionBERT X3D."
    )
    ap.add_argument(
        "ver",
        help="Workbook tag, e.g. V7",
    )
    ap.add_argument("--height", type=int, default=480, help="Pane height (each side)")
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS (default: from source video)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Also write side-by-side MP4 to this path",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Only write --out; do not open a window",
    )
    args = ap.parse_args()

    ver = args.ver.strip().upper()
    if not ver.startswith("V") or not ver[1:].isdigit():
        raise SystemExit(f"Expected tag like V7, got {args.ver!r}")

    rgb_dir = repo / "Dataset" / "RGB_videos" / "source_youtube"
    matches = sorted(glob.glob(str(rgb_dir / f"{ver}_*.mp4")))
    if not matches:
        raise SystemExit(f"No MP4 found for {ver} under {rgb_dir}/")

    left = Path(matches[0])
    right = repo / "Dataset" / "MotionBERT_3d" / ver / "X3D.mp4"
    if not right.is_file():
        raise SystemExit(f"Missing MotionBERT output (run infer_wild first): {right}")

    player = repo / "MotionBERT" / "compare_videos_side_by_side.py"
    if not player.is_file():
        raise SystemExit(f"Missing {player}")

    cmd: list[str | Path] = [
        sys.executable,
        str(player),
        "--left",
        str(left),
        "--right",
        str(right),
        "--height",
        str(args.height),
    ]
    if args.fps is not None:
        cmd.extend(["--fps", str(args.fps)])
    if args.out is not None:
        cmd.extend(["--out", str(args.out)])
    if args.no_show:
        cmd.append("--no-show")

    print(f"Left:  {left.relative_to(repo)}")
    print(f"Right: {right.relative_to(repo)}")
    raise SystemExit(subprocess.call(cmd, cwd=str(repo / "MotionBERT")))


if __name__ == "__main__":
    main()
