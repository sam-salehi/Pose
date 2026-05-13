#!/usr/bin/env python3
"""
Review annotation quality for one workbook (``Vx``).

Loads ``Dataset/Annotation_files/{VER}.xlsx``, finds the matching RGB MP4, then
plays each annotated segment with the label overlaid. Default playback is slower
than realtime so you can judge timing and labels.

**Frames vs Excel (BoxingVI / BoxIV-style convention, same as ``preprocess`` / training):**
spreadsheet ``start`` and ``end`` are **1-based inclusive** frame indices into
the **same** MP4 you open here (frame ``1`` = first decoded frame). Internally
this script uses 0-based **inclusive** ``i0..i1`` matching training’s half-open
slice ``frames[s0:e0)`` with ``s0=start-1``, ``e0=end`` (``end`` = last 1-based
frame index, same integer as in the sheet).

**If labels look shifted in time:** the spreadsheet is usually aligned to the
player or toolchain used during BoxingVI labelling, not necessarily to OpenCV’s
decoder. Try ``--frame-offset N`` (shifts both ends by ``N`` OpenCV frames).
Compare ``CAP_PROP_FRAME_COUNT`` vs ``Dataset/MotionBERT_3d/{ver}/X3D.npy`` length
printed at startup — if they differ, 3D training rows may not line up with this
RGB decode either.

**Alignment:** ``cv2.VideoCapture.set(CAP_PROP_POS_FRAMES)`` is unreliable on
many H.264/MP4 files (keyframe snapping). By default this script **decodes
sequentially** so on-screen frames match Excel; use ``--fast-seek`` for the old
seek-based path (faster, may drift).

GUI: **q** / **Esc** quit entire run · **n** skip to next clip

Example::

    python eval.py V7
    python eval.py V7 --speed 0.2
    python eval.py V3 --limit 5
    python eval.py V1 --export-dir figures/eval_V1   # headless: write MP4s
    python eval.py V7 --fast-seek                   # fast OpenCV seek (may drift)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent

from preprocess import (
    _ANNOTATIONS,
    _find_video,
    _load_annotations,
    _opencv_has_gui,
    _safe_name,
)


def _normalize_ver(raw: str) -> str:
    s = raw.strip().upper()
    if re.fullmatch(r"V?\d+", s):
        if not s.startswith("V"):
            s = "V" + s
        n = int(s[1:])
        if n >= 1:
            return f"V{n}"
    raise SystemExit(f"Expected workbook tag like V7 or 7, got {raw!r}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Play annotated clips for one workbook with labels (slow-mo by default)."
    )
    ap.add_argument(
        "ver",
        type=str,
        help="Workbook tag, e.g. V7 or 7",
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=0.35,
        metavar="FACTOR",
        help="Playback speed vs realtime (default: 0.35 = slow motion). Use 1.0 for realtime.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only play the first N annotated clips (after optional shuffle).",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize clip order (uses --seed).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed when --shuffle is set (default: 42).",
    )
    ap.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="If OpenCV has no GUI, or you force headless: write one MP4 per clip here.",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a window; write clips to --export-dir (required unless GUI works).",
    )
    ap.add_argument(
        "--fast-seek",
        action="store_true",
        help=(
            "Use OpenCV frame-index seek (fast; often misaligned on compressed MP4). "
            "Default: sequential decode so 1-based Excel frames match the video."
        ),
    )
    ap.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Add N to both 0-based decode indices after Excel conversion (negative allowed). "
            "Use when labels look consistently early/late vs the RGB (different decoder / export)."
        ),
    )
    args = ap.parse_args()

    ver = _normalize_ver(args.ver)
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")

    xlsx = _ANNOTATIONS / f"{ver}.xlsx"
    if not xlsx.is_file():
        raise SystemExit(f"Missing annotation file: {xlsx}")

    rows = _load_annotations(xlsx)
    if not rows:
        raise SystemExit(f"No rows in {xlsx}")

    if args.shuffle:
        import random

        random.seed(args.seed)
        random.shuffle(rows)

    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    if args.shuffle and not args.fast_seek:
        print(
            "note: clips are sorted by start frame for sequential decode "
            "(shuffle only affects which subset --limit keeps).",
            file=sys.stderr,
        )
    # Time order so we can decode forward without rewinding; tie-break stabilizes overlaps.
    rows = sorted(rows, key=lambda r: (int(r[0]), int(r[1])))

    video_path = _find_video(ver)
    if video_path is None:
        raise SystemExit(
            f"No video under Dataset/RGB_videos/source_youtube/{ver}_*.mp4"
        )

    use_gui = _opencv_has_gui() and not args.headless
    out_root = args.export_dir
    if not use_gui:
        out_root = out_root or _REPO / ".cache" / "eval_previews" / ver
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"No GUI — writing clips to {out_root.resolve()}\n", file=sys.stderr)

    total = len(rows)
    decode_mode = "fast seek (may drift)" if args.fast_seek else "sequential (frame-accurate)"
    print(
        f"{ver}: {total} clip(s) from {xlsx.name}  |  video: {video_path.name}\n"
        f"decode={decode_mode}  |  speed={args.speed}x  |  [n] next  [q/Esc] quit\n",
        file=sys.stderr,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    x3d_path = _REPO / "Dataset" / "MotionBERT_3d" / ver / "X3D.npy"
    n_x3d: int | None = None
    if x3d_path.is_file():
        n_x3d = int(np.load(x3d_path, mmap_mode="r").shape[0])
        print(
            f"frame counts: OpenCV CAP_PROP_FRAME_COUNT={n_vid}  "
            f"MotionBERT X3D.npy T={n_x3d}  ({x3d_path.relative_to(_REPO)})",
            file=sys.stderr,
        )
        if n_vid > 0 and n_x3d > 0 and n_vid != n_x3d:
            print(
                "warning: RGB frame count ≠ X3D length — annotations may match one stream "
                "better than the other; try --frame-offset if RGB looks shifted vs labels.",
                file=sys.stderr,
            )
    else:
        print(f"frame counts: OpenCV CAP_PROP_FRAME_COUNT={n_vid}  (no X3D.npy)", file=sys.stderr)
    if args.frame_offset != 0:
        print(f"--frame-offset={args.frame_offset} (applied after 1-based→0-based)", file=sys.stderr)

    delay_ms = max(1, int(1000.0 / fps / args.speed))
    decoder_next = 0

    try:
        for idx, (s1, e1, label) in enumerate(rows, 1):
            i0, i1 = int(s1) - 1, int(e1) - 1
            i0 += args.frame_offset
            i1 += args.frame_offset
            if n_vid > 0:
                i0 = max(0, min(n_vid - 1, i0))
                i1 = max(0, min(n_vid - 1, i1))
            if i1 < i0:
                print(f"  skip clip {idx}: empty range after offset/clamp", file=sys.stderr)
                continue
            t0, t1 = i0 / fps, i1 / fps
            tag = (
                f"[{idx}/{total}] {ver} | {label} | Excel frames {s1}–{e1} (1-based) "
                f"| decode {i0}–{i1} (0-based, after offset) | {t0:.2f}s–{t1:.2f}s"
            )
            print(tag, file=sys.stderr)

            writer: cv2.VideoWriter | None = None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = (
                out_root / f"{idx:03d}_{ver}_{_safe_name(label)}_{s1}-{e1}.mp4"
                if not use_gui
                else None
            )

            skip_clip = False
            if args.fast_seek:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(i0))
            else:
                if i0 < decoder_next:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                    decoder_next = 0
                while decoder_next < i0:
                    ok_skip, _ = cap.read()
                    if not ok_skip:
                        print(
                            f"  warning: EOF before frame {i0} "
                            f"(at decoder {decoder_next})",
                            file=sys.stderr,
                        )
                        skip_clip = True
                        break
                    decoder_next += 1
            if skip_clip:
                continue

            for fi in range(i0, i1 + 1):
                ok, frame = cap.read()
                if not ok:
                    print(f"  warning: could not read frame index {fi}", file=sys.stderr)
                    break
                if not args.fast_seek:
                    decoder_next += 1

                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 72), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    tag[:130],
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                line2 = (
                    f"Excel frame {fi + 1} (1-based)  |  decode idx {fi}  |  "
                    f"in-clip {fi - i0 + 1}/{e1 - s1 + 1}"
                )
                cv2.putText(
                    frame,
                    line2[:130],
                    (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (200, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

                if use_gui:
                    scale = min(1280 / w, 720 / h, 1.0)
                    disp = (
                        cv2.resize(frame, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
                        if scale < 1
                        else frame
                    )
                    cv2.imshow("Annotation eval (slow-mo)", disp)
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key in (ord("q"), 27):
                        return
                    if key == ord("n"):
                        skip_clip = True
                        break
                else:
                    if writer is None:
                        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                        if not writer.isOpened():
                            raise SystemExit(f"VideoWriter failed for {out_path}")
                    writer.write(frame)

            if writer is not None:
                writer.release()
                print(f"  saved {out_path}", file=sys.stderr)

            if skip_clip:
                continue

    finally:
        cap.release()
        if use_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
