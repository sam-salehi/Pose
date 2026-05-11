#!/usr/bin/env python3
"""
Review **gaps between punch annotations** (spreadsheet rows whose label is a known punch type).
Each gap is split into **chunks** of ``--window`` frames (default **20**). For **each chunk**, play the RGB clip and label::

    [p] Punch motion in this chunk (you will add punch-type labels later).
    [n] No punch in this chunk (negative example for training).
    [s] Skip this chunk (no write — review again on next run).
    [q] Quit and save everything recorded so far.

If a gap is no longer than ``window``, it is a **single** chunk. Longer gaps are tiled without
overlap; the last chunk may be shorter than ``window``.

Gaps with length **≤10 frames** are skipped by default (``--min-gap-frames`` default is **11**,
i.e. only gaps strictly longer than 10 frames). Override with e.g. ``--min-gap-frames 10`` for
“at least 10 frames”.

Output: ``Dataset/gap_labels/gap_review.json`` — each row is one **chunk** ``[start,end)`` with
optional ``gap_span``, ``chunk_index``, ``n_chunks``. Indices are **0-based half-open**.

If you previously saved **whole-gap** spans (one row per gap), delete those entries or use a new
``--out`` file — chunk keys ``(start,end)`` differ and old rows will not count as labeled chunks.

Already-labeled **chunks** are skipped on load unless ``--redo``.

Examples::

    python label_punch_gaps.py --ver V7
    python label_punch_gaps.py --ver V7 --window 20 --redo
    python label_punch_gaps.py --all --list-only --list-chunks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from skeleton_constants import PUNCH_CLASSES
from preprocess import _find_video, _load_annotations, _normalize_label

_REPO = Path(__file__).resolve().parent
_ANNOTATION_DIR = _REPO / "Dataset" / "Annotation_files"
_GAP_LABEL_DIR = _REPO / "Dataset" / "gap_labels"

_LABEL_TO_IDX: dict[str, int] = {
    "Cross": PUNCH_CLASSES.index("cross"),
    "Jab": PUNCH_CLASSES.index("jab"),
    "Lead Hook": PUNCH_CLASSES.index("lead_hook"),
    "Lead Uppercut": PUNCH_CLASSES.index("lead_uppercut"),
    "Rear Hook": PUNCH_CLASSES.index("rear_hook"),
    "Rear Uppercut": PUNCH_CLASSES.index("rear_uppercut"),
}


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out: list[tuple[int, int]] = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def _busy_punch_intervals(
    annotations: list[tuple[int, int, str]],
    n_total: int,
) -> list[tuple[int, int]]:
    """Merge intervals from rows whose normalized label is a training punch type."""
    raw: list[tuple[int, int]] = []
    for s, e, raw_label in annotations:
        label = _normalize_label(str(raw_label))
        if label not in _LABEL_TO_IDX:
            continue
        s0, e0 = s - 1, min(e, n_total)
        if e0 > s0:
            raw.append((s0, e0))
    return _merge_intervals(raw)


def _gaps_from_busy(busy: list[tuple[int, int]], n_total: int) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    cur = 0
    for a, b in busy:
        a = max(0, min(a, n_total))
        b = max(0, min(b, n_total))
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if cur < n_total:
        gaps.append((cur, n_total))
    return gaps


def _gap_key(ver: str, g0: int, g1: int) -> str:
    return f"{ver}:{g0}:{g1}"


def _split_gap_into_chunks(g0: int, g1: int, window: int) -> list[tuple[int, int]]:
    """
    Non-overlapping slices of length ``window``. If len(gap) <= window, one chunk ``[g0,g1)``.
    Otherwise tiles ``[g0,g0+w), [g0+w,g0+2w), ...`` with the last slice possibly shorter.
    """
    w = max(1, int(window))
    if g1 <= g0:
        return []
    if g1 - g0 <= w:
        return [(g0, g1)]
    out: list[tuple[int, int]] = []
    cur = g0
    while cur < g1:
        nxt = min(cur + w, g1)
        out.append((cur, nxt))
        cur += w
    return out


def _gap_all_chunks_labeled(
    ver: str,
    g0: int,
    g1: int,
    window: int,
    labeled: set[str],
    redo: bool,
) -> bool:
    """True if every chunk of this gap already has an entry in ``labeled``."""
    if redo:
        return False
    for c0, c1 in _split_gap_into_chunks(g0, g1, window):
        if _gap_key(ver, c0, c1) not in labeled:
            return False
    return True


def _load_review_json(path: Path) -> dict:
    if not path.is_file():
        return {"version": 1, "entries": [], "by_workbook": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rebuild_by_workbook(entries: list[dict]) -> dict[str, dict[str, list[list[int]]]]:
    out: dict[str, dict[str, list[list[int]]]] = {}
    for e in entries:
        ver = str(e["workbook"]).upper()
        lab = e["label"]
        span = [int(e["start"]), int(e["end"])]
        out.setdefault(ver, {"no_punch": [], "punch_in_gap": []})
        if lab == "no_punch":
            out[ver]["no_punch"].append(span)
        elif lab == "punch_in_gap":
            out[ver]["punch_in_gap"].append(span)
    return out


def _save_review(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "frame_indexing": "0-based half-open [start, end); matches frames[start:end] and Excel via s0=s-1",
        "entries": sorted(
            entries,
            key=lambda e: (str(e["workbook"]).upper(), int(e["start"]), int(e["end"])),
        ),
        "by_workbook": _rebuild_by_workbook(entries),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _overlay(
    frame: np.ndarray,
    *,
    ver: str,
    gap_idx: int,
    n_gaps: int,
    gap_g0: int,
    gap_g1: int,
    chunk_idx: int,
    n_chunks: int,
    chunk_c0: int,
    chunk_c1: int,
    fi: int,
    frame_in_clip: int,
    n_clip: int,
) -> None:
    h, w = frame.shape[:2]
    bar_h = 96
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
    y = 22
    cv2.putText(
        frame,
        f"{ver}  gap {gap_idx + 1}/{n_gaps}   chunk {chunk_idx + 1}/{n_chunks}   "
        f"chunk [{chunk_c0},{chunk_c1})   gap [{gap_g0},{gap_g1})",
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y += 22
    cv2.putText(
        frame,
        f"play {frame_in_clip + 1}/{n_clip}  fi={fi}   [p] punch  [n] no punch  [s] skip  [q] quit",
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )
    y += 22
    cv2.putText(
        frame,
        "Window-sized slice (--window frames)",
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 180, 160),
        1,
        cv2.LINE_AA,
    )
    y += 22
    cv2.putText(
        frame,
        "Label this chunk only (not the whole gap)",
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (140, 160, 140),
        1,
        cv2.LINE_AA,
    )


def _opencv_gui_ok() -> bool:
    try:
        cv2.namedWindow("__gap_lbl", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__gap_lbl")
        return True
    except cv2.error:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Label inter-punch gaps as punch vs no_punch.")
    ap.add_argument("--ver", type=str, default=None, help="Workbook tag e.g. V7 (use with single workbook)")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Process V1–V10 where annotation + video exist",
    )
    ap.add_argument(
        "--min-gap-frames",
        type=int,
        default=11,
        help="Minimum gap length in frames (default 11 = only gaps >10 frames long)",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=20,
        help="Chunk length in frames for labeling (default 20)",
    )
    ap.add_argument(
        "--max-preview-frames",
        type=int,
        default=450,
        help="Max frames to play per chunk (centre-crop long chunks; rarely needed if window is small)",
    )
    ap.add_argument("--playback-speed", type=float, default=1.0, help="Multiply source FPS")
    ap.add_argument("--out", type=Path, default=None, help=f"JSON output (default: {_GAP_LABEL_DIR}/gap_review.json)")
    ap.add_argument(
        "--redo",
        action="store_true",
        help="Re-label chunks (ignore prior saved labels for matching chunk spans)",
    )
    ap.add_argument("--list-only", action="store_true", help="Print gaps / chunk counts and exit")
    ap.add_argument(
        "--list-chunks",
        action="store_true",
        help="With --list-only, print every chunk span (can be very long)",
    )
    args = ap.parse_args()

    out_path = (args.out.resolve() if args.out else _GAP_LABEL_DIR / "gap_review.json")

    if args.all and args.ver:
        raise SystemExit("Use either --ver V7 or --all, not both.")
    if not args.all and args.ver is None:
        raise SystemExit("Provide --ver V7 or --all")

    versions: list[str]
    if args.all:
        versions = [f"V{i}" for i in range(1, 11)]
    else:
        versions = [args.ver.strip().upper()]

    data = _load_review_json(out_path)
    entries: list[dict] = list(data.get("entries", []))
    labeled: set[str] = set()
    if not args.redo:
        labeled = {
            _gap_key(str(e["workbook"]).upper(), int(e["start"]), int(e["end"]))
            for e in entries
        }

    workplans: list[tuple[str, Path, int, list[tuple[int, int]]]] = []

    for ver in versions:
        ann_path = _ANNOTATION_DIR / f"{ver}.xlsx"
        if not ann_path.is_file():
            print(f"[skip] no {ann_path.name}")
            continue
        vid = _find_video(ver)
        if vid is None:
            print(f"[skip] no RGB video for {ver}")
            continue
        npy_path = _REPO / "Dataset" / "MotionBERT_3d" / ver / "X3D.npy"
        if npy_path.is_file():
            n_total = int(np.load(npy_path, mmap_mode="r").shape[0])
        else:
            cap = cv2.VideoCapture(str(vid))
            if not cap.isOpened():
                print(f"[skip] cannot open video {vid}")
                continue
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        annotations = _load_annotations(ann_path)
        busy = _busy_punch_intervals(annotations, n_total)
        gaps = _gaps_from_busy(busy, n_total)
        gaps = [(a, b) for a, b in gaps if b - a >= args.min_gap_frames]
        workplans.append((ver, vid, n_total, gaps))
        print(f"{ver}: {len(busy)} punch spans merged → {len(gaps)} gaps (min {args.min_gap_frames} fr)")

    if args.list_only:
        for ver, vid, n_total, gaps in workplans:
            for g0, g1 in gaps:
                chunks = _split_gap_into_chunks(g0, g1, args.window)
                print(
                    f"  {ver}  gap [{g0},{g1}) len={g1-g0}  →  {len(chunks)} chunk(s) @ window={args.window}"
                )
                if args.list_chunks:
                    for i, (c0, c1) in enumerate(chunks):
                        print(f"      chunk {i + 1}/{len(chunks)}  [{c0},{c1}) len={c1-c0}")
        return

    if not _opencv_gui_ok():
        raise SystemExit(
            "OpenCV cannot create a window (headless?). Use --list-only or run with a display."
        )

    win = "gap_label_preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    new_entries: list[dict] = []
    quit_early = False

    for ver, vid, n_total, gaps in workplans:
        if quit_early:
            break
        to_do = [
            (g0, g1)
            for g0, g1 in gaps
            if not _gap_all_chunks_labeled(ver, g0, g1, args.window, labeled, args.redo)
        ]
        if not to_do:
            print(f"{ver}: no gaps with unfinished chunks (all done or empty).")
            continue

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"[skip] open failed {vid}", file=sys.stderr)
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        delay_ms = max(1, int(1000 / (fps * max(0.1, args.playback_speed))))

        n_gaps = len(to_do)
        for gi, (g0, g1) in enumerate(to_do):
            if quit_early:
                break

            chunks = _split_gap_into_chunks(g0, g1, args.window)
            n_ch = len(chunks)

            for chi, (c0, c1) in enumerate(chunks):
                if quit_early:
                    break
                if not args.redo and _gap_key(ver, c0, c1) in labeled:
                    continue

                length = c1 - c0
                if length > args.max_preview_frames:
                    pad = (length - args.max_preview_frames) // 2
                    play_lo = c0 + pad
                    play_hi = play_lo + args.max_preview_frames
                else:
                    play_lo, play_hi = c0, c1

                clip_frames = list(range(play_lo, play_hi))
                n_clip = len(clip_frames)
                decision = None

                for ci, fi in enumerate(clip_frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
                    ok, frame = cap.read()
                    if not ok:
                        print(f"warning: read fail frame {fi}", file=sys.stderr)
                        break
                    _overlay(
                        frame,
                        ver=ver,
                        gap_idx=gi,
                        n_gaps=n_gaps,
                        gap_g0=g0,
                        gap_g1=g1,
                        chunk_idx=chi,
                        n_chunks=n_ch,
                        chunk_c0=c0,
                        chunk_c1=c1,
                        fi=fi,
                        frame_in_clip=ci,
                        n_clip=n_clip,
                    )
                    cv2.imshow(win, frame)
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key == ord("q"):
                        decision = "__quit__"
                        break
                    if key == ord("n"):
                        decision = "no_punch"
                        break
                    if key == ord("p"):
                        decision = "punch_in_gap"
                        break
                    if key == ord("s"):
                        decision = "__skip__"
                        break

                while decision is None and not quit_early:
                    blank = np.zeros((130, 980, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        f"{ver} gap [{g0},{g1}) chunk {chi + 1}/{n_ch} [{c0},{c1}) — [p] punch [n] no [s] skip [q] quit",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(win, blank)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        decision = "__quit__"
                        break
                    if key == ord("n"):
                        decision = "no_punch"
                        break
                    if key == ord("p"):
                        decision = "punch_in_gap"
                        break
                    if key == ord("s"):
                        decision = "__skip__"
                        break

                if decision == "__quit__":
                    quit_early = True
                    break
                if decision == "__skip__" or decision is None:
                    continue
                assert decision in ("no_punch", "punch_in_gap")

                ent = {
                    "workbook": ver,
                    "start": c0,
                    "end": c1,
                    "label": decision,
                    "gap_span": [g0, g1],
                    "chunk_index": chi,
                    "n_chunks": n_ch,
                    "window": int(args.window),
                }
                new_entries.append(ent)
                ck = _gap_key(ver, c0, c1)
                labeled.add(ck)
                entries = [
                    e
                    for e in entries
                    if _gap_key(str(e["workbook"]).upper(), int(e["start"]), int(e["end"])) != ck
                ]
                entries.append(ent)
                _save_review(out_path, entries)
                print(f"  saved {ver} chunk [{c0},{c1}) gap [{g0},{g1}) {chi + 1}/{n_ch} → {decision}")

        cap.release()

    cv2.destroyWindow(win)

    if new_entries:
        try:
            rel = out_path.relative_to(_REPO)
        except ValueError:
            rel = out_path
        print(f"\nRecorded {len(new_entries)} chunk labels this session → {rel}")
    elif quit_early:
        try:
            rel = out_path.relative_to(_REPO)
        except ValueError:
            rel = out_path
        print(f"\nQuit early — saved state → {rel}")
    else:
        print("\nNo new labels written.")


if __name__ == "__main__":
    main()
