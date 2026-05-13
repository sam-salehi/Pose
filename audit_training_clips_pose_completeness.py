#!/usr/bin/env python3
"""
For classifier training clips (same enumeration as ``train_3d_classifier_bio.py``),
report what fraction have **all 3D joint positions** valid in the MotionBERT export.

``X3D.npy`` is ``(T, 17, 3)``: per frame, 17 H36M joints, each with **three real-valued
coordinates** (the 3D point for that joint). A clip passes if **every** of those
``T × 17 × 3`` numbers is finite (no NaN, no Inf) in the raw slice used to build the clip.

- Punch clips: xlsx rows → ``frames[s0:e0]`` (half-open end from training).
- no_punch clips: ``gap_review.json`` spans → two raw windows per span
  (``_two_raw_spans_for_no_punch``), same as training.

Default workbook set: **V3–V10** (override with ``--vers``).

Usage::

    python audit_training_clips_pose_completeness.py
    python audit_training_clips_pose_completeness.py --vers V3 V4 V5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from preprocess import _load_annotations, _normalize_label
from skeleton_constants import PUNCH_CLASSES

_REPO = Path(__file__).resolve().parent
_MOTIONBERT_DIR = _REPO / "Dataset" / "MotionBERT_3d"
_ANNOTATION_DIR = _REPO / "Dataset" / "Annotation_files"
_GAP_REVIEW_JSON = _REPO / "Dataset" / "gap_labels" / "gap_review.json"

CLF_WINDOW = 16
NO_PUNCH_SAMPLES_PER_GAP_SPAN = 2

_LABEL_TO_IDX: dict[str, int] = {
    "Cross":         PUNCH_CLASSES.index("cross"),
    "Jab":           PUNCH_CLASSES.index("jab"),
    "Lead Hook":     PUNCH_CLASSES.index("lead_hook"),
    "Lead Uppercut": PUNCH_CLASSES.index("lead_uppercut"),
    "Rear Hook":     PUNCH_CLASSES.index("rear_hook"),
    "Rear Uppercut": PUNCH_CLASSES.index("rear_uppercut"),
}


def _two_raw_spans_for_no_punch(g0: int, g1: int, window: int) -> list[tuple[int, int]]:
    L = g1 - g0
    if L <= 0:
        return []
    if L < window:
        sp = (g0, g1)
        return [sp, sp]
    last_start = g1 - window
    first_start = g0
    if last_start <= first_start:
        sp = (g0, g0 + window)
        return [sp, sp]
    return [(first_start, first_start + window), (last_start, g1)]


def _no_punch_spans_from_gap_review(versions: frozenset[str]) -> list[tuple[str, int, int]]:
    if not _GAP_REVIEW_JSON.is_file():
        raise SystemExit(f"Missing {_GAP_REVIEW_JSON}")
    data = json.loads(_GAP_REVIEW_JSON.read_text(encoding="utf-8"))
    rows: list[tuple[str, int, int]] = []
    for e in data.get("entries", []):
        if e.get("label") != "no_punch":
            continue
        ver = str(e["workbook"]).strip().upper()
        if ver not in versions:
            continue
        s0, e0 = int(e["start"]), int(e["end"])
        if e0 > s0:
            rows.append((ver, s0, e0))
    return rows


def _slice_all_finite(frames: np.ndarray, a: int, b: int) -> bool:
    """True iff every 3D coordinate in ``frames[a:b]`` is finite — shape (T,17,3)."""
    sl = frames[a:b]
    if sl.size == 0:
        return False
    return bool(np.isfinite(sl).all())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fraction of training clips whose MotionBERT X3D has all joint 3D points "
            "(T×17×3 coordinates) finite."
        )
    )
    ap.add_argument(
        "--vers",
        nargs="+",
        default=[f"V{i}" for i in range(3, 11)],
        help="Workbook tags (default: V3 … V10)",
    )
    args = ap.parse_args()
    versions = frozenset(v.strip().upper() for v in args.vers)
    if not versions:
        raise SystemExit("Provide at least one --vers tag.")

    total = complete = 0
    punch_tot = punch_ok = 0
    np_tot = np_ok = 0
    per_ver: dict[str, list[int]] = {v: [0, 0] for v in sorted(versions)}  # [total, ok]

    # ── Punch clips (per version, same as train_3d_classifier_bio) ───────────
    for ver in sorted(versions):
        npy_path = _MOTIONBERT_DIR / ver / "X3D.npy"
        ann_path = _ANNOTATION_DIR / f"{ver}.xlsx"
        if not npy_path.is_file():
            print(f"[skip {ver}] no {npy_path.relative_to(_REPO)}", file=sys.stderr)
            continue
        if not ann_path.is_file():
            print(f"[skip {ver}] no {ann_path.relative_to(_REPO)}", file=sys.stderr)
            continue

        frames = np.load(npy_path)
        if frames.ndim != 3 or frames.shape[1] != 17 or frames.shape[2] != 3:
            print(f"[skip {ver}] bad X3D shape {frames.shape}", file=sys.stderr)
            continue
        n_total = frames.shape[0]

        for s, e, raw_label in _load_annotations(ann_path):
            label = _normalize_label(str(raw_label))
            if label not in _LABEL_TO_IDX:
                continue
            s0, e0 = s - 1, min(e, n_total)
            if e0 <= s0:
                continue
            ok = _slice_all_finite(frames, s0, e0)
            total += 1
            punch_tot += 1
            if ok:
                complete += 1
                punch_ok += 1
            per_ver[ver][0] += 1
            per_ver[ver][1] += int(ok)

    # ── no_punch clips from gap_review ───────────────────────────────────────
    spans = _no_punch_spans_from_gap_review(versions)
    cache: dict[str, np.ndarray] = {}
    for ver, s0, e0 in spans:
        npy_path = _MOTIONBERT_DIR / ver / "X3D.npy"
        if not npy_path.is_file():
            continue
        if ver not in cache:
            cache[ver] = np.load(npy_path)
        frames = cache[ver]
        n_total = frames.shape[0]
        e0 = min(e0, n_total)
        s0 = max(0, s0)
        if e0 <= s0:
            continue
        raws = _two_raw_spans_for_no_punch(s0, e0, CLF_WINDOW)
        if len(raws) != NO_PUNCH_SAMPLES_PER_GAP_SPAN:
            continue
        for a, b in raws:
            ok = _slice_all_finite(frames, a, b)
            total += 1
            np_tot += 1
            if ok:
                complete += 1
                np_ok += 1
            if ver in per_ver:
                per_ver[ver][0] += 1
                per_ver[ver][1] += int(ok)

    print(f"Workbooks: {', '.join(sorted(versions))}")
    print(f"gap_review no_punch spans (filtered): {len(spans)}")
    print()
    if total == 0:
        raise SystemExit("No clips enumerated — check paths, xlsx, gap_review, and --vers.")

    r = complete / total
    print(f"Criterion: all 3D joint coordinates finite in raw X3D.npy (shape T×17×3 per clip).")
    print()
    print(f"All clips:           {complete:6d} / {total:6d}  pass  ({100.0 * r:.2f} %)")
    if punch_tot:
        print(
            f"  Punch (xlsx):      {punch_ok:6d} / {punch_tot:6d}  ({100.0 * punch_ok / punch_tot:.2f} %)"
        )
    if np_tot:
        print(
            f"  no_punch (JSON):   {np_ok:6d} / {np_tot:6d}  ({100.0 * np_ok / np_tot:.2f} %)"
        )
    print()
    print("Per-version (punch + no_punch clips that include this workbook):")
    for ver in sorted(versions):
        t, o = per_ver[ver]
        if t == 0:
            print(f"  {ver}:  (no clips)")
        else:
            print(f"  {ver}:  {o:5d} / {t:5d}  ({100.0 * o / t:.2f} %)")


if __name__ == "__main__":
    main()
