#!/usr/bin/env python3
"""
Evaluate BoxingPunchClassifier (rule-based, punch_classifier.py) on MotionBERT 3D NPY files.

For each video that has both an X3D.npy and an annotation xlsx:
  1. Loads every annotated punch window from the 3D pose array.
  2. Runs BoxingPunchClassifier.predict() on each window.
  3. Prints per-class accuracy + confusion matrix.
  4. Writes an annotated MP4 to figures/ showing the RGB video with
     ground-truth label (top bar) and predicted label (bottom bar).

Usage
-----
    python eval_punch_classifier.py              # all videos that have 3D npy + annotation
    python eval_punch_classifier.py --ver V2     # single video
    python eval_punch_classifier.py --ver V2 --out figures/V2_eval.mp4
    python eval_punch_classifier.py --no-video   # evaluation only, no MP4 written
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from punch_classifier import BoxingPunchClassifier, PUNCH_LABELS

_REPO = Path(__file__).resolve().parent
_MOTIONBERT_DIR = _REPO / "Dataset" / "MotionBERT_3d"
_ANNOT_DIR      = _REPO / "Dataset" / "Annotation_files"
_VIDEOS_DIR     = _REPO / "Dataset" / "RGB_videos" / "source_youtube"
_FIGURES        = _REPO / "figures"
_STAGE_B_PATH   = _REPO / "stage_b_clf.pkl"


def _load_clf() -> BoxingPunchClassifier:
    """Load classifier, attaching the trained Stage-B model if available."""
    stage_b = None
    if _STAGE_B_PATH.is_file():
        import joblib
        stage_b = joblib.load(_STAGE_B_PATH)
        print(f"Loaded Stage-B model: {_STAGE_B_PATH.name}")
    return BoxingPunchClassifier(stage_b_clf=stage_b)

_LABEL_NORM: dict[str, str] = {
    "jab":           "jab",
    "jab ":          "jab",
    "cross":         "cross",
    "lead hook":     "lead_hook",
    "lead hook ":    "lead_hook",
    "rear hook":     "rear_hook",
    "lead uppercut": "lead_uppercut",
    "rear uppercut": "rear_uppercut",
}

_NAME_TO_INT: dict[str, int] = {v: k for k, v in PUNCH_LABELS.items()}

# BGR colors per punch type (for text overlay)
_CLR: dict[str, tuple[int, int, int]] = {
    "jab":           (50,  220, 255),
    "cross":         (255, 200,  50),
    "lead_hook":     (50,  255, 100),
    "rear_hook":     (200,  50, 255),
    "lead_uppercut": (100, 180, 255),
    "rear_uppercut": (255,  80,  80),
}
_CLR_WHITE  = (255, 255, 255)
_CLR_RED    = (80,  80,  255)
_CLR_GREEN  = (80,  255,  80)


# ── Annotation loading ─────────────────────────────────────────────────────────

def _load_annotations(xlsx: Path) -> list[tuple[int, int, str]]:
    """
    Returns list of (start_frame, end_frame, canonical_label_string).
    Handles files with or without a header row and int/float frame numbers.
    """
    wb = openpyxl.load_workbook(xlsx)
    ws = wb.active
    punches: list[tuple[int, int, str]] = []
    for row in ws.iter_rows(values_only=True):
        s, e, lbl = row[0], row[1], row[2]
        if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
            continue
        if not isinstance(lbl, str):
            continue
        lbl_norm = _LABEL_NORM.get(lbl.strip().lower())
        if lbl_norm is None:
            continue
        punches.append((int(s), int(e), lbl_norm))
    return punches


# ── Available videos ───────────────────────────────────────────────────────────

def _find_rgb_video(ver: str) -> Path | None:
    for p in sorted(_VIDEOS_DIR.glob(f"{ver}_*.mp4")):
        return p
    return None


def _available_versions() -> list[str]:
    """Versions that have both X3D.npy and Annotation xlsx."""
    vers = []
    for d in sorted(_MOTIONBERT_DIR.iterdir()):
        if not d.is_dir():
            continue
        npy = d / "X3D.npy"
        xlsx = _ANNOT_DIR / f"{d.name}.xlsx"
        if npy.is_file() and xlsx.is_file():
            vers.append(d.name)
    return vers


# ── Evaluation core ────────────────────────────────────────────────────────────

def _evaluate(ver: str, clf: BoxingPunchClassifier) -> tuple[
    list[tuple[int, int, str, str]],   # (start, end, gt, pred)
    dict[str, dict[str, int]],         # confusion  gt → pred → count
]:
    npy_path  = _MOTIONBERT_DIR / ver / "X3D.npy"
    xlsx_path = _ANNOT_DIR / f"{ver}.xlsx"

    poses = np.load(str(npy_path))            # (T, 17, 3)
    annotations = _load_annotations(xlsx_path)

    results: list[tuple[int, int, str, str]] = []
    confusion: dict[str, dict[str, int]] = {}

    for start, end, gt in annotations:
        window = poses[start : end + 1]
        if window.shape[0] < 3:
            window = poses[max(0, start - 3) : end + 4]
        if window.shape[0] < 3:
            continue
        try:
            pred_int = clf.predict(window)
            pred = PUNCH_LABELS[pred_int]
        except Exception as exc:
            print(f"  {ver} frame {start}-{end}: classifier error — {exc}", file=sys.stderr)
            continue

        results.append((start, end, gt, pred))
        confusion.setdefault(gt, {})
        confusion[gt][pred] = confusion[gt].get(pred, 0) + 1

    return results, confusion


def _print_report(ver: str, results: list[tuple[int, int, str, str]],
                  confusion: dict[str, dict[str, int]]) -> None:
    if not results:
        print(f"\n{ver}: no results")
        return

    all_classes = sorted({gt for _, _, gt, _ in results} | {pred for _, _, _, pred in results})
    correct = sum(gt == pred for _, _, gt, pred in results)
    total   = len(results)

    print(f"\n{'='*60}")
    print(f"  {ver}  —  {correct}/{total} correct  ({100*correct/total:.1f}%)")
    print(f"{'='*60}")

    # Per-class accuracy (GT classes only — predicted-only classes have no GT to score)
    gt_classes = sorted({gt for _, _, gt, _ in results})
    print(f"  {'Class':<20} {'Correct':>8} {'Total':>8} {'Acc%':>8}")
    print(f"  {'-'*46}")
    for cls in gt_classes:
        cls_rows = [(g, p) for _, _, g, p in results if g == cls]
        c = sum(g == p for g, p in cls_rows)
        t = len(cls_rows)
        print(f"  {cls:<20} {c:>8} {t:>8} {100*c/t:>7.1f}%")

    # Confusion matrix
    print(f"\n  Confusion matrix (rows=GT, cols=predicted):")
    hdr = f"  {'':20}" + "".join(f"{c[:10]:>12}" for c in all_classes)
    print(hdr)
    for gt_cls in all_classes:
        row_vals = confusion.get(gt_cls, {})
        row = f"  {gt_cls:<20}" + "".join(
            f"{row_vals.get(pr, 0):>12}" for pr in all_classes
        )
        print(row)


# ── Video annotation ───────────────────────────────────────────────────────────

def _overlay_bar(frame: np.ndarray, text: str, color: tuple[int, int, int],
                 y_top: int, height: int = 36) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, y_top), (w, y_top + height), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, y_top + height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _write_video(ver: str, results: list[tuple[int, int, str, str]],
                 out_path: Path) -> None:
    vid_path = _find_rgb_video(ver)
    if vid_path is None:
        print(f"  {ver}: no RGB video found in {_VIDEOS_DIR}, skipping video output.")
        return

    # Build frame-level GT and pred labels
    # Use the last result covering a frame as the overlay (most recent punch)
    max_frame = max(e for _, e, _, _ in results) + 1 if results else 0

    gt_label:   dict[int, str] = {}
    pred_label: dict[int, str] = {}
    for start, end, gt, pred in results:
        for f in range(start, end + 1):
            gt_label[f]   = gt
            pred_label[f] = pred

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        print(f"  {ver}: cannot open video {vid_path}", file=sys.stderr)
        return

    fps  = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_v  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
    if not writer.isOpened():
        cap.release()
        print(f"  {ver}: VideoWriter failed for {out_path}", file=sys.stderr)
        return

    bar_h = 38
    try:
        for f in range(n_v):
            ok, frame = cap.read()
            if not ok:
                break

            gt   = gt_label.get(f)
            pred = pred_label.get(f)

            if gt is not None:
                match = gt == pred
                gt_clr   = _CLR.get(gt,   _CLR_WHITE)
                pred_clr = _CLR_GREEN if match else _CLR_RED

                gt_text   = f"GT:   {gt.replace('_',' ').title()}"
                pred_text = f"PRED: {pred.replace('_',' ').title() if pred else '?'}"

                _overlay_bar(frame, gt_text,   gt_clr,         0,     bar_h)
                _overlay_bar(frame, pred_text, pred_clr, bar_h, bar_h)
            else:
                # Show faint frame counter when not in a punch window
                cv2.putText(frame, f"{ver}  frame {f}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

            writer.write(frame)
    finally:
        writer.release()
        cap.release()

    print(f"  Video written → {out_path.relative_to(_REPO)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate rule-based BoxingPunchClassifier on MotionBERT 3D NPY data."
    )
    ap.add_argument("--ver",      type=str,  default=None,
                    help="E.g. V2 (default: all versions with 3D npy + annotation)")
    ap.add_argument("--out",      type=Path, default=None,
                    help="Output MP4 path (only valid when --ver is set)")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip video output (evaluation metrics only)")
    args = ap.parse_args()

    clf = _load_clf()

    versions = [args.ver.strip().upper()] if args.ver else _available_versions()
    if not versions:
        sys.exit("No versions found with both X3D.npy and annotation xlsx.")

    print(f"Running on: {versions}")

    all_results: list[tuple[int, int, str, str]] = []

    for ver in versions:
        npy_path = _MOTIONBERT_DIR / ver / "X3D.npy"
        xlsx_path = _ANNOT_DIR / f"{ver}.xlsx"
        if not npy_path.is_file():
            print(f"{ver}: missing X3D.npy — skipping", file=sys.stderr)
            continue
        if not xlsx_path.is_file():
            print(f"{ver}: missing annotation xlsx — skipping", file=sys.stderr)
            continue

        print(f"\nProcessing {ver} …", end=" ", flush=True)
        results, confusion = _evaluate(ver, clf)
        print(f"{len(results)} punches classified.")
        _print_report(ver, results, confusion)
        all_results.extend(results)

        if not args.no_video:
            out = args.out if (args.out and len(versions) == 1) else (
                _FIGURES / f"{ver}_punch_eval.mp4"
            )
            _write_video(ver, results, out)

    # Aggregate summary across all videos
    if len(versions) > 1 and all_results:
        print(f"\n{'='*60}")
        print(f"  AGGREGATE  —  all {len(versions)} videos")
        correct = sum(gt == pred for _, _, gt, pred in all_results)
        total   = len(all_results)
        print(f"  {correct}/{total} correct  ({100*correct/total:.1f}%)")
        all_classes = sorted({gt for _, _, gt, _ in all_results})
        print(f"  {'Class':<20} {'Acc%':>8}")
        for cls in all_classes:
            rows = [(g, p) for _, _, g, p in all_results if g == cls]
            c = sum(g == p for g, p in rows)
            t = len(rows)
            print(f"  {cls:<20} {100*c/t:>7.1f}%  ({c}/{t})")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
