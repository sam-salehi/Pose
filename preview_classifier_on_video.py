#!/usr/bin/env python3
"""
Run punch **detection** + **type classification** on an RGB workout video, draw the 12-joint
skeleton, and overlay predicted punch classes on frames that fall inside detector-positive
windows (same MediaPipe layout as ``train_classifier.py`` / ``preview_detection_on_video.py``).

Workflow
--------
1. Load full-video poses ``(F, 12, 2)`` from ``Dataset/pose_sequences/{ver}_pose.npz`` if present,
   otherwise run MediaPipe sequentially (slow on first use).
2. **Detector** (``GCNDetector``): sliding windows with length 11 / stride 1 (matches ``train.ipynb``).
   Preprocess each window like ``DetectorWindowNpzDataset``: ``nan_to_num`` + hip-centered coords.
3. On windows with detector score ≥ ``--det-threshold``, build a **classifier** clip of ``window``
   frames (default 20 from the classifier checkpoint) centered on that detector window, preprocess
   like training (``_sanitize_pose_clip`` + ``nan_to_num``, **no** hip centering).
4. Encode an MP4 with skeleton + banner text (predicted class + probability).

Usage
-----
    python preview_classifier_on_video.py --ver V7
    python preview_classifier_on_video.py --video Dataset/RGB_videos/source_youtube/V7_foo.mp4
    python preview_classifier_on_video.py --ver V8 --classifier checkpoints/gcn_classifier_V7_V8_V9.pt

Optional: ``--detection-npz PATH`` or ``--use-detection-labels`` uses punch/not labels from
``Dataset/detection_frame_labels/{ver}_detection.npz`` (Excel-derived) instead of the neural
detector — classifier still runs **only** on punch-positive windows.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from GCN import (
    BOXINGVI_BONE_PAIRS,
    BOXINGVI_CENTER_JOINT,
    BOXINGVI_GRAPH_EDGES,
    GCNClassifier,
    GCNDetector,
    NUM_BOXINGVI_JOINTS,
    PUNCH_CLASSES,
)
from detector_data import extract_full_video_poses, load_pose_sequence_npz
from preprocess import _find_video, _sanitize_pose_clip, center_pose_on_hip_midpoint

_REPO = Path(__file__).resolve().parent
_POSE_SEQ_DIR = _REPO / "Dataset" / "pose_sequences"
_DEFAULT_DET_LABEL_DIR = _REPO / "Dataset" / "detection_frame_labels"
_FIGURES = _REPO / "figures"
_CHECKPOINTS = _REPO / "checkpoints"

# Sliding windows for detector training (train.ipynb)
DET_WINDOW_LENGTH = 11
DET_STRIDE = 1

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
    cv2.putText(frame, line1[:200], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if line2:
        y += 22
        cv2.putText(frame, line2[:200], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    if line3:
        y += 22
        cv2.putText(frame, line3[:200], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1, cv2.LINE_AA)


def _infer_ver_tag(path: Path) -> str | None:
    m = re.search(r"(V\d+)", path.stem.upper())
    return m.group(1) if m else None


def _latest_ckpt(prefix: str) -> Path | None:
    """Newest ``checkpoints/{prefix}*.pt`` by mtime (used for classifier default)."""
    if not _CHECKPOINTS.is_dir():
        return None
    cands = sorted(_CHECKPOINTS.glob(f"{prefix}*.pt"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _detector_ckpt_version_tags(path: Path) -> frozenset[str]:
    """Tags after ``gcn_detector_`` (e.g. V7 from ``..._V6_V7_V8_...``)."""
    stem = path.stem
    if not stem.startswith("gcn_detector_"):
        return frozenset()
    rest = stem[len("gcn_detector_") :]
    parts = rest.split("_")
    return frozenset(p for p in parts if len(p) >= 2 and p[0] == "V" and p[1:].isdigit())


def _default_detector_ckpt(ver: str | None) -> tuple[Path | None, str | None]:
    """
    Prefer the newest checkpoint whose **filename** lists the workbook tag (e.g. V7).

    Using plain ``mtime`` can pick a detector trained *without* that workbook — scores then
    often stay below ``--det-threshold`` for the whole video.
    """
    if not _CHECKPOINTS.is_dir():
        return None, None
    cands = sorted(_CHECKPOINTS.glob("gcn_detector_*.pt"))
    if not cands:
        return None, None
    if ver and re.fullmatch(r"V\d+", ver):
        tagged = [p for p in cands if ver in _detector_ckpt_version_tags(p)]
        if tagged:
            return max(tagged, key=lambda p: p.stat().st_mtime), None
        picked = max(cands, key=lambda p: p.stat().st_mtime)
        warn = (
            f"No detector checkpoint filename lists {ver!r} — using newest {picked.name}. "
            "Scores may stay below --det-threshold; pass --detector PATH to a model trained "
            f"with {ver}, or use --use-detection-labels."
        )
        return picked, warn
    return max(cands, key=lambda p: p.stat().st_mtime), None


def _crop_centered_window(poses: np.ndarray, center: int, length: int) -> np.ndarray:
    """Extract ``length`` frames centered at ``center``, pad with edge frames like ``prepare_windows``."""
    f = poses.shape[0]
    half = length // 2
    start = int(center) - half
    end = start + length

    if start >= 0 and end <= f:
        return poses[start:end].copy()

    pad_pre = max(0, -start)
    pad_post = max(0, end - f)
    start_clamped = max(0, start)
    end_clamped = min(f, end)
    chunk = poses[start_clamped:end_clamped].copy()
    if chunk.shape[0] == 0:
        return np.zeros((length, 12, 2), dtype=np.float32)

    out_parts: list[np.ndarray] = []
    if pad_pre > 0:
        out_parts.append(np.tile(chunk[[0]], (pad_pre, 1, 1)))
    out_parts.append(chunk)
    if pad_post > 0:
        out_parts.append(np.tile(chunk[[-1]], (pad_post, 1, 1)))
    out = np.concatenate(out_parts, axis=0)
    if out.shape[0] < length:
        pad = length - out.shape[0]
        out = np.concatenate([out, np.tile(out[[-1]], (pad, 1, 1))], axis=0)
    elif out.shape[0] > length:
        out = out[:length]
    return out.astype(np.float32)


def _detector_batch_tensor(windows: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """``windows``: list of (T_det, 12, 2) → [N, 1, T, 12, 2]."""
    batch = []
    for w in windows:
        x = np.nan_to_num(w.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x = center_pose_on_hip_midpoint(x)
        batch.append(torch.from_numpy(x).reshape(1, 1, x.shape[0], 12, 2))
    return torch.cat(batch, dim=0).to(device)


def _classifier_batch_tensor(windows: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Training-aligned: sanitize clip then ``nan_to_num``."""
    batch = []
    for w in windows:
        x = _sanitize_pose_clip(w)
        x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        batch.append(torch.from_numpy(x).reshape(1, 1, x.shape[0], 12, 2))
    return torch.cat(batch, dim=0).to(device)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay punch-type predictions + skeleton on workout video (detector + classifier)."
    )
    ap.add_argument("--ver", type=str, default=None, help="Workbook tag, e.g. V7 (finds RGB_videos/{ver}_*.mp4)")
    ap.add_argument("--video", type=Path, default=None, help="Explicit path to source MP4 (overrides --ver)")
    ap.add_argument(
        "--poses",
        type=Path,
        default=None,
        help=f"Cached pose npz (default: {_POSE_SEQ_DIR}/{{ver}}_pose.npz)",
    )
    ap.add_argument(
        "--detection-npz",
        type=Path,
        default=None,
        help="Use labeled punch windows from this npz instead of the neural detector.",
    )
    ap.add_argument(
        "--use-detection-labels",
        action="store_true",
        help=f"Same as --detection-npz {_DEFAULT_DET_LABEL_DIR}/{{ver}}_detection.npz (requires --ver V#).",
    )
    ap.add_argument(
        "--classifier",
        type=Path,
        default=None,
        help="Classifier checkpoint (.pt). Default: newest checkpoints/gcn_classifier_*.pt",
    )
    ap.add_argument(
        "--detector",
        type=Path,
        default=None,
        help="Detector checkpoint (.pt). Default: prefer a file whose name includes this video's "
        "--ver tag (e.g. …_V7_…); else newest gcn_detector_*.pt",
    )
    ap.add_argument("--det-threshold", type=float, default=0.5, help="Detector probability threshold")
    ap.add_argument("--out", type=Path, default=None, help=f"Output MP4 (default: {_FIGURES}/…)")
    ap.add_argument("--max-frames", type=int, default=None, help="Encode only first N frames")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size for detector/classifier forwards")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_path: Path | None = args.video
    ver: str | None = args.ver.strip().upper() if args.ver else None

    if video_path is None:
        if ver is None:
            raise SystemExit("Provide --ver (e.g. V7) or --video /path/to.mp4")
        if not ver.startswith("V"):
            raise SystemExit("--ver should look like V7")
        vp = _find_video(ver)
        if vp is None:
            raise SystemExit(f"No video under Dataset/RGB_videos/source_youtube/{ver}_*.mp4")
        video_path = vp
    else:
        video_path = video_path.resolve()
        if not video_path.is_file():
            raise SystemExit(f"Not found: {video_path}")
        if ver is None:
            ver = _infer_ver_tag(video_path)
        if ver is None:
            ver = "VIDEO"

    assert video_path is not None

    clf_path = args.classifier if args.classifier is not None else _latest_ckpt("gcn_classifier_")
    if clf_path is None or not clf_path.is_file():
        raise SystemExit(
            "Missing classifier checkpoint — train with train_classifier.py or pass --classifier PATH"
        )

    det_ckpt_path = args.detector if args.detector is not None else None
    det_ckpt_warn: str | None = None
    if det_ckpt_path is None:
        det_ckpt_path, det_ckpt_warn = _default_detector_ckpt(ver)

    det_npz_path: Path | None = None
    if args.detection_npz is not None:
        det_npz_path = args.detection_npz.resolve()
        if not det_npz_path.is_file():
            raise SystemExit(f"Not found: {det_npz_path}")
    elif args.use_detection_labels:
        if not ver.startswith("V"):
            raise SystemExit("--use-detection-labels needs --ver V# (not an inferred VIDEO stem).")
        det_npz_path = (_DEFAULT_DET_LABEL_DIR / f"{ver}_detection.npz").resolve()
        if not det_npz_path.is_file():
            raise SystemExit(f"Not found: {det_npz_path}")

    use_label_npz = det_npz_path is not None
    if not use_label_npz and (det_ckpt_path is None or not det_ckpt_path.is_file()):
        raise SystemExit(
            "Missing detector checkpoint — train with train_detection.py, pass --detector PATH, "
            "or use --detection-npz / --use-detection-labels with a valid detection npz."
        )

    if det_ckpt_warn:
        print(det_ckpt_warn, file=sys.stderr)

    clf_sd = torch.load(clf_path, map_location="cpu", weights_only=False)
    clf_window = int(clf_sd.get("window", 20))
    punch_classes: list[str] = list(clf_sd.get("punch_classes", PUNCH_CLASSES))

    clf = GCNClassifier(
        num_classes=len(punch_classes),
        in_channels=2,
        num_joints=NUM_BOXINGVI_JOINTS,
        bone_pairs=BOXINGVI_BONE_PAIRS,
        backbone_kwargs={
            "edges": BOXINGVI_GRAPH_EDGES,
            "center": BOXINGVI_CENTER_JOINT,
            "dropout": 0.1,
            "data_bn": True,
        },
        dropout=0.3,
    )
    if clf_sd.get("model_state") is None:
        raise SystemExit(f"Checkpoint missing model_state: {clf_path}")
    clf.load_state_dict(clf_sd["model_state"])
    clf.eval()
    clf.to(device)

    det: GCNDetector | None = None
    det_wl = DET_WINDOW_LENGTH
    det_stride = DET_STRIDE
    w_starts_npz: np.ndarray | None = None
    w_is_punch_npz: np.ndarray | None = None

    if use_label_npz:
        assert det_npz_path is not None
        d = np.load(det_npz_path, allow_pickle=True)
        for k in ("window_starts", "window_is_punch", "window_length", "num_frames"):
            if k not in d:
                raise SystemExit(f"Missing {k!r} in {det_npz_path}")
        w_starts_npz = np.asarray(d["window_starts"])
        w_is_punch_npz = np.asarray(d["window_is_punch"])
        det_wl = int(d["window_length"])
        det_stride = int(d["stride"]) if "stride" in d else DET_STRIDE
        try:
            rel = det_npz_path.relative_to(_REPO)
        except ValueError:
            rel = det_npz_path
        print(f"Using punch windows from {rel} (neural detector skipped).")
    else:
        det = GCNDetector(
            in_channels=2,
            num_joints=NUM_BOXINGVI_JOINTS,
            bone_pairs=BOXINGVI_BONE_PAIRS,
            backbone_kwargs={
                "edges": BOXINGVI_GRAPH_EDGES,
                "center": BOXINGVI_CENTER_JOINT,
                "dropout": 0.1,
                "data_bn": True,
            },
            dropout=0,
        ).to(device)
        dsd = torch.load(det_ckpt_path, map_location="cpu", weights_only=False)
        det.load_state_dict(dsd["model_state"])
        det.eval()
        print(f"Detector checkpoint: {det_ckpt_path.relative_to(_REPO)}")

    print(f"Classifier checkpoint: {clf_path.relative_to(_REPO)}  |  clf_window={clf_window}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    n_frames_full = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    pose_path = args.poses if args.poses is not None else (_POSE_SEQ_DIR / f"{ver}_pose.npz")
    if pose_path.is_file():
        poses = load_pose_sequence_npz(pose_path, max_frames=None)
        print(f"Pose cache: {pose_path.relative_to(_REPO)}  shape={poses.shape}")
        if poses.shape[0] < n_frames_full:
            print(
                f"warning: pose rows {poses.shape[0]} < video FRAME_COUNT {n_frames_full} — truncating video.",
                file=sys.stderr,
            )
    else:
        print(f"No pose cache at {pose_path} — running MediaPipe on all frames …")
        poses = extract_full_video_poses(video_path, n_frames_full)

    nf = poses.shape[0]
    encode_n = nf if args.max_frames is None else min(nf, int(args.max_frames))

    # Per-frame display: best overlay among punch windows covering this frame
    frame_best_line: list[str] = ["" for _ in range(encode_n)]

    if use_label_npz:
        assert w_starts_npz is not None and w_is_punch_npz is not None
        punch_idx = np.flatnonzero(w_is_punch_npz > 0.5)
        score_pf = np.zeros(encode_n, dtype=np.float32)
        lines_pf = ["" for _ in range(encode_n)]

        bs = max(1, args.batch_size)
        k = 0
        while k < len(punch_idx):
            batch_pi = punch_idx[k : k + bs]
            clf_chunks: list[np.ndarray] = []
            for pi in batch_pi:
                t0 = int(w_starts_npz[pi])
                center = t0 + det_wl // 2
                clf_chunks.append(_crop_centered_window(poses, center, clf_window))
            xb = _classifier_batch_tensor(clf_chunks, device)
            probs_t = torch.softmax(clf(xb), dim=-1).cpu().numpy()
            for j, pi in enumerate(batch_pi):
                st = int(w_starts_npz[pi])
                pr = probs_t[j]
                pred_i = int(pr.argmax())
                name = punch_classes[pred_i].replace("_", " ").title()
                pc = float(pr[pred_i])
                line = f"Punch (labels): {name}  p={pc:.2f}"
                t_lo = max(0, st)
                t_hi = min(encode_n, st + det_wl)
                for t in range(t_lo, t_hi):
                    if pc >= score_pf[t]:
                        score_pf[t] = pc
                        lines_pf[t] = line
            k += bs

        frame_best_line = lines_pf
    else:
        assert det is not None
        max_start = encode_n - det_wl
        if max_start < 0:
            raise SystemExit(f"Need at least {det_wl} frames; video has {encode_n}")

        bs = max(1, args.batch_size)
        hits: list[tuple[int, float, np.ndarray]] = []
        max_det_seen = 0.0
        t0 = 0
        while t0 <= max_start:
            batch_starts: list[int] = []
            batch_windows: list[np.ndarray] = []
            while t0 <= max_start and len(batch_starts) < bs:
                batch_starts.append(t0)
                batch_windows.append(poses[t0 : t0 + det_wl])
                t0 += det_stride

            xdet = _detector_batch_tensor(batch_windows, device)
            det_p = det(xdet).cpu().numpy()
            max_det_seen = max(max_det_seen, float(det_p.max()))

            clf_chunks: list[np.ndarray] = []
            hit_ix: list[int] = []
            for i, st in enumerate(batch_starts):
                if float(det_p[i]) < args.det_threshold:
                    continue
                hit_ix.append(i)
                center = st + det_wl // 2
                clf_chunks.append(_crop_centered_window(poses, center, clf_window))

            if hit_ix:
                xb = _classifier_batch_tensor(clf_chunks, device)
                probs_t = torch.softmax(clf(xb), dim=-1).cpu().numpy()
                for j, bi in enumerate(hit_ix):
                    hits.append((batch_starts[bi], float(det_p[bi]), probs_t[j]))

        score_per_frame = np.zeros(encode_n, dtype=np.float32)
        line_per_frame: list[str | None] = [None] * encode_n

        for st, dp, pr in hits:
            pred_i = int(pr.argmax())
            name = punch_classes[pred_i].replace("_", " ").title()
            pc = float(pr[pred_i])
            combined = dp * pc
            line = f"Punch: {name}  det={dp:.2f}  cls={pc:.2f}"
            t_lo = max(0, st)
            t_hi = min(encode_n, st + det_wl)
            for t in range(t_lo, t_hi):
                if combined >= score_per_frame[t]:
                    score_per_frame[t] = combined
                    line_per_frame[t] = line

        frame_best_line = [x if x else "" for x in line_per_frame]

        if not hits:
            print(
                "\nNo frames passed the detector threshold "
                f"({args.det_threshold:g}). Max detector probability seen on this video: "
                f"{max_det_seen:.4f}. Try:\n"
                "  --det-threshold 0.25   (or lower)\n"
                f"  --detector checkpoints/gcn_detector_…  (trained with {ver} in the filename)\n"
                "  --use-detection-labels   (use Excel-derived punch windows; skips neural detector)\n",
                file=sys.stderr,
            )

    out_path = (
        args.out
        if args.out is not None
        else (_FIGURES / f"{ver}_classifier_preview.mp4").resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fps = 24.0
    cap = cv2.VideoCapture(str(video_path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"VideoWriter failed for {out_path}")

    try:
        for t in range(encode_n):
            ok, frame = cap.read()
            if not ok:
                print(f"warning: could not read frame {t}", file=sys.stderr)
                break
            if t < poses.shape[0]:
                _draw_pose_bgr(frame, poses[t])
            line1 = f"{ver}  |  frame {t + 1}/{encode_n}"
            line2 = frame_best_line[t] if t < len(frame_best_line) else ""
            if not line2.strip():
                line2 = "No punch window / below threshold (detector)"
            line3 = (
                f"clf_ckpt={clf_path.name}  |  joints NaN: "
                f"{int(np.sum(~np.all(np.isfinite(poses[t]), axis=1)))}/12"
                if t < poses.shape[0]
                else ""
            )
            _overlay_banner(frame, line1, line2, line3)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()

    print(f"\nDone. Wrote → {out_path.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
