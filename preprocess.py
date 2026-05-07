"""Preprocessing utilities: clip preview and landmark extraction for boxing pose dataset."""

from __future__ import annotations

import glob
import random
import re
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from openpyxl import load_workbook

_REPO        = Path(__file__).resolve().parent
_ANNOTATIONS = _REPO / "Dataset" / "Annotation_files"
_VIDEOS      = _REPO / "Dataset" / "RGB_videos" / "source_youtube"
_MODEL_PATH  = _REPO / "pose_landmarker_full.task"
_MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# ── Landmark mapping ───────────────────────────────────────────────────────────
MEDIAPIPE_TO_PAPER = {
    'left_shoulder':  11,
    'right_shoulder': 12,
    'left_elbow':     13,
    'right_elbow':    14,
    'left_wrist':     15,
    'right_wrist':    16,
    'left_hip':       23,
    'right_hip':      24,
    'left_knee':      25,
    'right_knee':     26,
    'left_ankle':     27,
    'right_ankle':    28,
}

# Ordered indices (same order as MEDIAPIPE_TO_PAPER keys)
_LANDMARK_IDX: list[int] = list(MEDIAPIPE_TO_PAPER.values())

_LABEL_NORM = {
    'cross':        'Cross',
    'lead hook':    'Lead Hook',
    'rear uppercut':'Rear Uppercut',
}


def _normalize_label(label: str) -> str:
    return _LABEL_NORM.get(label.strip().lower(), label.strip())


def _ensure_model() -> None:
    if _MODEL_PATH.exists():
        return
    print(f"Downloading pose model to {_MODEL_PATH} …", end=" ", flush=True)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("done")


def _load_annotations(xlsx_path: Path) -> list[tuple[int, int, str]]:
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active

    # V1/V9/V10: header has text at cols 0 and 2 with None spacer at col 1 → sparse (0,2,4)
    # V2–V8: compact headers or all-None header row → dense (0,1,2)
    header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
    is_sparse = len(header) > 4 and header[1] is None and header[2] is not None
    sc, ec, lc = (0, 2, 4) if is_sparse else (0, 1, 2)

    rows: list[tuple[int, int, str]] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or len(row) <= lc:
            continue
        start, end, label = row[sc], row[ec], row[lc]
        if start is None or end is None or label is None:
            continue
        try:
            s, e = int(start), int(end)
        except (TypeError, ValueError):
            continue
        if e >= s:
            rows.append((s, e, str(label).strip()))
    wb.close()
    return rows


def _find_video(version: str) -> Path | None:
    matches = sorted(glob.glob(str(_VIDEOS / f"{version}_*.mp4")))
    return Path(matches[0]) if matches else None


def _opencv_has_gui() -> bool:
    try:
        cv2.namedWindow("__probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe")
        return True
    except cv2.error:
        return False


def _safe_name(s: str, max_len: int = 40) -> str:
    return re.sub(r"[^\w\-]+", "_", s).strip("_")[:max_len] or "clip"


def preview_clips(
    n_per_version: int = 1,
    *,
    seed: int | None = None,
    speed: float = 1.0,
    export_dir: Path | str | None = None,
) -> None:
    """
    Pick n_per_version random clips from each of V1–V10 and play them with
    the label and frame range overlaid so you can confirm annotation timing.

    GUI controls: [q / Esc] quit   [n] next clip
    No GUI (headless OpenCV): clips are saved as MP4s to export_dir.
    """
    if seed is not None:
        random.seed(seed)

    all_clips: list[tuple[str, Path, int, int, str]] = []
    for i in range(1, 11):
        ver = f"V{i}"
        video = _find_video(ver)
        if video is None:
            print(f"[skip] no video for {ver}", file=sys.stderr)
            continue
        annotations = _load_annotations(_ANNOTATIONS / f"{ver}.xlsx")
        if not annotations:
            print(f"[skip] no annotations for {ver}", file=sys.stderr)
            continue
        for s, e, label in random.sample(annotations, min(n_per_version, len(annotations))):
            all_clips.append((ver, video, s, e, label))

    if not all_clips:
        raise RuntimeError("No clips found — check Dataset/ paths.")

    use_gui = _opencv_has_gui()
    out_root = Path(export_dir) if export_dir else _REPO / ".cache" / "clip_previews"
    if not use_gui:
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"No GUI — writing clips to {out_root.resolve()}\n", file=sys.stderr)

    total = len(all_clips)
    print(f"Previewing {total} clips ({n_per_version} per version, seed={seed})\n", file=sys.stderr)
    if use_gui:
        print("  [n] next clip   [q / Esc] quit\n", file=sys.stderr)

    for idx, (ver, video_path, s1, e1, label) in enumerate(all_clips, 1):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        delay_ms = max(1, int(1000.0 / fps / speed))

        i0, i1 = s1 - 1, e1 - 1  # convert 1-based inclusive → 0-based
        t0, t1 = i0 / fps, i1 / fps
        tag = f"[{idx}/{total}] {ver} | {label} | frames {s1}–{e1}  ({t0:.1f}s–{t1:.1f}s)"
        print(tag, file=sys.stderr)

        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)

        writer: cv2.VideoWriter | None = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = out_root / f"{idx:02d}_{ver}_{_safe_name(label)}_{s1}-{e1}.mp4" if not use_gui else None

        skip = False
        for fi in range(i0, i1 + 1):
            ok, frame = cap.read()
            if not ok:
                print(f"  warning: could not read frame {fi}", file=sys.stderr)
                break

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(frame, tag[:120], (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            if use_gui:
                scale = min(1280 / w, 720 / h, 1.0)
                disp = cv2.resize(frame, (int(w * scale), int(h * scale)), cv2.INTER_AREA) if scale < 1 else frame
                cv2.imshow("Clip preview", disp)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                if key == ord("n"):
                    skip = True
                    break
            else:
                if writer is None:
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"VideoWriter failed for {out_path}")
                writer.write(frame)

        cap.release()
        if writer is not None:
            writer.release()
            print(f"  saved {out_path}", file=sys.stderr)

    if use_gui:
        cv2.destroyAllWindows()


def extract_landmarks(
    *,
    min_confidence: float = 0.5,
) -> tuple[list[np.ndarray], list[str], list[dict]]:
    """
    Extract 12-landmark pose sequences from every annotated boxing clip.

    For each annotation (start_frame, end_frame, label) across V1–V10, reads
    every frame in that range, runs MediaPipe PoseLandmarker, and records the
    12 landmarks in MEDIAPIPE_TO_PAPER as (x, y) normalised image coordinates.

    Landmarks with visibility < min_confidence are stored as NaN.

    Returns:
        sequences : list of float32 arrays shaped (T, 12, 2), one per clip.
                    T = end_frame - start_frame + 1; varies across clips.
        labels    : normalised class-label string for each clip.
        meta      : list of dicts with keys 'version', 'start_frame', 'end_frame'.
    """
    _ensure_model()

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    sequences: list[np.ndarray] = []
    labels:    list[str]        = []
    meta_list: list[dict]       = []

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for v in range(1, 11):
            ver        = f"V{v}"
            video_path = _find_video(ver)
            if video_path is None:
                print(f"[skip] no video for {ver}", file=sys.stderr)
                continue

            annotations = _load_annotations(_ANNOTATIONS / f"{ver}.xlsx")
            if not annotations:
                print(f"[skip] no annotations for {ver}", file=sys.stderr)
                continue

            print(f"[{ver}] {len(annotations)} clips — {video_path.name}")
            cap = cv2.VideoCapture(str(video_path))

            for clip_idx, (s, e, label) in enumerate(annotations):
                i0 = s - 1          # 1-based inclusive → 0-based
                T  = e - s + 1
                seq = np.full((T, 12, 2), np.nan, dtype=np.float32)

                cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
                for t in range(T):
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    result    = landmarker.detect(mp_img)

                    if result.pose_landmarks:
                        lms = result.pose_landmarks[0]
                        for j, idx in enumerate(_LANDMARK_IDX):
                            lm = lms[idx]
                            if lm.visibility >= min_confidence:
                                seq[t, j, 0] = lm.x
                                seq[t, j, 1] = lm.y

                sequences.append(seq)
                labels.append(_normalize_label(label))
                meta_list.append({'version': ver, 'start_frame': s, 'end_frame': e})

                if (clip_idx + 1) % 20 == 0:
                    print(f"  {clip_idx + 1}/{len(annotations)} clips done")

            cap.release()
            print(f"[{ver}] complete")

    return sequences, labels, meta_list


def build_dataset(
    output_path: Path | str | None = None,
    *,
    min_confidence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build and save the landmark dataset for all annotated clips.

    Each clip is represented as a (T, 12, 2) float32 array (T varies per clip).
    Saves an .npz file containing:
        sequences    – object array of shape (N,), each element is (T_i, 12, 2)
        labels       – (N,) string array of class labels
        versions     – (N,) string array, e.g. 'V1'
        start_frames – (N,) int array
        end_frames   – (N,) int array

    For training, pass ``sequences`` and labels through :func:`prepare_windows` → (N, 20, 12, 2).

    Load with:
        data = np.load('Dataset/landmarks.npz', allow_pickle=True)
        X, y = data['sequences'], data['labels']

    Args:
        output_path: destination .npz file.  Defaults to Dataset/landmarks.npz.
        min_confidence: MediaPipe visibility threshold.

    Returns:
        sequences as an object array and labels as a string array.
    """
    if output_path is None:
        output_path = _REPO / "Dataset" / "landmarks.npz"
    output_path = Path(output_path)

    sequences, labels, meta = extract_landmarks(min_confidence=min_confidence)

    seq_arr = np.empty(len(sequences), dtype=object)
    for i, s in enumerate(sequences):
        seq_arr[i] = s

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        sequences    = seq_arr,
        labels       = np.array(labels),
        versions     = np.array([m['version']     for m in meta]),
        start_frames = np.array([m['start_frame'] for m in meta]),
        end_frames   = np.array([m['end_frame']   for m in meta]),
    )
    print(f"\nSaved {len(sequences)} clips → {output_path}")
    return seq_arr, np.array(labels)


def _interpolate_partial_nans_along_time(seq: np.ndarray) -> np.ndarray:
    """
    Linear interpolation along time for each joint coordinate.

    MediaPipe often leaves **partial** frames (some joints NaN). Whole-frame
    forward-fill does not touch those; zeros after nan_to_num distort the skeleton.
    """
    seq = np.asarray(seq, dtype=np.float32).copy()
    T = seq.shape[0]
    if T < 2:
        return seq
    t = np.arange(T, dtype=np.float64)
    for j in range(12):
        for d in range(2):
            v = seq[:, j, d]
            if not np.any(np.isnan(v)):
                continue
            good = np.isfinite(v)
            if not np.any(good):
                continue
            bad = ~good
            seq[bad, j, d] = np.interp(t[bad], t[good], v[good]).astype(np.float32)
    return seq


def _sanitize_pose_clip(seq: np.ndarray) -> np.ndarray:
    """
    Copy with forward/backward NaN handling within one clip (matches prepare_windows).

    Whole-frame fill first, then **per-joint temporal interpolation** for brief
    occlusions (partial NaNs).

    seq : (T, 12, 2)
    """
    seq = np.array(seq, dtype=np.float32).copy()
    T = seq.shape[0]

    mask = np.all(np.isnan(seq), axis=(1, 2))
    last_good = seq[0].copy() if not mask[0] else np.zeros((12, 2), dtype=np.float32)
    for t in range(T):
        if mask[t]:
            seq[t] = last_good
        else:
            last_good = seq[t]

    last_good = seq[-1].copy()
    for t in range(T - 1, -1, -1):
        if np.all(seq[t] == 0):
            seq[t] = last_good
        else:
            last_good = seq[t]

    seq = _interpolate_partial_nans_along_time(seq)
    return seq


def prepare_windows(
    sequences: np.ndarray | list,
    labels: np.ndarray | list,
    *,
    window: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the paper's windowing to pre-trimmed clips (BoxingVI case).

    For each clip the peak is the middle frame.  A symmetric window of
    `window` frames is sliced out:
        [peak - window//2  …  peak + window//2)
    so the default gives frames [peak-10 … peak+9] → shape (20, 12, 2).

    Clips shorter than `window` frames are padded by repeating the first /
    last valid frame.  Frames where the pose detector returned NaN are
    forward-then-backward filled within the clip; any remaining NaN (whole
    clip undetected) becomes 0.

    Args:
        sequences : object array or list of (T_i, 12, 2) float32 arrays.
        labels    : matching array / list of label strings.
        window    : total window length (default 20).

    Returns:
        X : (N, window, 12, 2) float32
        y : (N,) label-string array   ← encode to int in train.py
    """
    half = window // 2
    X_out, y_out = [], []

    for seq, lbl in zip(sequences, labels):
        seq = _sanitize_pose_clip(seq)
        T   = seq.shape[0]

        # ── pad if clip is shorter than window ───────────────────────────────
        if T < window:
            pad_pre  = (window - T) // 2
            pad_post = window - T - pad_pre
            seq = np.concatenate([
                np.tile(seq[[0]], (pad_pre,  1, 1)),
                seq,
                np.tile(seq[[-1]], (pad_post, 1, 1)),
            ], axis=0)
            T = window

        # ── extract centred window ────────────────────────────────────────────
        peak  = T // 2
        start = max(0, peak - half)
        chunk = seq[start : start + window]

        # handle clips where peak is very close to an edge
        if chunk.shape[0] < window:
            pad = window - chunk.shape[0]
            chunk = np.concatenate([chunk, np.tile(chunk[[-1]], (pad, 1, 1))], axis=0)

        X_out.append(chunk)
        y_out.append(str(lbl))

    return np.stack(X_out, axis=0), np.array(y_out)


def validate_landmarks(
    n_clips: int = 30,
    *,
    seed: int | None = 42,
    min_confidence: float = 0.5,
    output_path: Path | str | None = None,
) -> None:
    """
    Sanity-check landmark extraction on a random sample of clips.

    Picks n_clips annotations at random across V1–V10, runs MediaPipe on the
    middle frame of each, overlays the 12 MEDIAPIPE_TO_PAPER keypoints, and
    saves a grid image so you can visually confirm the points land correctly.

    Args:
        n_clips:     how many clips to sample (default 30 → 5 × 6 grid).
        seed:        random seed for reproducibility.
        min_confidence: visibility threshold passed to MediaPipe.
        output_path: where to save the grid image.
                     Defaults to Dataset/landmark_check.png.
    """
    import matplotlib
    matplotlib.use("Agg")          # always write to file; avoids display issues
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if output_path is None:
        output_path = _REPO / "Dataset" / "landmark_check.png"
    output_path = Path(output_path)

    if seed is not None:
        random.seed(seed)

    # ── Collect all (version, video_path, start, end, label) tuples ───────────
    pool: list[tuple[str, Path, int, int, str]] = []
    for v in range(1, 11):
        ver = f"V{v}"
        vp  = _find_video(ver)
        if vp is None:
            continue
        for s, e, label in _load_annotations(_ANNOTATIONS / f"{ver}.xlsx"):
            pool.append((ver, vp, s, e, label))

    if not pool:
        raise RuntimeError("No annotated clips found — check Dataset/ paths.")

    sample = random.sample(pool, min(n_clips, len(pool)))

    _ensure_model()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    # Colour per landmark (12 distinct colours)
    _COLOURS = [
        "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
        "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
        "#469990","#dcbeff",
    ]
    joint_names = list(MEDIAPIPE_TO_PAPER.keys())  # 12 names in order

    # ── Grid layout ────────────────────────────────────────────────────────────
    ncols = 6
    nrows = (len(sample) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).reshape(-1)

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for ax, (ver, vp, s, e, label) in zip(axes, sample):
            mid_frame = (s + e) // 2          # 1-based
            cap = cv2.VideoCapture(str(vp))
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame - 1)
            ok, frame_bgr = cap.read()
            cap.release()

            if not ok:
                ax.axis("off")
                ax.set_title(f"{ver} | {label}\n(frame read failed)", fontsize=7)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w      = frame_rgb.shape[:2]

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_img)

            ax.imshow(frame_rgb)
            ax.axis("off")
            ax.set_title(
                f"{ver} | {_normalize_label(label)}\nframe {mid_frame}",
                fontsize=7, pad=2,
            )

            detected = False
            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                for j, (mp_idx, colour, name) in enumerate(
                    zip(_LANDMARK_IDX, _COLOURS, joint_names)
                ):
                    lm = lms[mp_idx]
                    if lm.visibility < min_confidence:
                        continue
                    detected = True
                    px, py = lm.x * w, lm.y * h
                    ax.plot(px, py, "o", color=colour, markersize=5,
                            markeredgewidth=0.5, markeredgecolor="white")

            if not detected:
                ax.text(
                    w / 2, h / 2, "no pose detected",
                    ha="center", va="center", color="red",
                    fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
                )

    # Turn off any unused axes
    for ax in axes[len(sample):]:
        ax.axis("off")

    # Legend mapping colour → joint name
    patches = [
        mpatches.Patch(color=c, label=n)
        for c, n in zip(_COLOURS, joint_names)
    ]
    fig.legend(
        handles=patches, ncol=6, loc="lower center",
        fontsize=6.5, framealpha=0.8,
        bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1,
    )

    fig.suptitle(
        f"Landmark extraction check — {len(sample)} random clips "
        f"(middle frame, confidence ≥ {min_confidence})",
        fontsize=10, y=1.01,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved validation grid → {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["preview", "extract", "validate"], default="preview")
    ap.add_argument("--output", default=None, help="output path for extract / validate modes")
    ap.add_argument("--n-clips", type=int, default=30, help="clips to sample in validate mode")
    ap.add_argument("--seed", type=int, default=42, help="random seed for validate mode")
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="extract / validate: MediaPipe landmark visibility threshold (try 0.35–0.5)",
    )
    args = ap.parse_args()

    if args.mode == "extract":
        build_dataset(args.output, min_confidence=args.min_confidence)
    elif args.mode == "validate":
        validate_landmarks(
            args.n_clips,
            seed=args.seed,
            output_path=args.output,
            min_confidence=args.min_confidence,
        )
    else:
        preview_clips(n_per_version=1, seed=None)
