"""
Build punch-detector training tensors: one MediaPipe pass per video, then stack windows.

Reads ``Dataset/detection_frame_labels/{ver}_detection.npz`` (from ``train.ipynb`` preprocessing)
and the matching RGB MP4, writes a compact cache for ``GCNDetector`` training.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from torch.utils.data import Dataset

from preprocess import _LANDMARK_IDX, _ensure_model

_REPO = Path(__file__).resolve().parent
_MODEL_PATH = _REPO / "pose_landmarker_full.task"


def extract_full_video_poses(
    video_path: Path,
    num_frames: int,
    *,
    min_confidence: float = 0.5,
) -> np.ndarray:
    """
    Sequential decode + MediaPipe pose. Returns ``(F, 12, 2)`` float32 (NaN if missing).
    ``num_frames`` should match OpenCV ``FRAME_COUNT`` from the detection npz.
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
    import mediapipe as mp

    out = np.full((num_frames, 12, 2), np.nan, dtype=np.float32)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return out

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for t in range(num_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_img)
            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                for j, idx in enumerate(_LANDMARK_IDX):
                    lm = lms[idx]
                    if lm.visibility >= min_confidence:
                        out[t, j, 0] = lm.x
                        out[t, j, 1] = lm.y

            if t % 100 == 0:
                print(f"Frame: ",t, "/", num_frames)
    cap.release()
    return out


def sample_window_indices(
    window_is_punch: np.ndarray,
    max_windows: int | None,
    *,
    balance: bool,
    seed: int,
) -> np.ndarray:
    """Return indices into ``range(K)`` where K = len(window_is_punch)."""
    rng = np.random.default_rng(seed)
    K = len(window_is_punch)
    if max_windows is None or max_windows >= K:
        return np.arange(K, dtype=np.int64)

    pos_idx = np.flatnonzero(window_is_punch > 0)
    neg_idx = np.flatnonzero(window_is_punch == 0)

    if balance and len(pos_idx) > 0 and len(neg_idx) > 0:
        half = max_windows // 2
        take_pos = min(len(pos_idx), half)
        take_neg = min(len(neg_idx), max_windows - take_pos)
        if take_neg < half:
            take_pos = min(len(pos_idx), max_windows - take_neg)
        sel_pos = rng.choice(pos_idx, size=take_pos, replace=False)
        sel_neg = rng.choice(neg_idx, size=take_neg, replace=False)
        out = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(out)
        return out.astype(np.int64)

    return rng.choice(K, size=min(max_windows, K), replace=False).astype(np.int64)


def stack_windows(
    poses: np.ndarray,
    window_starts: np.ndarray,
    indices: np.ndarray,
    window_length: int,
) -> np.ndarray:
    """``poses`` (F,12,2) → ``(N, T, 12, 2)`` for selected window indices."""
    X = np.empty((len(indices), window_length, 12, 2), dtype=np.float32)
    F = poses.shape[0]
    for k, wi in enumerate(indices):
        t0 = int(window_starts[wi])
        end = t0 + window_length
        if t0 < 0 or end > F:
            X[k] = 0.0
            continue
        X[k] = poses[t0:end]
    return X


def build_detector_windows_npz(
    detection_npz: Path,
    video_path: Path,
    out_npz: Path,
    *,
    max_windows: int | None = 16_000,
    balance: bool = True,
    seed: int = 42,
    min_confidence: float = 0.5,
) -> Path:
    """
    Cache joint windows + binary punch labels for one video.

    Parameters
    ----------
    max_windows
        Cap training size (balanced pos/neg when ``balance=True``). ``None`` = all windows.
    """
    data = np.load(detection_npz, allow_pickle=True)
    window_starts = np.asarray(data["window_starts"])
    window_is_punch = np.asarray(data["window_is_punch"])
    wl = int(data["window_length"])
    nf = int(data["num_frames"])

    sel = sample_window_indices(window_is_punch, max_windows, balance=balance, seed=seed)
    poses = extract_full_video_poses(video_path, nf, min_confidence=min_confidence)
    X = stack_windows(poses, window_starts, sel, wl)
    y = window_is_punch[sel].astype(np.float32)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        joints=X,
        labels=y,
        window_indices=sel,
        window_length=np.int32(wl),
        source_video=np.array(str(video_path)),
        detection_npz=np.array(str(detection_npz)),
    )
    return out_npz


def load_pose_sequence_npz(
    path: Path | str,
    *,
    max_frames: int | None = None,
) -> np.ndarray:
    """
    Load a full-video pose cache written by ``extract_pose.py`` /
    ``Dataset/pose_sequences/Vx_pose.npz``: key ``pose`` with shape ``(F, 12, 2)``.
    key ``pose`` with shape ``(F, 12, 2)`` float32.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    if "pose" not in d:
        raise ValueError(f"{path} missing 'pose' array")
    pose = np.asarray(d["pose"], dtype=np.float32)
    if pose.ndim != 3 or pose.shape[-2:] != (12, 2):
        raise ValueError(f"expected pose (F, 12, 2), got {pose.shape}")
    if max_frames is not None:
        pose = pose[: int(max_frames)]
    return pose


class DetectorWindowNpzDataset(Dataset):
    """Loads ``build_detector_windows_npz`` output; yields ``x`` [1,1,T,12,2], ``y`` scalar."""

    def __init__(self, npz_path: Path | str):
        d = np.load(npz_path, allow_pickle=True)
        self.X = np.asarray(d["joints"], dtype=np.float32)
        self.y = np.asarray(d["labels"], dtype=np.float32)
        if self.X.ndim != 4 or self.X.shape[-2:] != (12, 2):
            raise ValueError(f"expected joints (N,T,12,2), got {self.X.shape}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.nan_to_num(self.X[i], nan=0.0, posinf=0.0, neginf=0.0)
        x = center_pose_on_hip_midpoint(x.astype(np.float32))
        n_frames = int(x.shape[0])
        # Explicit [1, 1, T, 12, 2] so DataLoader batches to [N, 1, T, 12, 2] for ST-GCN.
        t = torch.from_numpy(x).reshape(1, 1, n_frames, 12, 2).contiguous()
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return t, y
