# --- Build pose cache for one workbook (repeat or loop V1–V10) FIXME: seems to be repeating other file ---
from pathlib import Path

from preprocess import _find_video
from detector_data import build_detector_windows_npz
import sys

_REPO = Path.cwd().resolve()
DET_VER = sys.argv[1]

DET_NPZ = _REPO / "Dataset" / "detection_frame_labels" / f"{DET_VER}_detection.npz"
OUT_CACHE = _REPO / "Dataset" / "detector_training" / f"{DET_VER}_windows.npz"
VIDEO = _find_video(DET_VER)

MAX_WINDOWS = 8000  # None = keep every window after pose extract (very large)

assert DET_NPZ.exists(), "Run the preprocessing cell first."
assert VIDEO is not None and VIDEO.exists(), f"No MP4 for {DET_VER}"

build_detector_windows_npz(
    DET_NPZ,
    VIDEO,
    OUT_CACHE,
    max_windows=MAX_WINDOWS,
    balance=True,
    seed=42,
)
print(f"Saved detector cache → {OUT_CACHE.relative_to(_REPO)}")


