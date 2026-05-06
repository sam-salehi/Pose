"""
Project MediaPipe pose skeleton onto a dataset video (V1–V10) and play it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONE-TIME SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pip install mediapipe opencv-python

The pose model (~6 MB) is downloaded automatically on first run.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python alpha.py          # V1 by default
    python alpha.py V3       # any of V1–V10

GUI controls:  [q / Esc] quit    [Space] pause/resume    [← →] ±10 frames
"""

from __future__ import annotations

import glob
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Repo layout ───────────────────────────────────────────────────────────────
_REPO   = Path(__file__).resolve().parent
_VIDEOS = _REPO / "Dataset" / "RGB_videos" / "source_youtube"

# ── Model ─────────────────────────────────────────────────────────────────────
_MODEL_PATH = _REPO / "pose_landmarker_full.task"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# ── BlazePose-33 skeleton connections ─────────────────────────────────────────
_POSE_CONNECTIONS = [
    (0,1),(0,4),(1,2),(2,3),(3,7),(4,5),(5,6),(6,8),   # face
    (9,10),                                              # mouth
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),  # L arm
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),     # R arm
    (11,23),(12,24),(23,24),                             # torso
    (23,25),(25,27),(27,29),(29,31),(27,31),             # L leg
    (24,26),(26,28),(28,30),(30,32),(28,32),             # R leg
]

CONF_THRESH = 0.5   # landmark visibility threshold

# Colour palette (one per connection)
_COLOURS = [
    (255,128,0),(255,153,51),(255,178,102),(230,230,0),(255,153,255),
    (153,204,255),(255,102,255),(255,51,255),(102,178,255),(51,153,255),
    (255,153,153),(255,102,102),(255,51,51),(153,255,153),(102,255,102),
    (51,255,51),(0,255,0),(0,0,255),(255,0,0),(0,255,255),
    (255,0,255),(51,255,255),(255,255,51),(112,133,182),(214,112,218),
    (0,160,82),(255,200,100),(100,200,255),(200,100,255),(100,255,200),
    (255,100,200),(200,255,100),(50,150,200),(200,150,50),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_model() -> None:
    if _MODEL_PATH.exists():
        return
    print(f"[alpha] Downloading pose model to {_MODEL_PATH} …", end=" ", flush=True)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("done")


def _find_video(version: str) -> Path:
    version = version.upper()
    matches = sorted(glob.glob(str(_VIDEOS / f"{version}_*.mp4")))
    if not matches:
        raise FileNotFoundError(
            f"No video found for {version} in {_VIDEOS}.\n"
            f"Expected a file like  {version}_<youtube-id>.mp4"
        )
    return Path(matches[0])


def _draw_poses(frame: np.ndarray, pose_landmarks_list) -> np.ndarray:
    h, w = frame.shape[:2]
    out  = frame.copy()
    for landmarks in pose_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for i, (a, b) in enumerate(_POSE_CONNECTIONS):
            if landmarks[a].visibility < CONF_THRESH or landmarks[b].visibility < CONF_THRESH:
                continue
            colour = _COLOURS[i % len(_COLOURS)]
            cv2.line(out, pts[a], pts[b], colour, 2, cv2.LINE_AA)

        for k, (x, y) in enumerate(pts):
            if landmarks[k].visibility < CONF_THRESH:
                continue
            colour = _COLOURS[k % len(_COLOURS)]
            cv2.circle(out, (x, y), 4, colour, -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), 4, (0, 0, 0),  1, cv2.LINE_AA)
    return out


def _overlay_info(frame: np.ndarray, text: str, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} fps", (w - 90, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def project_pose(version: str = "V1") -> None:
    """Run MediaPipe Pose on every frame of the video for *version* and display it."""
    _ensure_model()

    video_path = _find_video(version)
    print(f"[alpha] video : {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    win = f"MediaPipe Pose — {version}"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
    except cv2.error:
        print("[alpha] Warning: no GUI available — cannot display window", file=sys.stderr)
        cap.release()
        return

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,   # IMAGE mode: no timestamp ordering needed
        num_poses=4,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    paused    = False
    frame_idx = 0
    t_prev    = time.perf_counter()
    inf_fps   = 0.0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            if not paused:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_idx += 1

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results   = landmarker.detect(mp_image)

                vis = _draw_poses(frame_bgr, results.pose_landmarks)

                now     = time.perf_counter()
                inf_fps = 0.8 * inf_fps + 0.2 * (1.0 / max(now - t_prev, 1e-6))
                t_prev  = now

                n_persons = len(results.pose_landmarks)
                label     = f"{version}  frame {frame_idx}/{total_frames}  persons: {n_persons}"
                vis       = _overlay_info(vis, label, inf_fps)

                h, w  = vis.shape[:2]
                scale = min(1280 / w, 720 / h, 1.0)
                if scale < 1.0:
                    vis = cv2.resize(vis, (int(w * scale), int(h * scale)), cv2.INTER_AREA)

                cv2.imshow(win, vis)

            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == 81 or key == ord("a"):    # ← back 10 frames
                target = max(0, frame_idx - 11)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                frame_idx = target
            if key == 83 or key == ord("d"):    # → forward 10 frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 10)
                frame_idx += 10

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "V1"
    project_pose(ver)
