"""
Frame Angular Encoding (FAE) — Steps 1–9 from math.md.

Maps windowed pose clips (N, 20, 12, 2) in pixel space to (N, 192) feature vectors.
Any NaN/inf inputs are mapped to finite values before encoding (defensive; ``train.load_dataset``
also sanitizes windows after ``prepare_windows``).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8

# Moving joints for angular features (math.md): elbows + wrists
_MOVING = (2, 3, 4, 5)


def encode_fae(X: np.ndarray) -> np.ndarray:
    """
    Encode pose windows with FAE.

    Parameters
    ----------
    X : (N, 20, 12, 2) float — joint coordinates (same order as math.md table).

    Returns
    -------
    (N, 192) float32
    """
    if X.ndim != 4 or X.shape[1:] != (20, 12, 2):
        raise ValueError(f"Expected X shape (N, 20, 12, 2), got {X.shape}")

    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    N = X.shape[0]
    # Neck / pelvis midpoints per frame
    neck = (X[:, :, 0, :] + X[:, :, 1, :]) / 2.0
    pelvis = (X[:, :, 6, :] + X[:, :, 7, :]) / 2.0

    # Sequence reference (Step 2)
    r_bar = (neck + pelvis).sum(axis=1) / (2.0 * 20.0)  # (N, 2)

    P = X - r_bar[:, np.newaxis, np.newaxis, :]

    neck_c = (P[:, :, 0, :] + P[:, :, 1, :]) / 2.0
    pelvis_c = (P[:, :, 6, :] + P[:, :, 7, :]) / 2.0

    # (N, 20, 4, 2) joint positions for moving indices
    P_u = P[:, :, list(_MOVING), :]
    b_neck = neck_c[:, :, np.newaxis, :] - P_u
    b_pelvis = pelvis_c[:, :, np.newaxis, :] - P_u

    dot = (b_neck * b_pelvis).sum(axis=-1)
    norm_n = np.linalg.norm(b_neck, axis=-1)
    norm_p = np.linalg.norm(b_pelvis, axis=-1)
    theta = 1.0 - dot / (norm_n * norm_p + EPS)  # (N, 20, 4)

    # Velocities at frames 1..18 (paper); index k = 0..17 → θ[k+1] - θ[k-1]
    V = theta[:, 2:20, :] - theta[:, 0:18, :]  # (N, 18, 4)

    t = np.arange(2, 18)
    acc = (
        theta[:, t + 2, :]
        + theta[:, t - 2, :]
        - 2.0 * theta[:, t, :]
    )  # (N, 16, 4)

    theta_a = theta[:, 2:18, :]
    vel_a = V[:, 2:18, :]

    feat = np.concatenate([theta_a, vel_a, acc], axis=-1)  # (N, 16, 12)
    out = feat.reshape(N, -1).astype(np.float32)
    return out
