"""
Paper CVCI 2024 §4.2.1 style features for pose sequences (windowed clips).

All encoders expect **pixel-space** windows (N, 20, 12, 2) matching ``landmarks.npz``
after ``prepare_windows``. Each applies Eq. (5) sequence reference centering (neck +
pelvis midpoints), then branch-specific tensors.

Branches
--------
- **UAE**: static angular encoding d(u) (Eq. 6) + forward temporal difference (Eq. 7)
  on those angles → (N, 20, 8).
- **2DMDD**: Taylor stack [P; P_v; P_a] with central differences on centered joints (Eq. 8)
  → (N, 20, 72).
- **FAE**: fifth-order angular encoding on θ, central differences (Eq. 9, math.md)
  → (N, 16, 12) timesteps or (N, 192) flat for KNN.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8

_MOVING = (2, 3, 4, 5)


def _sanitize_xy(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _theta_static_from_xy(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sequence-level centering (Eq. 5) and static angular encoding (Eq. 6) per moving joint.

    Returns
    -------
    P : (N, 20, 12, 2) centered coordinates
    theta : (N, 20, 4) same as FAE static angles
    """
    X = _sanitize_xy(X)
    N = X.shape[0]
    neck = (X[:, :, 0, :] + X[:, :, 1, :]) / 2.0
    pelvis = (X[:, :, 6, :] + X[:, :, 7, :]) / 2.0
    r_bar = (neck + pelvis).sum(axis=1) / (2.0 * 20.0)

    P = X - r_bar[:, np.newaxis, np.newaxis, :]
    neck_c = (P[:, :, 0, :] + P[:, :, 1, :]) / 2.0
    pelvis_c = (P[:, :, 6, :] + P[:, :, 7, :]) / 2.0

    P_u = P[:, :, list(_MOVING), :]
    b_neck = neck_c[:, :, np.newaxis, :] - P_u
    b_pelvis = pelvis_c[:, :, np.newaxis, :] - P_u
    dot = (b_neck * b_pelvis).sum(axis=-1)
    norm_n = np.linalg.norm(b_neck, axis=-1)
    norm_p = np.linalg.norm(b_pelvis, axis=-1)
    theta = 1.0 - dot / (norm_n * norm_p + EPS)
    return P, theta


def encode_uae(X: np.ndarray) -> np.ndarray:
    """
    Unified-axis angular encoding (§4.2.1): static d(u) + forward difference (Eq. 7).

    Returns (N, 20, 8): concat of theta (4) and v_t = d_t - d_{t-1} with v_0 = 0.
    """
    _, theta = _theta_static_from_xy(X)
    N, T, J = theta.shape
    v_fwd = np.zeros_like(theta)
    v_fwd[:, 1:, :] = theta[:, 1:, :] - theta[:, :-1, :]
    return np.concatenate([theta, v_fwd], axis=-1).astype(np.float32)


def encode_2dmdd(X: np.ndarray) -> np.ndarray:
    """
    2D motion dynamics descriptors: [P_t; P_t^v; P_t^a] with central differences (Eq. 8).

    P uses sequence-centered (x,y) stacked as 24-D per frame. Velocity / acceleration
    rows are zero where the stencil is out of range.
    """
    P, _ = _theta_static_from_xy(X)
    N = P.shape[0]
    Pf = P.reshape(N, 20, 24)
    Pv = np.zeros_like(Pf)
    Pa = np.zeros_like(Pf)
    # v_t = P_{t+1} - P_{t-1}, t = 1 .. 18
    Pv[:, 1:19, :] = Pf[:, 2:20, :] - Pf[:, 0:18, :]
    # a_t = P_{t+2} + P_{t-2} - 2 P_t, t = 2 .. 17
    Pa[:, 2:18, :] = (
        Pf[:, 4:20, :]
        + Pf[:, 0:16, :]
        - 2.0 * Pf[:, 2:18, :]
    )
    return np.concatenate([Pf, Pv, Pa], axis=-1).astype(np.float32)


def encode_fae_timesteps(X: np.ndarray) -> np.ndarray:
    """
    Fifth-order angular encoding (Eq. 9 / math.md): aligned θ, θ_v, θ_a → (N, 16, 12).
    """
    _, theta = _theta_static_from_xy(X)
    N = theta.shape[0]

    V = theta[:, 2:20, :] - theta[:, 0:18, :]
    t = np.arange(2, 18)
    acc = theta[:, t + 2, :] + theta[:, t - 2, :] - 2.0 * theta[:, t, :]

    theta_a = theta[:, 2:18, :]
    vel_a = V[:, 2:18, :]
    feat = np.concatenate([theta_a, vel_a, acc], axis=-1)
    return feat.astype(np.float32)


def encode_fae_flat(X: np.ndarray) -> np.ndarray:
    """Same as ``encode_fae_timesteps`` then flatten → (N, 192)."""
    ft = encode_fae_timesteps(X)
    return ft.reshape(len(ft), -1)


def encode_branch(X: np.ndarray, branch: str) -> tuple[np.ndarray, int, int]:
    """
    Encode windowed clips for deep sequence models.

    Returns
    -------
    array : (N, T, F)
    T, F  : time length and feature dim for PunchNet
    """
    b = branch.lower().strip()
    if b == "uae":
        z = encode_uae(X)
        return z, z.shape[1], z.shape[2]
    if b in ("2dmdd", "mdd", "2dmd"):
        z = encode_2dmdd(X)
        return z, z.shape[1], z.shape[2]
    if b == "fae":
        z = encode_fae_timesteps(X)
        return z, z.shape[1], z.shape[2]
    raise ValueError(f"unknown branch {branch!r} (use uae, 2dmdd, fae)")
