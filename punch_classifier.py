"""
Rule-based boxing punch classifier for MotionBERT H36M-17 3D pose.

Input: (T, 17, 3) numpy array (one punch window from X3D.npy).
Output: integer label 1–6:
    1 jab  2 cross  3 lead hook  4 rear hook  5 lead uppercut  6 rear uppercut

Classification is a two-stage decomposition (see boxing.md):
    Stage A — which hand (lead vs rear), from wrist speed + stance geometry
    Stage B — trajectory family (straight / hook / uppercut), from wrist path geometry

Stage B uses hard thresholds first; ties fall to a soft weighted sum.
The nine weights w1–w9 default to 1.0 and can be tuned on a labelled set.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

# ── H36M-17 joint index map (MotionBERT convention) ──────────────────────────
_J = {
    "pelvis":     0,
    "r_hip":      1, "r_knee": 2, "r_ankle": 3,
    "l_hip":      4, "l_knee": 5, "l_ankle": 6,
    "spine":      7, "thorax": 8, "neck": 9, "head": 10,
    "l_shoulder": 11, "l_elbow": 12, "l_wrist": 13,
    "r_shoulder": 14, "r_elbow": 15, "r_wrist": 16,
}

PUNCH_LABELS: dict[int, str] = {
    1: "jab",
    2: "cross",
    3: "lead_hook",
    4: "rear_hook",
    5: "lead_uppercut",
    6: "rear_uppercut",
}

_LABEL_MAP: dict[tuple[bool, str], int] = {
    (True,  "straight"): 1,
    (False, "straight"): 2,
    (True,  "hook"):     3,
    (False, "hook"):     4,
    (True,  "uppercut"): 5,
    (False, "uppercut"): 6,
}

_WEIGHT_KEYS = tuple(f"w{i}" for i in range(1, 10))


class BoxingPunchClassifier:
    """
    Boxing punch classifier for MotionBERT H36M-17 3D pose.

    Usage::

        clf = BoxingPunchClassifier()
        label = clf.predict(poses)           # int 1–6
        name  = clf.predict_named(poses)     # e.g. "jab"

    Stage B can be run in two modes:
      • Rule-based (default): hard thresholds then weighted soft scores (w1–w9).
      • Learned: pass a sklearn-compatible classifier to ``stage_b_clf``.  When set,
        hard thresholds and weights are bypassed entirely; the model receives the
        9-element feature vector from ``_stage_b_features()``.

    Feature vector order (for learned Stage B):
        0  ρ              straightness ratio
        1  |Δy|/D         forward displacement
        2  Δθ_elbow/180   elbow extension change
        3  Δz/D           vertical displacement   (signed)
        4  1−ρ            curviness
        5  |n̂·x̂|         sagittal-plane alignment
        6  |Δx|/D         lateral displacement
        7  |n̂·ẑ|         horizontal-plane alignment
        8  θ_min/180      minimum elbow angle
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        stage_b_clf=None,
    ) -> None:
        self.w: dict[str, float] = {k: 1.0 for k in _WEIGHT_KEYS}
        if weights:
            unknown = set(weights) - set(self.w)
            if unknown:
                raise ValueError(f"Unknown weight keys: {unknown!r}. Valid: {_WEIGHT_KEYS}")
            self.w.update(weights)
        self._stage_b_clf = stage_b_clf  # sklearn-style clf or None

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, poses: np.ndarray) -> int:
        """
        Classify one punch window.

        Args:
            poses: (T, 17, 3) MotionBERT H36M 3D predictions for the punch.

        Returns:
            Integer label 1–6.
        """
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1] != 17 or poses.shape[2] != 3:
            raise ValueError(f"Expected (T, 17, 3), got {poses.shape}")

        q = self._preprocess(poses)
        is_lead, active_side = self._detect_active_hand(q)
        family = self._classify_family(q, active_side)
        return _LABEL_MAP[(is_lead, family)]

    def predict_named(self, poses: np.ndarray) -> str:
        """Like predict() but returns the string label, e.g. 'jab'."""
        return PUNCH_LABELS[self.predict(poses)]

    def predict_batch(self, sequences: list[np.ndarray]) -> list[int]:
        """Classify a list of punch windows; returns list of integer labels."""
        return [self.predict(s) for s in sequences]

    def __repr__(self) -> str:
        non_default = {k: v for k, v in self.w.items() if v != 1.0}
        wstr = f"weights={non_default}" if non_default else ""
        b2 = f", stage_b_clf={self._stage_b_clf!r}" if self._stage_b_clf is not None else ""
        return f"BoxingPunchClassifier({wstr}{b2})"

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, poses: np.ndarray) -> np.ndarray:
        q = self._to_body_frame(poses)
        q = self._scale_normalize(q)
        q = self._smooth(q)
        return q

    def _to_body_frame(self, poses: np.ndarray) -> np.ndarray:
        """
        Translate pelvis to origin, then rotate into the boxer's own frame:
            +x  to boxer's right (R shoulder → L shoulder axis)
            +z  up (pelvis → thorax axis)
            +y  forward (out of chest = ẑ × x̂)

        Rotation is built from the first frame (guard / pre-launch pose).
        """
        q = poses - poses[:, [_J["pelvis"]], :]  # pelvis to origin

        ref = q[0]
        x_raw = ref[_J["r_shoulder"]] - ref[_J["l_shoulder"]]
        x_norm = np.linalg.norm(x_raw)
        if x_norm < 1e-6:
            return q
        x_hat = x_raw / x_norm

        z_raw = ref[_J["thorax"]] - ref[_J["pelvis"]]
        z_norm = np.linalg.norm(z_raw)
        if z_norm < 1e-6:
            return q
        z_raw = z_raw / z_norm
        z_hat = z_raw - np.dot(z_raw, x_hat) * x_hat  # re-orthogonalise
        z_norm2 = np.linalg.norm(z_hat)
        if z_norm2 < 1e-6:
            return q
        z_hat = z_hat / z_norm2

        y_hat = np.cross(z_hat, x_hat)  # forward (+y = toward opponent)

        R = np.stack([x_hat, y_hat, z_hat], axis=0)  # (3, 3) row-stacked
        return q @ R.T  # (T, 17, 3) in body frame

    def _scale_normalize(self, q: np.ndarray) -> np.ndarray:
        torso = float(np.linalg.norm(q[0, _J["thorax"]] - q[0, _J["pelvis"]]))
        return q / torso if torso > 1e-6 else q

    def _smooth(self, q: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
        T = q.shape[0]
        w = min(window, T)
        if w % 2 == 0:
            w -= 1
        if w < polyorder + 1:
            return q
        return savgol_filter(q, window_length=w, polyorder=polyorder, axis=0)

    # ── Stage A — which hand? ─────────────────────────────────────────────────

    def _detect_active_hand(self, q: np.ndarray) -> tuple[bool, str]:
        """
        Returns (is_lead_punch, active_side).

        Active side: whichever wrist shows the larger peak-speed gain above its
        initial speed (filters out the guard hand drifting).

        Lead side: whichever body side is more forward (+y) in the body frame,
        determined by averaging shoulder and hip y-positions across the full window.
        Shoulders encode body rotation; hips are stable during punching.  Averaging
        across all frames makes this robust to windows that begin mid-punch or during
        casual standing, where a single frame or wrist position would be unreliable.
        """
        def _wrist_score(idx: int) -> float:
            speeds = np.linalg.norm(np.diff(q[:, idx], axis=0), axis=1)
            return float(speeds.max() - speeds[0]) if len(speeds) else 0.0

        score_l = _wrist_score(_J["l_wrist"])
        score_r = _wrist_score(_J["r_wrist"])
        active_side = "L" if score_l > score_r else "R"

        # Shoulder + hip average over the full window.
        # Hips don't move during punching; shoulders rotate but the average across
        # both joints and all frames suppresses punch-induced bias.
        l_y = float(q[:, [_J["l_shoulder"], _J["l_hip"]], 1].mean())
        r_y = float(q[:, [_J["r_shoulder"], _J["r_hip"]], 1].mean())
        lead_side = "L" if l_y > r_y else "R"

        return active_side == lead_side, active_side

    # ── Stage B — trajectory family ───────────────────────────────────────────

    def _stage_b_features(self, q: np.ndarray, side: str) -> np.ndarray:
        """Return the 9-element Stage-B feature vector (see class docstring)."""
        w_idx = _J["l_wrist"]    if side == "L" else _J["r_wrist"]
        e_idx = _J["l_elbow"]    if side == "L" else _J["r_elbow"]
        s_idx = _J["l_shoulder"] if side == "L" else _J["r_shoulder"]

        wrist    = q[:, w_idx, :]
        elbow    = q[:, e_idx, :]
        shoulder = q[:, s_idx, :]

        delta = wrist[-1] - wrist[0]
        D = max(float(np.linalg.norm(delta)), 1e-6)
        L = max(float(np.sum(np.linalg.norm(np.diff(wrist, axis=0), axis=1))), 1e-6)
        rho = D / L

        dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
        theta_min, d_theta = self._elbow_angle_features(shoulder, elbow, wrist)
        n_hat = self._pca_normal(wrist)

        return np.array([
            rho,
            abs(dy) / D,
            d_theta / 180.0,
            dz / D,
            1.0 - rho,
            abs(float(n_hat @ np.array([1.0, 0.0, 0.0]))),
            abs(dx) / D,
            abs(float(n_hat @ np.array([0.0, 0.0, 1.0]))),
            theta_min / 180.0,
        ], dtype=np.float64)

    def _classify_family(self, q: np.ndarray, side: str) -> str:
        """Returns 'straight', 'hook', or 'uppercut'."""
        if self._stage_b_clf is not None:
            feats = self._stage_b_features(q, side)
            idx = int(self._stage_b_clf.predict(feats[None])[0])
            return ["straight", "uppercut", "hook"][idx]

        w_idx = _J["l_wrist"]    if side == "L" else _J["r_wrist"]
        e_idx = _J["l_elbow"]   if side == "L" else _J["r_elbow"]
        s_idx = _J["l_shoulder"] if side == "L" else _J["r_shoulder"]

        wrist    = q[:, w_idx, :]
        elbow    = q[:, e_idx, :]
        shoulder = q[:, s_idx, :]

        delta = wrist[-1] - wrist[0]
        D = max(float(np.linalg.norm(delta)), 1e-6)
        L = max(float(np.sum(np.linalg.norm(np.diff(wrist, axis=0), axis=1))), 1e-6)
        rho = D / L

        dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
        theta_min, d_theta = self._elbow_angle_features(shoulder, elbow, wrist)
        n_hat = self._pca_normal(wrist)

        x_hat = np.array([1.0, 0.0, 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        # Hard thresholds
        if rho > 0.85 and abs(dy) / D > 0.7 and d_theta > 70.0:
            return "straight"
        if dz / D > 0.5 and theta_min < 110.0 and abs(float(n_hat @ x_hat)) > 0.7:
            return "uppercut"
        if abs(dx) / D > 0.5 and theta_min < 110.0 and abs(float(n_hat @ z_hat)) > 0.7:
            return "hook"

        # Soft scoring fallback
        w = self.w
        score_straight = (
            w["w1"] * rho
            + w["w2"] * (abs(dy) / D)
            + w["w3"] * (d_theta / 180.0)
        )
        score_uppercut = (
            w["w4"] * (dz / D)
            + w["w5"] * (1.0 - rho)
            + w["w6"] * abs(float(n_hat @ x_hat))
        )
        score_hook = (
            w["w7"] * (abs(dx) / D)
            + w["w8"] * (1.0 - rho)
            + w["w9"] * abs(float(n_hat @ z_hat))
        )

        return ["straight", "uppercut", "hook"][int(np.argmax([score_straight, score_uppercut, score_hook]))]

    def _elbow_angle_features(
        self,
        shoulder: np.ndarray,
        elbow: np.ndarray,
        wrist: np.ndarray,
    ) -> tuple[float, float]:
        """
        Elbow flexion angle θ = ∠(shoulder, elbow, wrist) at each frame.
        Returns (theta_min, theta_max - theta_min) in degrees.
        ~180° = arm extended (jab/cross), ~90° = bent (hook/uppercut).
        """
        u = shoulder - elbow  # upper-arm vectors (T, 3)
        v = wrist - elbow     # forearm vectors   (T, 3)
        cross_norms = np.linalg.norm(np.cross(u, v), axis=1)
        dots = np.einsum("ti,ti->t", u, v)
        angles = np.degrees(np.arctan2(cross_norms, dots))
        return float(angles.min()), float(angles.max() - angles.min())

    def _pca_normal(self, wrist: np.ndarray) -> np.ndarray:
        """
        Third PCA eigenvector of the wrist trajectory = normal to its plane of motion.
        Normal ≈ ±ẑ  →  horizontal-plane motion  →  hook
        Normal ≈ ±x̂  →  sagittal-plane motion    →  uppercut
        """
        if wrist.shape[0] < 3:
            return np.array([0.0, 0.0, 1.0])
        centered = wrist - wrist.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
        return eigenvectors[:, 0]  # eigenvector of smallest eigenvalue = normal
