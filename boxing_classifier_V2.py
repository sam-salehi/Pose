"""
Rule-based boxing punch classifier for MotionBERT H36M-17 3D pose,
with an integrated punch/no-punch gate.

Input: (T, 17, 3) numpy array (one candidate window from the segmenter).
Output: integer label 0–6:
    0 not_a_punch
    1 jab     2 cross    3 lead hook
    4 rear hook   5 lead uppercut    6 rear uppercut

Pipeline:
    1. Preprocess (body-frame transform, scale-normalise, smooth)
    2. Stage A — detect active hand (which wrist moved most, with forward intent)
    3. Punch gate — verify the window actually contains a punch
    4. Stage B — classify trajectory family (straight / hook / uppercut)
    5. Compose final label from (is_lead, family)

Stage B uses hard thresholds first; ties fall to a soft weighted sum.
The nine classifier weights w1–w9 default to 1.0 and can be tuned on a labelled set.
The five gate weights g1–g5 control the punch/no-punch decision.
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
    0: "not_a_punch",
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

_CLASSIFIER_WEIGHT_KEYS = tuple(f"w{i}" for i in range(1, 10))
_GATE_WEIGHT_KEYS = tuple(f"g{i}" for i in range(1, 6))


class PunchGateResult:
    """Diagnostic output from the punch/no-punch gate."""

    __slots__ = (
        "is_punch", "confidence", "forward_intent", "displacement",
        "speed_prominence", "elbow_engagement", "peak_centrality",
    )

    def __init__(
        self,
        is_punch: bool,
        confidence: float,
        forward_intent: float,
        displacement: float,
        speed_prominence: float,
        elbow_engagement: float,
        peak_centrality: float,
    ) -> None:
        self.is_punch = is_punch
        self.confidence = confidence
        self.forward_intent = forward_intent
        self.displacement = displacement
        self.speed_prominence = speed_prominence
        self.elbow_engagement = elbow_engagement
        self.peak_centrality = peak_centrality

    def __repr__(self) -> str:
        return (
            f"PunchGateResult(is_punch={self.is_punch}, "
            f"confidence={self.confidence:.3f}, "
            f"forward_intent={self.forward_intent:.3f}, "
            f"displacement={self.displacement:.3f}, "
            f"speed_prominence={self.speed_prominence:.3f}, "
            f"elbow_engagement={self.elbow_engagement:.3f}, "
            f"peak_centrality={self.peak_centrality:.3f})"
        )


class PredictionResult:
    """Full diagnostic output from predict_detailed()."""

    __slots__ = ("label", "label_name", "is_lead", "active_side", "family", "gate")

    def __init__(
        self,
        label: int,
        label_name: str,
        is_lead: bool | None,
        active_side: str | None,
        family: str | None,
        gate: PunchGateResult,
    ) -> None:
        self.label = label
        self.label_name = label_name
        self.is_lead = is_lead
        self.active_side = active_side
        self.family = family
        self.gate = gate

    def __repr__(self) -> str:
        return (
            f"PredictionResult(label={self.label} ({self.label_name}), "
            f"is_lead={self.is_lead}, active_side={self.active_side!r}, "
            f"family={self.family!r}, gate={self.gate})"
        )


class BoxingPunchClassifier:
    """
    Rule-based boxing punch classifier for MotionBERT H36M-17 3D pose,
    with integrated punch/no-punch gate.

    Usage::

        clf = BoxingPunchClassifier()
        label = clf.predict(poses)               # int 0–6 (0 = not a punch)
        name  = clf.predict_named(poses)         # e.g. "jab" or "not_a_punch"
        info  = clf.predict_detailed(poses)      # full diagnostics

    Classifier weights w1–w9 control Stage B soft scoring (see classify):
        w1–w3  straight score:  ρ (straightness), Δy/D (forward), Δθ_elbow/180°
        w4–w6  uppercut score:  Δz/D (vertical), 1−ρ (curviness), |n̂·x̂| (sagittal)
        w7–w9  hook score:      |Δx|/D (lateral), 1−ρ (curviness), |n̂·ẑ| (horizontal)

    Gate weights g1–g5 control punch/no-punch confidence:
        g1  forward intent (Δy/D, signed; only positive contributes)
        g2  total displacement (in torso units)
        g3  speed peak prominence above mean
        g4  elbow angle range (extension or flexion)
        g5  peak centrality (peak should be mid-window, not at edges)

    Gate threshold is the confidence above which the window is accepted as a punch.
    """

    DEFAULT_GATE_WEIGHTS = {"g1": 0.25, "g2": 0.20, "g3": 0.20, "g4": 0.20, "g5": 0.15}
    DEFAULT_GATE_THRESHOLD = 0.5

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        gate_weights: dict[str, float] | None = None,
        gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    ) -> None:
        self.w: dict[str, float] = {k: 1.0 for k in _CLASSIFIER_WEIGHT_KEYS}
        if weights:
            unknown = set(weights) - set(self.w)
            if unknown:
                raise ValueError(
                    f"Unknown classifier weight keys: {unknown!r}. "
                    f"Valid: {_CLASSIFIER_WEIGHT_KEYS}"
                )
            self.w.update(weights)

        self.g: dict[str, float] = dict(self.DEFAULT_GATE_WEIGHTS)
        if gate_weights:
            unknown = set(gate_weights) - set(self.g)
            if unknown:
                raise ValueError(
                    f"Unknown gate weight keys: {unknown!r}. "
                    f"Valid: {_GATE_WEIGHT_KEYS}"
                )
            self.g.update(gate_weights)

        self.gate_threshold = float(gate_threshold)

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, poses: np.ndarray) -> int:
        """
        Classify one window. Returns 0 if the window is not a punch,
        otherwise an integer label 1–6.
        """
        return self.predict_detailed(poses).label

    def predict_named(self, poses: np.ndarray) -> str:
        """Like predict() but returns the string label."""
        return PUNCH_LABELS[self.predict(poses)]

    def predict_batch(self, sequences: list[np.ndarray]) -> list[int]:
        """Classify a list of windows; returns list of integer labels (0–6)."""
        return [self.predict(s) for s in sequences]

    def predict_detailed(self, poses: np.ndarray) -> PredictionResult:
        """
        Classify with full diagnostic output. Useful for debugging,
        threshold tuning, and downstream confidence-aware processing.
        """
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1] != 17 or poses.shape[2] != 3:
            raise ValueError(f"Expected (T, 17, 3), got {poses.shape}")

        q = self._preprocess(poses)
        is_lead, active_side = self._detect_active_hand(q)
        gate = self._punch_gate(q, active_side)

        if not gate.is_punch:
            return PredictionResult(
                label=0,
                label_name=PUNCH_LABELS[0],
                is_lead=None,
                active_side=active_side,
                family=None,
                gate=gate,
            )

        family = self._classify_family(q, active_side)
        label = _LABEL_MAP[(is_lead, family)]
        return PredictionResult(
            label=label,
            label_name=PUNCH_LABELS[label],
            is_lead=is_lead,
            active_side=active_side,
            family=family,
            gate=gate,
        )

    def gate_only(self, poses: np.ndarray) -> PunchGateResult:
        """Run only the punch/no-punch gate. Useful for tuning thresholds."""
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1] != 17 or poses.shape[2] != 3:
            raise ValueError(f"Expected (T, 17, 3), got {poses.shape}")
        q = self._preprocess(poses)
        _, active_side = self._detect_active_hand(q)
        return self._punch_gate(q, active_side)

    def __repr__(self) -> str:
        non_default_w = {k: v for k, v in self.w.items() if v != 1.0}
        non_default_g = {k: v for k, v in self.g.items() if v != self.DEFAULT_GATE_WEIGHTS[k]}
        parts = []
        if non_default_w:
            parts.append(f"weights={non_default_w}")
        if non_default_g:
            parts.append(f"gate_weights={non_default_g}")
        if self.gate_threshold != self.DEFAULT_GATE_THRESHOLD:
            parts.append(f"gate_threshold={self.gate_threshold}")
        return f"BoxingPunchClassifier({', '.join(parts)})"

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
        """
        q = poses - poses[:, [_J["pelvis"]], :]

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
        z_hat = z_raw - np.dot(z_raw, x_hat) * x_hat
        z_norm2 = np.linalg.norm(z_hat)
        if z_norm2 < 1e-6:
            return q
        z_hat = z_hat / z_norm2

        y_hat = np.cross(z_hat, x_hat)

        R = np.stack([x_hat, y_hat, z_hat], axis=0)
        return q @ R.T

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

        Active side: the wrist whose peak speed AND forward displacement
        product is largest. Forward-displacement multiplier kills the
        'fast retract' failure mode in tight combos.

        Lead side: the foot more forward (+y) in the body frame.
        """
        def _wrist_score(idx: int) -> float:
            wrist = q[:, idx, :]
            speeds = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
            peak_speed = float(speeds.max()) if len(speeds) else 0.0
            forward = float(wrist[:, 1].max() - wrist[0, 1])
            return peak_speed * max(0.0, forward)

        score_l = _wrist_score(_J["l_wrist"])
        score_r = _wrist_score(_J["r_wrist"])
        active_side = "L" if score_l > score_r else "R"

        l_y = float(q[0, _J["l_ankle"], 1])
        r_y = float(q[0, _J["r_ankle"], 1])
        lead_side = "L" if l_y > r_y else "R"

        return active_side == lead_side, active_side

    # ── Punch gate ────────────────────────────────────────────────────────────

    def _punch_gate(self, q: np.ndarray, side: str) -> PunchGateResult:
        """
        Verify the window actually contains a punch.

        Five kinematic checks combine into a confidence score:
            1. Forward intent       — wrist moves toward opponent (+y)
            2. Displacement         — wrist travels enough total distance
            3. Speed prominence     — speed peak rises clearly above baseline
            4. Elbow engagement     — elbow flexes or extends meaningfully
            5. Peak centrality      — speed peak is mid-window, not at the edges

        Each check is scaled into [0, 1] then weighted by g1–g5.
        Window is accepted as a punch iff confidence > gate_threshold.
        """
        w_idx = _J["l_wrist"] if side == "L" else _J["r_wrist"]
        e_idx = _J["l_elbow"] if side == "L" else _J["r_elbow"]
        s_idx = _J["l_shoulder"] if side == "L" else _J["r_shoulder"]

        wrist = q[:, w_idx, :]
        elbow = q[:, e_idx, :]
        shoulder = q[:, s_idx, :]

        # Speeds (used by features 1, 2, 3, 5)
        speeds = np.linalg.norm(np.diff(wrist, axis=0), axis=1)

        # Locate the moment of maximum extension. Using wrist[-1] - wrist[0]
        # would give zero displacement for any fully-retracted punch — wrong.
        # We use the frame of peak speed (≈ extension phase) as the reference.
        if len(speeds) > 0:
            peak_idx = int(np.argmax(speeds)) + 1  # +1: speed[i] = motion at frame i+1
            peak_idx = min(peak_idx, len(wrist) - 1)
        else:
            peak_idx = len(wrist) - 1

        # 1. Forward intent — Δy / D from start to peak extension
        delta = wrist[peak_idx] - wrist[0]
        D = float(np.linalg.norm(delta))
        if D < 1e-6:
            forward_intent = 0.0
        else:
            forward_intent = max(0.0, float(delta[1]) / D)

        # 2. Total displacement (in torso units, already scale-normalised)
        # Real punches travel ~0.6-1.2 torso units to peak. Saturates at 0.8.
        displacement = min(1.0, D / 0.8)

        # 3. Speed peak prominence above mean
        if len(speeds) == 0:
            speed_prominence = 0.0
        else:
            peak_speed = float(speeds.max())
            mean_speed = float(speeds.mean())
            # Saturates at prominence of 0.3 torso units / frame
            speed_prominence = min(1.0, max(0.0, (peak_speed - mean_speed) / 0.3))

        # 4. Elbow engagement — range of θ_elbow over window
        _, d_theta = self._elbow_angle_features(shoulder, elbow, wrist)
        # Saturates at 90° range. Real punches have 60-150° range.
        elbow_engagement = min(1.0, d_theta / 90.0)

        # 5. Peak centrality — peak should be mid-window, not at edges
        if len(speeds) >= 5:
            peak_idx = int(np.argmax(speeds))
            half = len(speeds) / 2.0
            # 1.0 if peak is exactly in the middle, 0.0 if at either edge
            peak_centrality = 1.0 - abs(peak_idx - half) / half
        else:
            peak_centrality = 0.0

        # Combine into confidence
        g = self.g
        confidence = (
            g["g1"] * forward_intent
            + g["g2"] * displacement
            + g["g3"] * speed_prominence
            + g["g4"] * elbow_engagement
            + g["g5"] * peak_centrality
        )
        # Normalise by sum of weights so threshold is interpretable on [0, 1]
        weight_sum = sum(g.values())
        if weight_sum > 1e-6:
            confidence = confidence / weight_sum

        is_punch = confidence > self.gate_threshold

        return PunchGateResult(
            is_punch=is_punch,
            confidence=float(confidence),
            forward_intent=float(forward_intent),
            displacement=float(displacement),
            speed_prominence=float(speed_prominence),
            elbow_engagement=float(elbow_engagement),
            peak_centrality=float(peak_centrality),
        )

    # ── Stage B — trajectory family ───────────────────────────────────────────

    def _classify_family(self, q: np.ndarray, side: str) -> str:
        """Returns 'straight', 'hook', or 'uppercut'."""
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

        return ["straight", "uppercut", "hook"][
            int(np.argmax([score_straight, score_uppercut, score_hook]))
        ]

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
        u = shoulder - elbow
        v = wrist - elbow
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
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        return eigenvectors[:, 0]