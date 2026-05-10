#!/usr/bin/env python3
"""
Optimize the 9 soft-score weights in BoxingPunchClassifier.

Strategy
--------
Extract Stage-B features for every annotated punch window, then minimise
cross-entropy over softmax([score_straight, score_uppercut, score_hook])
with scipy L-BFGS-B (weights ≥ 0).

Because the hard thresholds always fire first, we track which branch each punch
takes and report coverage + accuracy per branch.

Evaluation uses leave-one-video-out cross-validation (4 folds) so that each
video is held out exactly once.  Final weights are fit on all data and saved to
optimized_weights.json.

Usage
-----
    python optimize_weights.py
    python optimize_weights.py --no-cv    # skip CV, fit+save on all data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import joblib
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from punch_classifier import BoxingPunchClassifier, PUNCH_LABELS, _LABEL_MAP
from eval_punch_classifier import _load_annotations

_REPO         = Path(__file__).resolve().parent
_MOTIONBERT   = _REPO / "Dataset" / "MotionBERT_3d"
_ANNOT        = _REPO / "Dataset" / "Annotation_files"
_WEIGHTS_OUT  = _REPO / "optimized_weights.json"
_STAGE_B_OUT  = _REPO / "stage_b_clf.pkl"

_J = {
    "pelvis": 0, "r_hip": 1, "r_knee": 2, "r_ankle": 3,
    "l_hip": 4, "l_knee": 5, "l_ankle": 6,
    "spine": 7, "thorax": 8, "neck": 9, "head": 10,
    "l_shoulder": 11, "l_elbow": 12, "l_wrist": 13,
    "r_shoulder": 14, "r_elbow": 15, "r_wrist": 16,
}

# family label → class index used by the scorer
_FAMILY_IDX = {"straight": 0, "uppercut": 1, "hook": 2}
_PUNCH_FAMILY = {
    "jab":           "straight",
    "cross":         "straight",
    "lead_hook":     "hook",
    "rear_hook":     "hook",
    "lead_uppercut": "uppercut",
    "rear_uppercut": "uppercut",
}
# lead flag ground truth
_PUNCH_IS_LEAD = {
    "jab":           True,
    "cross":         False,
    "lead_hook":     True,
    "rear_hook":     False,
    "lead_uppercut": True,
    "rear_uppercut": False,
}


# ── Feature record ─────────────────────────────────────────────────────────────

class PunchRecord(NamedTuple):
    ver:         str
    start:       int
    end:         int
    gt_label:    str        # e.g. "jab"
    gt_family:   int        # 0=straight 1=uppercut 2=hook
    gt_is_lead:  bool
    pred_is_lead: bool      # Stage-A prediction
    features:    np.ndarray # shape (9,) — the w1..w9 multipliers
    hard_branch: str | None # "straight"/"uppercut"/"hook" or None (=soft fallback)
    theta_min:   float


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract(clf: BoxingPunchClassifier, poses: np.ndarray, side: str
             ) -> tuple[np.ndarray, str | None, float]:
    """
    Return (features_9, hard_branch, theta_min).
    Replicates _classify_family logic without the weight-gated soft score.
    """
    w_idx = _J["l_wrist"]    if side == "L" else _J["r_wrist"]
    e_idx = _J["l_elbow"]    if side == "L" else _J["r_elbow"]
    s_idx = _J["l_shoulder"] if side == "L" else _J["r_shoulder"]

    wrist    = poses[:, w_idx, :]
    elbow    = poses[:, e_idx, :]
    shoulder = poses[:, s_idx, :]

    delta = wrist[-1] - wrist[0]
    D = max(float(np.linalg.norm(delta)), 1e-6)
    L = max(float(np.sum(np.linalg.norm(np.diff(wrist, axis=0), axis=1))), 1e-6)
    rho = D / L

    dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
    theta_min, d_theta = clf._elbow_angle_features(shoulder, elbow, wrist)
    n_hat = clf._pca_normal(wrist)

    x_hat = np.array([1.0, 0.0, 0.0])
    z_hat = np.array([0.0, 0.0, 1.0])

    # features matching w1..w9 one-to-one
    feats = np.array([
        rho,                               # w1 → score_straight
        abs(dy) / D,                       # w2
        d_theta / 180.0,                   # w3
        dz / D,                            # w4 → score_uppercut  (can be < 0)
        1.0 - rho,                         # w5
        abs(float(n_hat @ x_hat)),         # w6
        abs(dx) / D,                       # w7 → score_hook
        1.0 - rho,                         # w8
        abs(float(n_hat @ z_hat)),         # w9
    ], dtype=np.float64)

    # Mirror the hard-threshold logic
    hard: str | None = None
    if rho > 0.85 and abs(dy) / D > 0.7 and d_theta > 70.0:
        hard = "straight"
    elif dz / D > 0.5 and theta_min < 110.0 and abs(float(n_hat @ x_hat)) > 0.7:
        hard = "uppercut"
    elif abs(dx) / D > 0.5 and theta_min < 110.0 and abs(float(n_hat @ z_hat)) > 0.7:
        hard = "hook"

    return feats, hard, theta_min


def _build_records(ver: str, clf: BoxingPunchClassifier) -> list[PunchRecord]:
    npy_path  = _MOTIONBERT / ver / "X3D.npy"
    xlsx_path = _ANNOT / f"{ver}.xlsx"
    if not npy_path.is_file() or not xlsx_path.is_file():
        return []

    poses      = np.load(str(npy_path))
    annotations = _load_annotations(xlsx_path)
    records: list[PunchRecord] = []

    for start, end, gt_label in annotations:
        window = poses[start : end + 1]
        if window.shape[0] < 3:
            window = poses[max(0, start - 3) : end + 4]
        if window.shape[0] < 3:
            continue
        try:
            q = clf._preprocess(window)
            pred_is_lead, active_side = clf._detect_active_hand(q)
            feats, hard, theta_min = _extract(clf, q, active_side)
        except Exception:
            continue

        records.append(PunchRecord(
            ver          = ver,
            start        = start,
            end          = end,
            gt_label     = gt_label,
            gt_family    = _FAMILY_IDX[_PUNCH_FAMILY[gt_label]],
            gt_is_lead   = _PUNCH_IS_LEAD[gt_label],
            pred_is_lead = pred_is_lead,
            features     = feats,
            hard_branch  = hard,
            theta_min    = theta_min,
        ))

    return records


# ── Loss + gradient ────────────────────────────────────────────────────────────

def _scores(w: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    w: (9,)   F: (N, 9)   → scores (N, 3)  [straight, uppercut, hook]
    """
    return np.column_stack([
        F[:, 0:3] @ w[0:3],
        F[:, 3:6] @ w[3:6],
        F[:, 6:9] @ w[6:9],
    ])


def _loss_and_grad(w: np.ndarray, F: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Cross-entropy loss + analytical gradient.
    y: (N,) integer class indices 0/1/2
    """
    N = len(y)
    S = _scores(w, F)                       # (N, 3)
    lsm = log_softmax(S, axis=1)            # (N, 3)
    loss = -lsm[np.arange(N), y].mean()

    # gradient of CE w.r.t. scores: p - one_hot(y)
    p = softmax(S, axis=1)                  # (N, 3)
    p[np.arange(N), y] -= 1.0
    p /= N                                  # (N, 3)  dL/dS

    # chain rule back to w
    # score_straight[:,0] = F[:,0]*w[0] + F[:,1]*w[1] + F[:,2]*w[2]
    # score_uppercut[:,1] = F[:,3]*w[3] + F[:,4]*w[4] + F[:,5]*w[5]
    # score_hook    [:,2] = F[:,6]*w[6] + F[:,7]*w[7] + F[:,8]*w[8]
    grad = np.concatenate([
        F[:, 0:3].T @ p[:, 0],   # dL/dw[0:3]
        F[:, 3:6].T @ p[:, 1],   # dL/dw[3:6]
        F[:, 6:9].T @ p[:, 2],   # dL/dw[6:9]
    ])
    return float(loss), grad


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _predict_family(w: np.ndarray, rec: PunchRecord) -> str:
    """Simulate the full Stage-B decision with given weights."""
    if rec.hard_branch is not None:
        return rec.hard_branch
    S = _scores(w, rec.features[None])  # (1, 3)
    return ["straight", "uppercut", "hook"][int(S[0].argmax())]


def _predict_full_label(w: np.ndarray, rec: PunchRecord) -> str:
    family = _predict_family(w, rec)
    is_lead = rec.pred_is_lead
    from punch_classifier import _LABEL_MAP
    return PUNCH_LABELS[_LABEL_MAP[(is_lead, family)]]


def _accuracy(w: np.ndarray, records: list[PunchRecord]) -> dict:
    """Returns overall + per-class accuracy dict."""
    correct_full = correct_family = correct_lead = 0
    per_class: dict[str, list[bool]] = {}
    for rec in records:
        fam_pred   = _predict_family(w, rec)
        label_pred = _predict_full_label(w, rec)
        correct_family += (fam_pred == ["straight", "uppercut", "hook"][rec.gt_family])
        correct_full   += (label_pred == rec.gt_label)
        correct_lead   += (rec.pred_is_lead == rec.gt_is_lead)
        per_class.setdefault(rec.gt_label, []).append(label_pred == rec.gt_label)

    N = len(records)
    return {
        "overall":   correct_full   / N,
        "family":    correct_family / N,
        "stage_a":   correct_lead   / N,
        "per_class": {k: sum(v) / len(v) for k, v in per_class.items()},
        "n":         N,
    }


# ── Branch coverage report ─────────────────────────────────────────────────────

def _branch_report(records: list[PunchRecord]) -> None:
    branches = {"straight": 0, "uppercut": 0, "hook": 0, None: 0}
    correct_hard = total_hard = 0
    for rec in records:
        branches[rec.hard_branch] = branches.get(rec.hard_branch, 0) + 1
        if rec.hard_branch is not None:
            total_hard += 1
            gt_fam = ["straight", "uppercut", "hook"][rec.gt_family]
            correct_hard += (rec.hard_branch == gt_fam)

    N = len(records)
    soft = branches[None]
    hard = N - soft
    print(f"\n  Branch coverage ({N} punches total)")
    print(f"  {'hard straight':20} {branches['straight']:5}  ({100*branches['straight']/N:.1f}%)")
    print(f"  {'hard uppercut':20} {branches['uppercut']:5}  ({100*branches['uppercut']/N:.1f}%)")
    print(f"  {'hard hook':20} {branches['hook']:5}  ({100*branches['hook']/N:.1f}%)")
    print(f"  {'soft fallback':20} {soft:5}  ({100*soft/N:.1f}%)")
    if total_hard:
        print(f"  hard-branch family acc:  {correct_hard}/{total_hard} = {100*correct_hard/total_hard:.1f}%")


# ── Optimize ───────────────────────────────────────────────────────────────────

def _fit(records: list[PunchRecord], w0: np.ndarray | None = None,
         max_iter: int = 500) -> np.ndarray:
    F = np.stack([r.features for r in records])   # (N, 9)
    y = np.array([r.gt_family for r in records])  # (N,)

    if w0 is None:
        w0 = np.ones(9)

    res = minimize(
        _loss_and_grad,
        x0     = w0,
        args   = (F, y),
        method = "L-BFGS-B",
        jac    = True,
        bounds = [(0.0, None)] * 9,   # weights ≥ 0
        options = {"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-7},
    )
    return res.x


# ── CV + reporting ─────────────────────────────────────────────────────────────

def _print_accuracy(label: str, acc: dict) -> None:
    print(f"\n  {label}")
    print(f"    full-label acc : {acc['overall']*100:5.1f}%  ({int(acc['overall']*acc['n'])}/{acc['n']})")
    print(f"    family acc     : {acc['family']*100:5.1f}%")
    print(f"    Stage-A acc    : {acc['stage_a']*100:5.1f}%  (lead/rear — unaffected by weights)")
    classes = sorted(acc["per_class"])
    for cls in classes:
        print(f"      {cls:<20} {acc['per_class'][cls]*100:5.1f}%")


def _leave_one_out_cv(all_records: list[PunchRecord]) -> None:
    vers = sorted({r.ver for r in all_records})
    if len(vers) < 2:
        print("  Only one video — skipping CV.")
        return

    print(f"\n{'='*60}")
    print(f"  Leave-one-video-out CV  ({len(vers)} folds)")
    print(f"{'='*60}")

    w_default = np.ones(9)
    cv_before = cv_after = 0
    cv_n = 0

    for val_ver in vers:
        train = [r for r in all_records if r.ver != val_ver]
        val   = [r for r in all_records if r.ver == val_ver]
        if not train or not val:
            continue

        w_opt = _fit(train)
        acc_before = _accuracy(w_default, val)
        acc_after  = _accuracy(w_opt,     val)

        cv_before += acc_before["overall"] * len(val)
        cv_after  += acc_after["overall"]  * len(val)
        cv_n      += len(val)

        print(f"\n  Fold val={val_ver}  (train {len(train)}, val {len(val)})")
        print(f"    before: {acc_before['overall']*100:.1f}%   after: {acc_after['overall']*100:.1f}%"
              f"   family before: {acc_before['family']*100:.1f}%   after: {acc_after['family']*100:.1f}%")

    print(f"\n  CV summary — weighted mean full-label accuracy")
    print(f"    default weights : {100*cv_before/cv_n:.1f}%")
    print(f"    optimized       : {100*cv_after/cv_n:.1f}%")


# ── Stage-B learned classifier ────────────────────────────────────────────────

def _to_stage_b_feats(rec: PunchRecord) -> np.ndarray:
    """Map a PunchRecord to the 9-element feature vector used by _stage_b_features()."""
    f = rec.features
    # f[7] is a duplicate of f[4] (both = 1-rho); replace with |n_hat@z_hat| = f[8]
    # and append theta_min as the 9th feature.
    return np.array([
        f[0],                    # rho
        f[1],                    # |dy|/D
        f[2],                    # d_theta/180
        f[3],                    # dz/D
        f[4],                    # 1-rho
        f[5],                    # |n_hat@x_hat|
        f[6],                    # |dx|/D
        f[8],                    # |n_hat@z_hat|
        rec.theta_min / 180.0,   # theta_min
    ], dtype=np.float64)


def _accuracy_clf(stage_b, records: list[PunchRecord]) -> dict:
    """Evaluate using a fitted sklearn Stage-B pipeline."""
    F = np.stack([_to_stage_b_feats(r) for r in records])
    preds = stage_b.predict(F)

    correct_full = correct_family = correct_lead = 0
    per_class: dict[str, list[bool]] = {}
    families = ["straight", "uppercut", "hook"]

    for rec, fam_idx in zip(records, preds):
        family     = families[int(fam_idx)]
        label_pred = PUNCH_LABELS[_LABEL_MAP[(rec.pred_is_lead, family)]]
        correct_family += (family == families[rec.gt_family])
        correct_full   += (label_pred == rec.gt_label)
        correct_lead   += (rec.pred_is_lead == rec.gt_is_lead)
        per_class.setdefault(rec.gt_label, []).append(label_pred == rec.gt_label)

    N = len(records)
    return {
        "overall":   correct_full   / N,
        "family":    correct_family / N,
        "stage_a":   correct_lead   / N,
        "per_class": {k: sum(v) / len(v) for k, v in per_class.items()},
        "n":         N,
    }


def _train_stage_b(records: list[PunchRecord]):
    """
    Train LogisticRegression and RBF-SVM via 3-fold GridSearchCV on the training
    records, return the better-scoring fitted pipeline.
    """
    F = np.stack([_to_stage_b_feats(r) for r in records])
    y = np.array([r.gt_family for r in records])

    lr_pipe = Pipeline([("sc", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs",
                                                   class_weight="balanced"))])
    svm_pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", SVC(kernel="rbf", class_weight="balanced"))])

    gs_lr  = GridSearchCV(lr_pipe,  {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
                          cv=3, scoring="accuracy")
    gs_svm = GridSearchCV(svm_pipe, {"clf__C": [0.1, 1.0, 10.0, 100.0],
                                     "clf__gamma": ["scale", "auto"]},
                          cv=3, scoring="accuracy")

    gs_lr.fit(F, y)
    gs_svm.fit(F, y)

    if gs_svm.best_score_ > gs_lr.best_score_:
        print(f"    SVM  wins: CV family acc = {gs_svm.best_score_*100:.1f}%"
              f"  params={gs_svm.best_params_}")
        return gs_svm.best_estimator_, "SVM"
    else:
        print(f"    LR   wins: CV family acc = {gs_lr.best_score_*100:.1f}%"
              f"  params={gs_lr.best_params_}")
        return gs_lr.best_estimator_, "LR"


def _loocv_stage_b(all_records: list[PunchRecord]) -> None:
    vers = sorted({r.ver for r in all_records})
    if len(vers) < 2:
        print("  Only one video — skipping CV.")
        return

    print(f"\n{'='*60}")
    print(f"  Stage-B learned clf  —  Leave-one-video-out CV")
    print(f"{'='*60}")

    cv_family = cv_full = cv_n = 0

    for val_ver in vers:
        train = [r for r in all_records if r.ver != val_ver]
        val   = [r for r in all_records if r.ver == val_ver]
        clf_b, kind = _train_stage_b(train)
        acc = _accuracy_clf(clf_b, val)
        cv_family += acc["family"]  * len(val)
        cv_full   += acc["overall"] * len(val)
        cv_n      += len(val)
        print(f"  Fold val={val_ver} ({kind}): "
              f"family={acc['family']*100:.1f}%  full={acc['overall']*100:.1f}%")

    print(f"\n  CV mean  family acc : {100*cv_family/cv_n:.1f}%")
    print(f"  CV mean  full  acc  : {100*cv_full/cv_n:.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optimise BoxingPunchClassifier weights via scipy L-BFGS-B."
    )
    ap.add_argument("--no-cv", action="store_true",
                    help="Skip leave-one-out CV; fit+save on all data only.")
    ap.add_argument("--max-iter", type=int, default=500,
                    help="L-BFGS-B max iterations (default 500).")
    args = ap.parse_args()

    clf = BoxingPunchClassifier()

    # ── Load all records ───────────────────────────────────────────────────────
    vers = sorted(
        d.name for d in _MOTIONBERT.iterdir()
        if d.is_dir() and (d / "X3D.npy").is_file()
        and (_ANNOT / f"{d.name}.xlsx").is_file()
    )
    if not vers:
        sys.exit("No videos found with both X3D.npy and annotation xlsx.")

    print(f"Loading features for: {vers}")
    all_records: list[PunchRecord] = []
    for ver in vers:
        recs = _build_records(ver, clf)
        print(f"  {ver}: {len(recs)} punches")
        all_records.extend(recs)

    print(f"\nTotal: {len(all_records)} punches across {len(vers)} videos")

    # ── Branch coverage ────────────────────────────────────────────────────────
    _branch_report(all_records)

    # ── Baseline with default weights ─────────────────────────────────────────
    w_default = np.ones(9)
    acc_default = _accuracy(w_default, all_records)
    _print_accuracy("Baseline (all data, w=1)", acc_default)

    # ── Leave-one-video-out CV (weights) ──────────────────────────────────────
    if not args.no_cv:
        _leave_one_out_cv(all_records)

    # ── Fit weights on all data ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Fitting soft-score weights on all data …")
    w_opt = _fit(all_records, max_iter=args.max_iter)
    acc_opt = _accuracy(w_opt, all_records)

    print(f"{'='*60}")
    _print_accuracy("Optimized weights (all data, in-sample)", acc_opt)

    keys = [f"w{i}" for i in range(1, 10)]
    descriptions = [
        "straightness ρ", "forward disp |Δy|/D", "elbow extension Δθ/180",
        "vertical disp Δz/D", "curviness 1−ρ (uppercut)", "sagittal plane |n̂·x̂|",
        "lateral disp |Δx|/D", "curviness 1−ρ (hook)", "horizontal plane |n̂·ẑ|",
    ]
    weights_dict: dict[str, float] = {}
    print(f"\n  Optimized weights:")
    for k, d, v in zip(keys, descriptions, w_opt):
        weights_dict[k] = float(v)
        print(f"    {k} = {v:7.4f}  ({d})")

    _WEIGHTS_OUT.write_text(json.dumps(weights_dict, indent=2))
    print(f"\n  Saved → {_WEIGHTS_OUT.relative_to(_REPO)}")

    # ── Stage-B learned classifier CV ─────────────────────────────────────────
    if not args.no_cv:
        _loocv_stage_b(all_records)

    # ── Train Stage-B on all data + save ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Training Stage-B classifier on all data …")
    stage_b, kind = _train_stage_b(all_records)
    acc_stage_b = _accuracy_clf(stage_b, all_records)
    print(f"{'='*60}")
    _print_accuracy(f"Stage-B {kind} (all data, in-sample)", acc_stage_b)

    joblib.dump(stage_b, _STAGE_B_OUT)
    print(f"\n  Saved → {_STAGE_B_OUT.relative_to(_REPO)}")
    print(f"\n  To use:")
    print(f"    import joblib")
    print(f"    clf = BoxingPunchClassifier(stage_b_clf=joblib.load('stage_b_clf.pkl'))")


if __name__ == "__main__":
    main()
