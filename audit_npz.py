#!/usr/bin/env python3
"""
Numeric sanity checks for Dataset/landmarks.npz and the training windowing path.

Run:
    python audit_npz.py
    python audit_npz.py --npz /path/to/landmarks.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
_DEFAULT_NPZ = _REPO / "Dataset" / "landmarks.npz"

# Upper-limb "bones" in 12-joint paper order (for length sanity)
_BONES = [
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (0, 1),
    (6, 7),
    (6, 8),
    (7, 9),
]


def _nan_stats(name: str, arr: np.ndarray) -> None:
    n = arr.size
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    n_zero = int((arr == 0).sum())
    finite = arr[np.isfinite(arr)]
    print(f"  {name}:")
    print(f"    shape {arr.shape}  elements={n}")
    print(f"    NaN {n_nan} ({100.0 * n_nan / max(n, 1):.2f}%)  inf {n_inf}")
    if finite.size:
        print(
            f"    finite: min {finite.min():.5f}  max {finite.max():.5f}  mean {finite.mean():.5f}"
        )
    print(f"    exact zeros {n_zero} ({100.0 * n_zero / max(n, 1):.2f}%)")


def _bone_lengths_midframe(seq: np.ndarray) -> list[float]:
    """Euclidean bone lengths on middle frame; seq (T,12,2)."""
    t = seq.shape[0] // 2
    f = seq[t]
    if np.all(np.isnan(f)):
        return []
    f = np.nan_to_num(f, nan=0.0)
    out: list[float] = []
    for a, b in _BONES:
        out.append(float(np.linalg.norm(f[a] - f[b])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit landmarks.npz quality")
    ap.add_argument("--npz", type=Path, default=_DEFAULT_NPZ)
    args = ap.parse_args()
    p = args.npz
    if not p.exists():
        raise SystemExit(f"Missing {p}")

    from preprocess import prepare_windows

    data = np.load(p, allow_pickle=True)
    seqs = data["sequences"]
    labels = data["labels"]
    n = len(seqs)
    print(f"=== {p} ===\n  clips: {n}  keys: {list(data.files)}")

    # ── raw variable-length sequences ───────────────────────────────────────
    all_nan = 0
    all_el = 0
    lens: list[int] = []
    clip_nan_frac: list[float] = []
    bone_medians: list[float] = []

    for i in range(n):
        s = np.asarray(seqs[i], dtype=np.float64)
        lens.append(s.shape[0])
        all_el += s.size
        nz = np.isnan(s).sum()
        all_nan += int(nz)
        clip_nan_frac.append(nz / max(s.size, 1))
        bl = _bone_lengths_midframe(s)
        if bl:
            bone_medians.append(float(np.median(bl)))

    print("\n--- Raw sequences (as stored, variable T) ---")
    print(f"  T per clip: min {min(lens)}  max {max(lens)}  mean {np.mean(lens):.1f}")
    print(f"  Overall NaN rate: {100.0 * all_nan / max(all_el, 1):.2f}%")
    bad = sum(1 for f in clip_nan_frac if f > 0.25)
    print(f"  Clips with >25% NaN values: {bad} ({100.0 * bad / max(n, 1):.1f}%)")
    if bone_medians:
        bm = np.array(bone_medians)
        print(
            f"  Mid-frame median bone length (per clip): "
            f"min {bm.min():.5f}  p50 {np.median(bm):.5f}  max {bm.max():.5f}"
        )
        print(
            f"  Clips with median bone < 0.01 (very short / collapsed): "
            f"{int((bm < 0.01).sum())}"
        )

    # Concatenate for global stats
    parts = [np.asarray(seqs[i], dtype=np.float64).ravel() for i in range(n)]
    raw_flat = np.concatenate(parts)
    _nan_stats("All raw values concatenated", raw_flat)

    # ── after prepare_windows (same as train.py before nan_to_num) ──────────
    X_w, _ = prepare_windows(seqs, labels)
    _nan_stats("After prepare_windows (20-frame crop, NaN-filled in-clip)", X_w)

    # ── same as GRU input after nan_to_num ─────────────────────────────────
    X_num = np.nan_to_num(X_w.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    _nan_stats("After nan_to_num (GRU path)", X_num)

    # ── per-frame motion proxy: mean abs frame-to-frame delta ──────────────
    d = np.abs(X_num[:, 1:, :, :] - X_num[:, :-1, :, :])
    mean_motion = d.mean(axis=(1, 2, 3))
    print("\n--- Temporal motion (mean |Δ| between consecutive frames in window) ---")
    print(f"  per clip: min {mean_motion.min():.6f}  p50 {np.median(mean_motion):.6f}  max {mean_motion.max():.6f}")
    still = int(np.sum(mean_motion < 1e-6))
    print(
        f"  Clips with near-zero motion (<1e-6): {still} "
        f"(repeated padding / static pose — may be suspicious)"
    )

    print("\n--- Tips ---")
    print("  • Visual:  python preview_npz_labels.py -n 16")
    print("  • Re-extract middle frames vs video:  python preprocess.py validate")
    print("  • Compare annotation timing:  python preprocess.py preview")


if __name__ == "__main__":
    main()
