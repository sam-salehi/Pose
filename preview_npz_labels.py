#!/usr/bin/env python3
"""
Random visual audit: clips from landmarks.npz with their stored labels.

Loads `sequences`, `labels`, `versions`, `start_frames`, `end_frames` from the
same .npz written by preprocess.build_dataset — same row index is one clip.

Usage:
    python preview_npz_labels.py
    python preview_npz_labels.py -n 20 --seed 42 -o figures/my_check.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent
_DEFAULT_NPZ = _REPO / "Dataset" / "landmarks.npz"

# 12-joint paper order — edges for stick figure (normalized x,y in [0,1])
_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (0, 6),
    (1, 7),
    (6, 7),
    (6, 8),
    (8, 10),
    (7, 9),
    (9, 11),
]


def _draw_pose(ax, xy: np.ndarray, *, title: str) -> None:
    xy = np.asarray(xy, dtype=np.float64)
    for i, j in _EDGES:
        a, b = xy[i], xy[j]
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#2563eb", lw=2.2, alpha=0.85)
    for k in range(12):
        if np.all(np.isfinite(xy[k])):
            ax.scatter(xy[k, 0], xy[k, 1], c="#dc2626", s=28, zorder=5, edgecolors="white", linewidths=0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Random skeleton previews with labels from landmarks.npz (for manual verification)."
    )
    ap.add_argument("--npz", type=Path, default=_DEFAULT_NPZ, help="path to landmarks.npz")
    ap.add_argument("-n", type=int, default=12, help="how many random clips to show")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (reproducible grid)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="output PNG path")
    ap.add_argument("--cols", type=int, default=4, help="grid columns")
    args = ap.parse_args()

    if not args.npz.exists():
        raise SystemExit(f"Not found: {args.npz}")

    data = np.load(args.npz, allow_pickle=True)
    required = ("sequences", "labels", "versions", "start_frames", "end_frames")
    for k in required:
        if k not in data:
            raise SystemExit(f"Missing array {k!r} in {args.npz}")

    seqs = data["sequences"]
    labels = data["labels"]
    versions = data["versions"]
    starts = data["start_frames"]
    ends = data["end_frames"]

    n_total = len(seqs)
    if n_total != len(labels):
        raise SystemExit(f"Mismatch: len(sequences)={n_total} vs len(labels)={len(labels)}")
    for name, arr in ("versions", versions), ("start_frames", starts), ("end_frames", ends):
        if len(arr) != n_total:
            raise SystemExit(f"Mismatch: len(sequences)={n_total} vs len({name})={len(arr)}")

    rng = np.random.default_rng(args.seed)
    n = min(args.n, n_total)
    pick = rng.choice(n_total, size=n, replace=False)

    cols = max(1, args.cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.45))
    axes_flat = np.atleast_1d(axes).ravel()

    print(f"Loaded {args.npz} — {n_total} clips. Showing {n} random indices (seed={args.seed!r}).\n")
    print(f"{'idx':>5}  {'label':<22}  {'ver':>5}  frames      T")
    print("-" * 58)

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i >= n:
            ax.axis("off")
            continue

        si = int(pick[i])
        seq = np.asarray(seqs[si], dtype=np.float32)
        T = seq.shape[0]
        mid = T // 2
        pose = np.nan_to_num(seq[mid], nan=0.0, posinf=0.0, neginf=0.0)

        lab = str(labels[si])
        ver = str(versions[si])
        sf, ef = int(starts[si]), int(ends[si])

        title = f"{lab}\nidx={si} {ver}  f[{sf}–{ef}]  frame {mid}/{T}"
        _draw_pose(ax, pose, title=title)

        print(f"{si:5d}  {lab:<22}  {ver:>5}  [{sf:5d},{ef:5d}]  {T:3d}")

    plt.tight_layout()

    out = args.output
    if out is None:
        fig_dir = _REPO / "figures"
        fig_dir.mkdir(exist_ok=True)
        seed_tag = args.seed if args.seed is not None else "none"
        out = fig_dir / f"npz_label_preview_n{n}_seed{seed_tag}.png"

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("-" * 58)
    print(f"\nSaved figure → {out}")
    print("Check each subplot title: label should match the pose you expect for that clip.")


if __name__ == "__main__":
    main()
