"""
H36M skeleton graph + punch taxonomy + training helpers.

Separated from the removed ``GCN.py`` module so ``punch_transformer`` and trainers
do not depend on the old ST-GCN implementation.
"""

from __future__ import annotations

import torch

# H36M-17 (MotionBERT output convention)
#   0: pelvis, 1: r_hip, 2: r_knee, 3: r_ankle,
#   4: l_hip,  5: l_knee, 6: l_ankle,
#   7: spine,  8: thorax, 9: neck, 10: head,
#   11: l_shoulder, 12: l_elbow, 13: l_wrist,
#   14: r_shoulder, 15: r_elbow, 16: r_wrist
NUM_H36M_JOINTS = 17

H36M_BONE_PAIRS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),         # right leg
    (0, 4), (4, 5), (5, 6),         # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine → head
    (8, 11), (11, 12), (12, 13),    # left arm
    (8, 14), (14, 15), (15, 16),    # right arm
]

# Canonical 6-class punch ordering (BoxingVI taxonomy)
PUNCH_CLASSES: list[str] = [
    "cross",
    "jab",
    "lead_hook",
    "lead_uppercut",
    "rear_hook",
    "rear_uppercut",
]


def _build_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an undirected adjacency matrix from an edge list (with self-loops)."""
    A = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    A += torch.eye(num_nodes)
    return A


def _normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization: D^(-1/2) A D^(-1/2)."""
    degree = A.sum(dim=1)
    d_inv_sqrt = degree.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def make_class_weights(
    class_counts: dict,
    classes: list | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced cross-entropy training."""
    if classes is None:
        classes = list(PUNCH_CLASSES)
    counts = torch.tensor(
        [class_counts[c] for c in classes],
        dtype=torch.float32,
        device=device,
    )
    weights = 1.0 / counts
    weights = weights * (len(classes) / weights.sum())
    return weights
