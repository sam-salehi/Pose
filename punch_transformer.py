"""
Transformer-based models for 3D punch classification and detection.

Designed for MotionBERT H36M-17 skeletons where median clip ≈ 8 frames.
ST-GCN is poorly suited to very short sequences because its stride-2 temporal
downsampling collapses T=8 → T=4 → T=2; self-attention has no such constraint.

Architecture (shared backbone)
--------------------------------
  Input (N, T, V, C)                e.g. (64, 8, 17, 3)
      │
  SpatialEncoder                    2-layer graph conv per frame
      │                             (N, T, V, C) → (N, T, d_model)
      │                             joints mean-pooled after layer 2
      │
  + learnable positional embed      (T, d_model)
      │
  TransformerEncoder                Pre-LN, batch_first=True
      │                             (N, T, d_model) → (N, T, d_model)
      │
  Mean-pool over T                  (N, d_model)
      │
  Task-specific head
      PunchTransformer      → (N, num_classes) logits
      PunchDetectorTransformer → (N,) sigmoid probabilities
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skeleton_constants import (
    H36M_BONE_PAIRS,
    NUM_H36M_JOINTS,
    PUNCH_CLASSES,
    _build_adjacency,
    _normalize_adjacency,
)

NUM_CLASSES = len(PUNCH_CLASSES)


# =============================================================================
# Spatial encoder — per-frame graph convolution
# =============================================================================


class _SpatialGCNLayer(nn.Module):
    """
    Single spatial graph-conv applied independently to each frame.

    Input/output: (N, T, V, C_in) → (N, T, V, C_out)

    Uses a single normalized adjacency (no K-partition). K-partitioning
    compensates for the lack of temporal reasoning in ST-GCN; here the
    Transformer handles all relational reasoning so a simple adjacency suffices.
    Learnable edge-importance mask retained so the network can suppress
    irrelevant skeleton connections.
    """

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor):
        super().__init__()
        V = A.shape[0]
        self.register_buffer("A", A)
        self.edge_importance = nn.Parameter(torch.ones(V, V))
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, T, V, _ = x.shape
        x = self.linear(x)                                  # (N, T, V, C_out)
        x = torch.einsum("ntvd,vw->ntwd", x, self.A * self.edge_importance)
        x = self.bn(x.reshape(N * T * V, -1)).reshape(N, T, V, -1)
        return F.relu(x, inplace=True)


class SpatialEncoder(nn.Module):
    """
    Two _SpatialGCNLayer blocks → joint mean-pool.

    Input:  (N, T, V, C)
    Output: (N, T, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edges: list[tuple[int, int]],
        num_joints: int,
    ):
        super().__init__()
        A = _normalize_adjacency(_build_adjacency(num_joints, edges))
        self.layer1 = _SpatialGCNLayer(in_channels, hidden_channels, A)
        self.layer2 = _SpatialGCNLayer(hidden_channels, out_channels, A)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)    # (N, T, V, hidden)
        x = self.layer2(x)    # (N, T, V, out)
        return x.mean(dim=2)  # (N, T, out)


# =============================================================================
# Shared backbone
# =============================================================================


class _PunchTransformerBase(nn.Module):
    """
    Spatial GCN + Temporal Transformer backbone shared by classifier and detector.

    Subclasses attach their own head and implement ``forward``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = NUM_H36M_JOINTS,
        edges: list[tuple[int, int]] | None = None,
        spatial_hidden: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        if edges is None:
            edges = H36M_BONE_PAIRS

        self.spatial = SpatialEncoder(
            in_channels, spatial_hidden, d_model, edges, num_joints
        )

        self.pos_embed = nn.Parameter(torch.zeros(max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable for small datasets
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.d_model = d_model

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (N, T, V, C) → (N, d_model) via spatial GCN + Transformer + mean-pool."""
        x = self.spatial(x)                      # (N, T, d_model)
        x = x + self.pos_embed[:x.shape[1]]
        x = self.transformer(x)                  # (N, T, d_model)
        return x.mean(dim=1)                     # (N, d_model)


# =============================================================================
# Classifier
# =============================================================================


class PunchTransformer(_PunchTransformerBase):
    """
    6-class punch-type classifier.

    Input:  (N, T, V, C)
    Output: (N, num_classes) raw logits
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 3,
        num_joints: int = NUM_H36M_JOINTS,
        edges: list[tuple[int, int]] | None = None,
        spatial_hidden: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels, num_joints=num_joints, edges=edges,
            spatial_hidden=spatial_hidden, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_class_indices, softmax_probs). Input: (N, T, V, C)."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs


# =============================================================================
# Detector
# =============================================================================


class PunchDetectorTransformer(_PunchTransformerBase):
    """
    Binary punch detector — outputs punch probability in [0, 1].

    Same spatial GCN + Transformer backbone as PunchTransformer; head
    collapses to a single sigmoid score instead of per-class logits.

    Input:  (N, T, V, C)   — sliding window of 3D skeleton frames
    Output: (N,)           — punch probabilities in [0, 1]
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = NUM_H36M_JOINTS,
        edges: list[tuple[int, int]] | None = None,
        spatial_hidden: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels, num_joints=num_joints, edges=edges,
            spatial_hidden=spatial_hidden, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self._encode(x)).squeeze(-1))


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":
    x = torch.randn(4, 8, 17, 3)

    clf = PunchTransformer()
    clf.eval()
    with torch.no_grad():
        logits = clf(x)
        preds, probs = clf.predict(x)
    print(f"Classifier  input {tuple(x.shape)} → logits {tuple(logits.shape)}")
    print(f"            params {sum(p.numel() for p in clf.parameters()):,}")

    det = PunchDetectorTransformer()
    det.eval()
    x_det = torch.randn(4, 16, 17, 3)
    with torch.no_grad():
        scores = det(x_det)
    print(f"Detector    input {tuple(x_det.shape)} → probs {tuple(scores.shape)}")
    print(f"            range [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"            params {sum(p.numel() for p in det.parameters()):,}")
