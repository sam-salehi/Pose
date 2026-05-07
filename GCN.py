"""
GCN-based punch detector and classifier with a from-scratch ST-GCN backbone.

No dependency on PYSKL — everything is implemented in pure PyTorch.

Architecture: dual-stream (joint + bone) ST-GCN with two downstream heads:
  - GCNDetector: regression head -> punch probability in [0, 1]
  - GCNClassifier: classification head -> logits over 6 punch types

Input shape (both models): [N, M, T, V, C]
  N = batch
  M = persons (1 for BoxingVI single-boxer setting)
  T = time (frames)
  V = joints (17 for COCO)
  C = channels (2 for xy, 3 for xy + confidence)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Skeleton topology — COCO 17-keypoint format
# =============================================================================

# Joint indices for COCO-17:
#   0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear,
#   5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow,
#   9: l_wrist, 10: r_wrist, 11: l_hip, 12: r_hip,
#   13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle
NUM_COCO_JOINTS = 17

# Skeleton edges as (parent, child) where parent is closer to the body center.
# "Closer to center" means: nearer to the hip midpoint along the kinematic
# chain. This convention is used for the bone stream (child - parent vectors)
# and for ST-GCN's centripetal/centrifugal partitioning.
COCO_BONE_PAIRS = [
    (5, 0), (0, 1), (0, 2), (1, 3), (2, 4),  # head → face joints
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 6),                                   # shoulders
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

# Center joint for centripetal/centrifugal partitioning. Conventionally the
# midpoint of the hips is "the center," but since we work with discrete joints
# we use joint 11 (left hip) as a reasonable proxy. The original ST-GCN paper
# uses joint 1 of OpenPose (neck) — choice doesn't matter much in practice.
COCO_CENTER_JOINT = 11


# ── BoxingVI (MediaPipe 12 joints, same order as preprocess.MEDIAPIPE_TO_PAPER)
NUM_BOXINGVI_JOINTS = 12

# Undirected edges for ST-GCN graph (matches preview_npz_labels stick figure).
BOXINGVI_GRAPH_EDGES: list[tuple[int, int]] = [
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

# Directed parent→child for bone vectors (root at left hip 6).
BOXINGVI_BONE_PAIRS: list[tuple[int, int]] = [
    (6, 7),
    (6, 0),
    (7, 1),
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (6, 8),
    (8, 10),
    (7, 9),
    (9, 11),
]

BOXINGVI_CENTER_JOINT = 6


# =============================================================================
# Graph construction — adjacency matrices for ST-GCN's spatial partitioning
# =============================================================================


def _build_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an undirected adjacency matrix from an edge list (with self-loops)."""
    A = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    # Self-loops — every node connected to itself.
    A += torch.eye(num_nodes)
    return A


def _normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalization: D^(-1/2) A D^(-1/2).

    This is the standard graph convolution normalization (Kipf & Welling 2017).
    Prevents feature magnitudes from blowing up at high-degree nodes.
    """
    degree = A.sum(dim=1)
    d_inv_sqrt = degree.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def _shortest_path_distances(A: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise hop distances on the graph defined by adjacency A.

    Uses iterative powers of A: A^k > 0 means a k-hop path exists.
    Returns a (V, V) tensor of integer distances; unreachable pairs get inf.
    """
    V = A.shape[0]
    # Remove self-loops for distance computation; we add them back explicitly.
    A_no_self = A - torch.eye(V)
    A_no_self = (A_no_self > 0).float()

    dist = torch.full((V, V), float("inf"))
    dist[torch.eye(V).bool()] = 0  # zero distance to self

    reach = torch.eye(V)
    for k in range(1, V):
        reach = (reach @ A_no_self > 0).float()
        # Wherever reach is 1 and dist is still inf, set dist = k.
        new_reach = (reach > 0) & torch.isinf(dist)
        dist[new_reach] = k
        if not torch.isinf(dist).any():
            break
    return dist


def build_spatial_partitions(
    num_nodes: int,
    edges: list[tuple[int, int]],
    center: int,
) -> torch.Tensor:
    """
    Build ST-GCN's three-partition spatial adjacency.

    For each (i, j) pair within 1 hop of each other, assign one of three
    partitions based on j's distance to the center compared to i's:
      - Partition 0 (self):         i == j
      - Partition 1 (centripetal):  d(j, center) < d(i, center)
      - Partition 2 (centrifugal):  d(j, center) > d(i, center)
                                    or d(j, center) == d(i, center) (sibling)

    Returns:
        A: tensor of shape [3, V, V], each slice a normalized adjacency matrix
           for one partition.
    """
    A_full = _build_adjacency(num_nodes, edges)
    dist_to_center = _shortest_path_distances(A_full)[:, center]

    A_self = torch.zeros(num_nodes, num_nodes)
    A_centripetal = torch.zeros(num_nodes, num_nodes)
    A_centrifugal = torch.zeros(num_nodes, num_nodes)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if A_full[i, j] == 0:
                continue
            if i == j:
                A_self[i, j] = 1
            elif dist_to_center[j] < dist_to_center[i]:
                A_centripetal[i, j] = 1
            else:
                A_centrifugal[i, j] = 1

    A = torch.stack([
        _normalize_adjacency(A_self + torch.eye(num_nodes)),
        _normalize_adjacency(A_centripetal + torch.eye(num_nodes) * 1e-6),
        _normalize_adjacency(A_centrifugal + torch.eye(num_nodes) * 1e-6),
    ], dim=0)

    return A  # [3, V, V]


# =============================================================================
# ST-GCN building blocks
# =============================================================================


class SpatialGraphConv(nn.Module):
    """
    Graph convolution over the three spatial partitions.

    Uses a learnable edge-importance mask on top of the fixed adjacency,
    following the original ST-GCN paper (Yan et al. 2018). The mask lets
    the network down-weight irrelevant edges and emphasize important ones.

    Args:
        in_channels: input channel count.
        out_channels: output channel count.
        A: adjacency tensor of shape [K, V, V] with K=3 partitions.
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.K = A.shape[0]
        # Adjacency is fixed (not learned), so register as buffer.
        self.register_buffer("A", A)
        # Learnable edge importance — initialized to 1 (no preference).
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        # Per-partition 1x1 conv applied to features before aggregation.
        self.conv = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, C_in, T, V]
        N, C_in, T, V = x.shape
        # Project to K * C_out channels, then split into K partitions.
        x = self.conv(x)  # [N, K*C_out, T, V]
        x = x.view(N, self.K, -1, T, V)  # [N, K, C_out, T, V]
        # Aggregate per partition using A_k * mask_k, then sum across partitions.
        A_masked = self.A * self.edge_importance  # [K, V, V]
        # einsum: for each k, multiply [N, C_out, T, V] by [V, V] over V dim.
        x = torch.einsum("nkctv,kvw->nctw", x, A_masked)
        return x  # [N, C_out, T, V]


class STGCNBlock(nn.Module):
    """
    One ST-GCN block: spatial graph conv + temporal conv + residual.

    Following the original paper:
      - Spatial: graph convolution over K partitions.
      - Temporal: 1D convolution along time with kernel_size=9 by default.
      - Residual connection (with optional projection if shape changes).
      - BN + ReLU after spatial, BN after temporal, ReLU after residual sum.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        temporal_kernel: int = 9,
        stride: int = 1,
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert temporal_kernel % 2 == 1, "temporal_kernel must be odd"
        pad = (temporal_kernel - 1) // 2

        self.spatial = SpatialGraphConv(in_channels, out_channels, A)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.temporal = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(pad, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True) if dropout > 0 else nn.Identity(),
        )

        # Residual path. If channel count or stride changes, project with 1x1.
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, C_in, T, V]
        res = self.residual(x)
        x = self.spatial(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.temporal(x)
        x = x + res
        return F.relu(x, inplace=True)


# =============================================================================
# ST-GCN backbone — drop-in replacement for pyskl.models.backbones.STGCN
# =============================================================================


class STGCN(nn.Module):
    """
    From-scratch ST-GCN backbone. Drop-in replacement for the PYSKL version
    in terms of input/output shape contracts.

    Input:  [N, M, T, V, C]
    Output: [N, M, C_out, T_out, V]

    Architecture: 10 ST-GCN blocks with channel doubling at stages 5 and 8,
    and temporal stride 2 at stages 5 and 8 (downsampling). Final channel
    width is base_channels * 4 (default 256).

    Args:
        in_channels: input feature channels per joint (2 or 3).
        base_channels: width of the first stage. Default 64.
        num_stages: number of ST-GCN blocks. Default 10.
        inflate_stages: 0-indexed stages where channel count doubles.
        down_stages: 0-indexed stages where temporal stride is 2.
        edges: skeleton edge list (default COCO).
        center: center joint for partitioning (default COCO hip).
        num_joints: V (default 17 for COCO).
        dropout: dropout in temporal convs.
        data_bn: whether to apply BatchNorm1d to the flattened input. Helps
                 normalize across persons and joints at the very start.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 10,
        inflate_stages: list[int] | None = None,
        down_stages: list[int] | None = None,
        edges: list[tuple[int, int]] | None = None,
        center: int = COCO_CENTER_JOINT,
        num_joints: int = NUM_COCO_JOINTS,
        dropout: float = 0.0,
        data_bn: bool = True,
    ):
        super().__init__()

        if inflate_stages is None:
            inflate_stages = [5, 8]
        if down_stages is None:
            down_stages = [5, 8]
        if edges is None:
            edges = COCO_BONE_PAIRS

        self.num_joints = num_joints

        # Build the spatial partition adjacency once. Shape: [3, V, V].
        A = build_spatial_partitions(num_joints, edges, center)

        # Optional input-side batch norm. Operates on the [N, M*V*C, T] view,
        # normalizing each (joint, channel) trajectory separately. This is
        # the same trick used by ST-GCN and helps with training stability.
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints) if data_bn else None

        # Build the stages. At each inflate stage, double the channel count;
        # at each down stage, temporal stride = 2.
        blocks = []
        c_in = in_channels
        c_out = base_channels
        for stage in range(num_stages):
            stride = 2 if stage in down_stages else 1
            # First block has no residual (channel mismatch + standard practice).
            residual = stage > 0
            blocks.append(STGCNBlock(c_in, c_out, A, stride=stride,
                                      residual=residual, dropout=dropout))
            c_in = c_out
            if (stage + 1) in inflate_stages:
                c_out = c_out * 2
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = c_in  # final block's output channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, M, T, V, C]

        Returns:
            features: [N, M, C_out, T_out, V]
        """
        N, M, T, V, C = x.shape
        # Reshape to [N*M, C, T, V] for 2D-conv-style processing.
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # [N, M, C, T, V]
        x = x.view(N * M, C, T, V)

        # Optional input batch norm.
        if self.data_bn is not None:
            # Reshape to [N*M, C*V, T] for BN1d.
            xn = x.permute(0, 1, 3, 2).contiguous().view(N * M, C * V, T)
            xn = self.data_bn(xn)
            x = xn.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()

        # Stack of ST-GCN blocks.
        for block in self.blocks:
            x = block(x)

        # Reshape back to [N, M, C_out, T_out, V].
        _, C_out, T_out, V_out = x.shape
        x = x.view(N, M, C_out, T_out, V_out)
        return x


# =============================================================================
# Helpers shared by detector and classifier
# =============================================================================


def _normalize_skeleton_input(x: torch.Tensor) -> torch.Tensor:
    """Coerce pose tensors to ST-GCN layout ``[N, M, T, V, C]``."""
    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4:
        x = x.unsqueeze(1)
    elif x.ndim == 6 and x.shape[2] == 1:
        x = x.squeeze(2)
    if x.ndim != 5:
        raise ValueError(
            "expected skeleton tensor [N, M, T, V, C] or [N, T, V, C]; "
            f"got {tuple(x.shape)} ({x.ndim} dims)"
        )
    return x


def joint_to_bone(x: torch.Tensor, bone_pairs: list = COCO_BONE_PAIRS) -> torch.Tensor:
    """
    Convert joint coordinates to bone vectors (child - parent).

    Args:
        x: tensor of shape [N, M, T, V, C].
        bone_pairs: list of (parent_idx, child_idx) tuples.

    Returns:
        bones: tensor of shape [N, M, T, V, C], joint i replaced by bone
               vector from its parent. Root joints stay zero.
    """
    bones = torch.zeros_like(x)
    for parent, child in bone_pairs:
        bones[..., child, :] = x[..., child, :] - x[..., parent, :]
    return bones


def joint_to_motion(x: torch.Tensor) -> torch.Tensor:
    """
    First-order temporal motion (velocity) per joint.

    Args:
        x: [N, M, T, V, C]

    Returns:
        Same shape; frame t>0 is x[t]-x[t-1], frame 0 is zero.
    """
    motion = torch.zeros_like(x)
    motion[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
    return motion


# Canonical class ordering for the classifier.
PUNCH_CLASSES = [
    "cross",
    "jab",
    "lead_hook",
    "lead_uppercut",
    "rear_hook",
    "rear_uppercut",
]
NUM_CLASSES = len(PUNCH_CLASSES)


# =============================================================================
# Detector — regression head, outputs punch probability in [0, 1]
# =============================================================================


class GCNDetector(nn.Module):
    """
    Multi-stream ST-GCN punch detector: joints, bones (spatial deltas), and
    temporal motion (frame-to-frame joint velocity).

    Same architecture and I/O shape contract as before; fusion width is
    ``3 * backbone_out`` (joint + bone + motion).

    Args:
        in_channels: 2 for (x, y), 3 for (x, y, confidence).
        num_joints: 17 for COCO format.
        feature_dim: width of the fused feature vector before the head.
        dropout: dropout in the regression head.
        backbone_kwargs: dict of additional kwargs forwarded to STGCN.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = NUM_COCO_JOINTS,
        feature_dim: int = 128,
        dropout: float = 0.2,
        backbone_kwargs: dict | None = None,
        bone_pairs: list[tuple[int, int]] | None = None,
    ):
        super().__init__()

        backbone_kwargs = backbone_kwargs or {}
        self.bone_pairs = bone_pairs if bone_pairs is not None else COCO_BONE_PAIRS

        self.joint_stream = STGCN(
            in_channels=in_channels,
            num_joints=num_joints,
            **backbone_kwargs,
        )
        self.bone_stream = STGCN(
            in_channels=in_channels,
            num_joints=num_joints,
            **backbone_kwargs,
        )
        self.motion_stream = STGCN(
            in_channels=in_channels,
            num_joints=num_joints,
            **backbone_kwargs,
        )

        # Backbone out_channels is set by the constructor; streams share config.
        backbone_out = self.joint_stream.out_channels

        self.head = nn.Sequential(
            nn.Linear(backbone_out * 3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    @staticmethod
    def _pool(feat: torch.Tensor) -> torch.Tensor:
        # [N, M, C, T, V] -> [N, C]
        feat = feat.mean(dim=[3, 4])  # over time + joints
        feat = feat.mean(dim=1)        # over persons
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, M, T, V, C] or [N, T, V, C] (single skeleton; M=1 assumed).

        Returns:
            probs: [N], punch probabilities in [0, 1].
        """
        x = _normalize_skeleton_input(x)
        bones = joint_to_bone(x, self.bone_pairs)
        motion = joint_to_motion(x)
        j = self._pool(self.joint_stream(x))
        b = self._pool(self.bone_stream(bones))
        m = self._pool(self.motion_stream(motion))
        fused = torch.cat([j, b, m], dim=-1)
        logit = self.head(fused).squeeze(-1)
        return torch.sigmoid(logit)


# =============================================================================
# Classifier — classification head, outputs logits over 6 punch types
# =============================================================================


class GCNClassifier(nn.Module):
    """
    Dual-stream ST-GCN punch type classifier.

    Args:
        num_classes: number of output classes. Default 6 (BoxingVI taxonomy).
        in_channels: 2 for (x, y), 3 for (x, y, confidence).
        num_joints: 17 for COCO format.
        feature_dim: width of the fused feature vector before the head.
        dropout: dropout rate in the classification head.
        backbone_kwargs: dict of additional kwargs forwarded to STGCN.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 3,
        num_joints: int = NUM_COCO_JOINTS,
        feature_dim: int = 256,
        dropout: float = 0.3,
        backbone_kwargs: dict | None = None,
        bone_pairs: list[tuple[int, int]] | None = None,
    ):
        super().__init__()

        backbone_kwargs = backbone_kwargs or {}
        self.bone_pairs = bone_pairs if bone_pairs is not None else COCO_BONE_PAIRS

        self.joint_stream = STGCN(
            in_channels=in_channels,
            num_joints=num_joints,
            **backbone_kwargs,
        )
        self.bone_stream = STGCN(
            in_channels=in_channels,
            num_joints=num_joints,
            **backbone_kwargs,
        )

        backbone_out = self.joint_stream.out_channels

        # Wider head than the detector — more capacity for the harder 6-way task.
        self.head = nn.Sequential(
            nn.Linear(backbone_out * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes),
        )

    @staticmethod
    def _pool(feat: torch.Tensor) -> torch.Tensor:
        feat = feat.mean(dim=[3, 4])
        feat = feat.mean(dim=1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, M, T, V, C] or [N, T, V, C] (single skeleton; M=1 assumed).

        Returns:
            logits: [N, num_classes].
        """
        x = _normalize_skeleton_input(x)
        bones = joint_to_bone(x, self.bone_pairs)
        j = self._pool(self.joint_stream(x))
        b = self._pool(self.bone_stream(bones))
        fused = torch.cat([j, b], dim=-1)
        return self.head(fused)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple:
        """Returns (predicted_class_indices, softmax_probs)."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs


# =============================================================================
# Helper for class-imbalanced training
# =============================================================================


def make_class_weights(
    class_counts: dict,
    classes: list = PUNCH_CLASSES,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Inverse-frequency class weights for imbalanced cross-entropy training.

    Pass to F.cross_entropy(weight=...) when training the classifier.
    """
    counts = torch.tensor(
        [class_counts[c] for c in classes],
        dtype=torch.float32,
        device=device,
    )
    weights = 1.0 / counts
    weights = weights * (len(classes) / weights.sum())
    return weights


# =============================================================================
# Sanity check
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Detector sanity check")
    print("=" * 60)
    det = GCNDetector(in_channels=3)
    det.eval()
    x_det = torch.randn(4, 1, 11, 17, 3)  # detection window: T=11
    with torch.no_grad():
        y_det = det(x_det)
    print(f"Input:        {tuple(x_det.shape)}")
    print(f"Output:       {tuple(y_det.shape)}")
    print(f"Output range: [{y_det.min().item():.4f}, {y_det.max().item():.4f}]")
    print(f"Parameters:   {sum(p.numel() for p in det.parameters()):,}")

    print()
    print("=" * 60)
    print("Classifier sanity check")
    print("=" * 60)
    clf = GCNClassifier(num_classes=6, in_channels=3)
    clf.eval()
    x_clf = torch.randn(4, 1, 25, 17, 3)  # full punch clip: T=25
    with torch.no_grad():
        logits = clf(x_clf)
        preds, probs = clf.predict(x_clf)
    print(f"Input:        {tuple(x_clf.shape)}")
    print(f"Logits:       {tuple(logits.shape)}")
    print(f"Preds:        {preds.tolist()}")
    print(f"Top probs:    {[f'{p:.3f}' for p in probs.max(dim=-1).values.tolist()]}")
    print(f"Parameters:   {sum(p.numel() for p in clf.parameters()):,}")

    print()
    print("=" * 60)
    print("Class weights helper")
    print("=" * 60)
    fake_counts = {
        "cross": 1200, "jab": 1500, "lead_hook": 800,
        "lead_uppercut": 600, "rear_hook": 750, "rear_uppercut": 450,
    }
    w = make_class_weights(fake_counts)
    print(f"Counts:  {list(fake_counts.values())}")
    print(f"Weights: {[f'{x:.3f}' for x in w.tolist()]}  (mean={w.mean().item():.3f})")