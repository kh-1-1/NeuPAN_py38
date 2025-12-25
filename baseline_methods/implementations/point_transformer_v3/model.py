"""
Point Transformer V3 for Dual Variable Prediction

This implementation DIRECTLY USES the original Point Transformer V3 model from:
Wu et al., "Point Transformer V3: Simpler, Faster, Stronger", CVPR 2024

We import the original PointTransformerV3 class and only modify the output head.

Key principles:
- Use the ORIGINAL Point Transformer V3 backbone
- Keep all encoder blocks unchanged
- Only replace the output head for dual variable prediction
- NO modifications to the core Transformer architecture
"""
import sys
import os
import torch
import torch.nn as nn
from typing import Tuple

# Add Point Transformer V3 path
ptv3_root = os.path.join(os.path.dirname(__file__), "../../")
ptv3_path = os.path.join(ptv3_root, "PointTransformerV3")
if ptv3_root not in sys.path:
    sys.path.insert(0, ptv3_root)

try:
    from PointTransformerV3.model import PointTransformerV3 as PTv3_Model, Point
    PTV3_AVAILABLE = True
except ImportError:
    PTV3_AVAILABLE = False
    print("Warning: Point Transformer V3 not available")


class PointTransformerV3(nn.Module):
    """
    Point Transformer V3 wrapper for dual variable prediction.

    Uses the ORIGINAL Point Transformer V3 backbone from the open-source implementation.
    Only modifies the output head to predict dual variables.
    """

    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3):
        super().__init__()

        self.edge_dim = edge_dim
        self.state_dim = state_dim

        if not PTV3_AVAILABLE:
            raise ImportError("Point Transformer V3 not available.")
        if not torch.cuda.is_available():
            raise RuntimeError("Point Transformer V3 requires CUDA.")

        # Use original Point Transformer V3 backbone
        # Configuration for classification mode (we'll use encoder only)
        self.backbone = PTv3_Model(
            in_channels=3,  # xyz coordinates (we'll pad z=0)
            order=("z", "z-trans"),  # Simplified order
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(48, 48, 48, 48, 48),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(48, 48, 48, 48),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            pre_norm=True,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=False,  # Disable flash attention for compatibility
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=True,  # Use classification mode (encoder only)
        )

        # Output head (replace classification head)
        # PTv3 encoder outputs 512-dim features
        self.ptv3_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, edge_dim + state_dim),
        )

        self.use_ptv3 = True

    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using original Point Transformer V3.

        Args:
            point_cloud (torch.Tensor): Point cloud, shape (N, 2)

        Returns:
            mu (torch.Tensor): Dual variable, shape (E, N)
            lam (torch.Tensor): Auxiliary variable, shape (3, N)
        """
        N = point_cloud.shape[0]

        if self.use_ptv3:
            if point_cloud.device.type != "cuda":
                raise RuntimeError("Point Transformer V3 requires CUDA tensors.")
            if not next(self.parameters()).is_cuda:
                raise RuntimeError("Point Transformer V3 model must be on CUDA.")

            # Prepare data for Point Transformer V3
            # PTv3 expects a Point dictionary with specific format

            # Add z=0 coordinate
            coord = torch.cat([
                point_cloud,
                torch.zeros(N, 1, device=point_cloud.device, dtype=point_cloud.dtype)
            ], dim=1)  # (N, 3)

            # Create Point dictionary
            data_dict = {
                "coord": coord,  # (N, 3)
                "feat": coord,   # Use coordinates as features
                "grid_size": 0.01,  # Grid size for voxelization
                "offset": torch.tensor([N], device=point_cloud.device),  # Batch offset
            }

            # Forward through PTv3 backbone
            point = self.backbone(data_dict)

            # Extract global features
            # PTv3 returns a Point object with features
            features = point.feat  # (M, 512) where M is number of voxels

            # Global pooling
            features = features.mean(dim=0, keepdim=True)  # (1, 512)

            # Output head
            output = self.ptv3_head(features)  # (1, E+3)

            # Expand to all points
            output = output.expand(N, -1)  # (N, E+3)

        # Split into mu and lam
        mu = output[:, :self.edge_dim].T  # (E, N)
        lam = output[:, self.edge_dim:].T  # (3, N)

        # NO constraint enforcement - use raw output
        # The original PTv3 doesn't have constraint handling

        return mu, lam

