"""PointNet++ Wrapper for Dual Variable Prediction"""
import sys
import os
import torch
import torch.nn as nn
from typing import Tuple

# Add PointNet++ path
pointnet_path = os.path.join(os.path.dirname(__file__), "../../Pointnet_Pointnet2_pytorch")
models_path = os.path.join(pointnet_path, "models")
if pointnet_path not in sys.path:
    sys.path.insert(0, pointnet_path)
if models_path not in sys.path:
    sys.path.insert(0, models_path)

try:
    from models.pointnet2_cls_ssg import get_model as get_pointnet2_ssg
except ImportError:
    get_pointnet2_ssg = None


class PointNetPlusPlus(nn.Module):
    """PointNet++ wrapper for dual variable prediction"""
    
    def __init__(self, edge_dim: int = 4, state_dim: int = 3):
        super().__init__()
        self.edge_dim = edge_dim
        self.state_dim = state_dim

        if get_pointnet2_ssg is None:
            raise ImportError("PointNet++ not found. Please check the path.")
        else:
            self.use_pointnet = True
            # Use PointNet++ backbone
            self.backbone = get_pointnet2_ssg(num_class=edge_dim + state_dim, normal_channel=False)

    def _freeze_bn(self) -> None:
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_cloud: (N, 2)
        Returns:
            mu: (E, N)
            lam: (3, N)
        """
        N = point_cloud.shape[0]

        if self.use_pointnet:
            if self.training:
                # PointNet++ sees a single cloud (batch=1); freeze BN to avoid batch-size errors.
                self._freeze_bn()
            if N < 32:
                raise ValueError("PointNet++ requires at least 32 points (nsample=32).")

            # Add batch and z-coordinate
            if point_cloud.dim() == 2:
                # Add z=0 coordinate
                xyz = torch.cat(
                    [point_cloud, torch.zeros(N, 1, device=point_cloud.device)],
                    dim=1,
                )
                xyz = xyz.unsqueeze(0)  # (1, N, 3)

            # Transpose to (B, 3, N)
            xyz = xyz.permute(0, 2, 1)

            # Forward through backbone
            logits, _ = self.backbone(xyz)  # (B, edge_dim+state_dim)

            # Expand to all points
            output = logits.unsqueeze(-1).expand(-1, -1, N)  # (B, E+3, N)

            # Split
            mu = output[:, :self.edge_dim, :].squeeze(0)  # (E, N)
            lam = output[:, self.edge_dim:, :].squeeze(0)  # (3, N)

        # NO constraint enforcement - use raw output from PointNet++
        # The original PointNet++ doesn't have constraint handling

        return mu, lam

