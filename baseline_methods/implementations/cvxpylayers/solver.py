"""CvxpyLayers Solver for Dual Variable Prediction"""
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Add cvxpylayers path
cvxpy_path = os.path.join(os.path.dirname(__file__), "../../cvxpylayers")
if cvxpy_path not in sys.path:
    sys.path.insert(0, cvxpy_path)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    cp = None
    CVXPY_AVAILABLE = False

try:
    from cvxpylayers.torch import CvxpyLayer
    CVXPYLAYERS_AVAILABLE = True
except ImportError:
    CvxpyLayer = None
    CVXPYLAYERS_AVAILABLE = False


class CvxpyLayersSolver(nn.Module):
    """Differentiable convex optimization using CvxpyLayers"""

    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 G: np.ndarray = None,
                 h: np.ndarray = None):
        super().__init__()

        self.edge_dim = edge_dim
        self.state_dim = state_dim

        # Default G/h for a square robot
        if G is None:
            G = np.array([
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ], dtype=np.float32)
        if h is None:
            h = np.ones(edge_dim, dtype=np.float32)

        self.G = torch.from_numpy(np.asarray(G, dtype=np.float32))
        self.h = torch.from_numpy(np.asarray(h, dtype=np.float32))

        if not CVXPY_AVAILABLE or not CVXPYLAYERS_AVAILABLE:
            raise ImportError("cvxpylayers and cvxpy are required for CvxpyLayersSolver.")

        G_np = self.G.cpu().numpy()
        h_np = self.h.cpu().numpy()

        mu_var = cp.Variable(edge_dim)
        p_param = cp.Parameter(state_dim)

        objective = cp.Maximize(mu_var.T @ (G_np @ p_param - h_np))
        constraints = [
            mu_var >= 0,
            cp.norm(G_np.T @ mu_var, 2) <= 1.0,
        ]

        problem = cp.Problem(objective, constraints)
        self.cvxpy_layer = CvxpyLayer(
            problem,
            parameters=[p_param],
            variables=[mu_var],
        )

    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_cloud: (N, 2)
        Returns:
            mu: (E, N)
            lam: (3, N)
        """
        N = point_cloud.shape[0]
        device = point_cloud.device
        point_cloud_cpu = point_cloud.to("cpu")
        G_cpu = self.G.to("cpu")

        mu_list = []
        lam_list = []
        for i in range(N):
            p_i = point_cloud_cpu[i]
            if self.state_dim > 2:
                p_aug = torch.zeros(self.state_dim, dtype=p_i.dtype)
                p_aug[:2] = p_i[:2]
            else:
                p_aug = p_i[: self.state_dim]

            mu_i = self.cvxpy_layer(p_aug)[0]
            lam_i = -G_cpu.t().mv(mu_i)

            mu_list.append(mu_i)
            lam_list.append(lam_i)

        mu = torch.stack(mu_list, dim=1).to(device)
        lam = torch.stack(lam_list, dim=1).to(device)
        return mu, lam

