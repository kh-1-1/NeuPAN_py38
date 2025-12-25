"""
Physics-Informed Hard Projection Model - Using KKThPINN's NNOPT

This implementation DIRECTLY USES the original KKThPINN model from:
Li et al., "KKT-informed neural network for economic model predictive control",
Computers & Chemical Engineering, 2025

We import the NNOPT class from the kkthpinn library and adapt it for our dual variable prediction task.

KKThPINN's NNOPT structure:
    z_free = MLP(input)
    z = B_star @ z_free + A_star @ input + b_star
where B_star, A_star, b_star are computed from constraint matrices A, B, b
such that A @ input + B @ z = b is automatically satisfied.

Key principles:
- Use the ORIGINAL KKThPINN NNOPT class
- Only adapt input/output interface for our problem
- NO modifications to the core algorithm
"""
import importlib.util
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

# Add kkthpinn path
kkthpinn_path = os.path.join(os.path.dirname(__file__), "../../kkthpinn")
if kkthpinn_path not in sys.path:
    sys.path.insert(0, kkthpinn_path)

KKTHPINN_AVAILABLE = False
NNOPT = None
models_path = os.path.join(kkthpinn_path, "models.py")
if os.path.isfile(models_path):
    try:
        spec = importlib.util.spec_from_file_location("kkthpinn_models", models_path)
        kkthpinn_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kkthpinn_models)
        NNOPT = kkthpinn_models.NNOPT
        KKTHPINN_AVAILABLE = True
    except Exception:
        KKTHPINN_AVAILABLE = False

if not KKTHPINN_AVAILABLE:
    print("Warning: KKThPINN not available")


class PhysicsInformedHardProj(nn.Module):
    """
    Physics-Informed Hard Projection using KKThPINN's NNOPT.

    This is a wrapper around the original KKThPINN NNOPT model.
    We only adapt the input/output interface for dual variable prediction.
    """

    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 hidden_dim: int = 128,
                 hidden_num: int = 3,
                 G: Optional[np.ndarray] = None,
                 A: Optional[np.ndarray] = None,
                 B: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None):
        """
        Initialize Physics-Informed model using KKThPINN.

        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            hidden_dim (int): Hidden layer dimension
            hidden_num (int): Number of hidden layers
            G (np.ndarray): Edge constraint matrix (E x 3)
            A (np.ndarray): Constraint matrix for input (m x 2)
            B (np.ndarray): Constraint matrix for output (m x (E+3))
            b (np.ndarray): Constraint vector (m x 1 or m,)
        """
        super().__init__()

        self.edge_dim = edge_dim
        self.state_dim = state_dim
        if not KKTHPINN_AVAILABLE:
            raise ImportError("KKThPINN not available.")

        # Default G matrix (square robot)
        if G is None:
            G = np.array([
                [1.0, 0.0, 0.0],   # right edge
                [-1.0, 0.0, 0.0],  # left edge
                [0.0, 1.0, 0.0],   # front edge
                [0.0, -1.0, 0.0],  # back edge
            ], dtype=np.float32)

        self.G = torch.from_numpy(G).float()  # (E, 3)

        if A is None or B is None or b is None:
            raise ValueError("KKThPINN requires constraint matrices A, B, b.")

        input_dim = 2  # point cloud (x, y)
        z0_dim = edge_dim + state_dim  # output dimension

        A_t = torch.as_tensor(A, dtype=torch.float32)
        B_t = torch.as_tensor(B, dtype=torch.float32)
        b_t = torch.as_tensor(b, dtype=torch.float32)

        self.kkthpinn_model = NNOPT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_num=hidden_num,
            z0_dim=z0_dim,
            A=A_t,
            B=B_t,
            b=b_t
        )

    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using KKThPINN's NNOPT.

        Args:
            point_cloud (torch.Tensor): Point cloud, shape (N, 2)

        Returns:
            mu (torch.Tensor): Dual variable, shape (E, N)
            lam (torch.Tensor): Auxiliary variable, shape (3, N)
        """
        N = point_cloud.shape[0]

        # Use original KKThPINN NNOPT
        output = self.kkthpinn_model(point_cloud)  # (N, E+3)

        # Split into mu and lam
        mu = output[:, :self.edge_dim].T  # (E, N)
        lam = output[:, self.edge_dim:].T  # (3, N)

        # NO constraint enforcement - use raw output from KKThPINN
        # The original KKThPINN doesn't have constraint handling

        return mu, lam

