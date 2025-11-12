"""CvxpyLayers Solver for Dual Variable Prediction"""
import sys
import os
import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

# Add cvxpylayers path
cvxpy_path = os.path.join(os.path.dirname(__file__), '../../cvxpylayers')
if cvxpy_path not in sys.path:
    sys.path.insert(0, cvxpy_path)

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    CVXPYLAYERS_AVAILABLE = True
except ImportError:
    CVXPYLAYERS_AVAILABLE = False


class CvxpyLayersSolver(nn.Module):
    """Differentiable convex optimization using CvxpyLayers"""
    
    def __init__(self, edge_dim: int = 4, state_dim: int = 3):
        super().__init__()
        
        if not CVXPYLAYERS_AVAILABLE:
            raise ImportError("cvxpylayers not available")
        
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        
        # Define G matrix (robot geometry)
        self.G = torch.tensor([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float32)
        
        # Create cvxpy problem
        self.mu_var = cp.Variable(edge_dim)
        self.lam_var = cp.Variable(state_dim)
        
        # Parameters (will be set during forward)
        self.distance_param = cp.Parameter(1)
        
        # Objective
        objective = cp.Minimize(
            cp.sum_squares(self.mu_var) + cp.sum_squares(self.lam_var)
        )
        
        # Constraints
        G_np = self.G.numpy()
        constraints = [
            self.mu_var >= 0,
            cp.norm(G_np.T @ self.mu_var, 2) <= 1.0,
            cp.norm(self.lam_var, 2) <= 1.0,
        ]
        
        # Create problem
        problem = cp.Problem(objective, constraints)
        
        # Create differentiable layer
        self.cvxpy_layer = CvxpyLayer(
            problem, 
            parameters=[self.distance_param],
            variables=[self.mu_var, self.lam_var]
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
        
        # Initialize outputs
        mu_list = []
        lam_list = []
        
        # Solve for each point
        for i in range(N):
            # Compute distance parameter
            dist = torch.norm(point_cloud[i]).unsqueeze(0)
            
            # Solve
            try:
                mu_i, lam_i = self.cvxpy_layer(dist)
                mu_list.append(mu_i)
                lam_list.append(lam_i)
            except:
                # Fallback
                mu_i = torch.zeros(self.edge_dim, device=device)
                lam_i = torch.zeros(self.state_dim, device=device)
                mu_list.append(mu_i)
                lam_list.append(lam_i)
        
        # Stack
        mu = torch.stack(mu_list, dim=1)  # (E, N)
        lam = torch.stack(lam_list, dim=1)  # (3, N)
        
        return mu, lam

