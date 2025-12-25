"""
FISTA Unrolling Model - Original Algorithm Structure

This implements the FISTA (Fast Iterative Shrinkage/Thresholding Algorithm) from:
Beck & Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM 2009

Key principles:
1. Keep the original FISTA iteration structure
2. Use momentum acceleration (Nesterov acceleration)
3. Use soft-thresholding as proximal operator
4. NO hard projection layer (that's our innovation)
5. NO KKT regularization (that's our innovation)
6. Only simple ReLU and normalization for constraints

FISTA iteration:
    u_k = z_k - step_size * grad(z_k)
    x_{k+1} = soft_threshold(u_k, threshold)
    z_{k+1} = x_{k+1} + alpha_k * (x_{k+1} - x_k)  # Momentum
where alpha_k = (k + a - 1) / (k + a)
"""
import sys
import os
import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class FISTAUnrolling(nn.Module):
    """
    FISTA (Fast ISTA) Algorithm Unrolling for dual variable prediction.

    FISTA is an accelerated version of ISTA using Nesterov momentum.
    It converges faster than ISTA (O(1/k^2) vs O(1/k)).

    Original FISTA structure:
    - Gradient step on momentum variable z
    - Soft-thresholding (proximal operator)
    - Momentum update with alpha_k = (k+a-1)/(k+a)

    Attributes:
        edge_dim (int): Number of edges (E)
        state_dim (int): State dimension (typically 3)
        num_layers (int): Number of unrolling layers (6-8 for FISTA)
        hidden_dim (int): Hidden dimension for feature extraction
        a (float): Momentum parameter (default: 3, must be > 2)
    """

    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 num_layers: int = 8,
                 hidden_dim: int = 64,
                 a: float = 3.0,
                 learnable_step: bool = True):
        """
        Initialize FISTA Unrolling.

        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            num_layers (int): Number of unrolling layers (default: 8)
            hidden_dim (int): Hidden dimension (default: 64)
            a (float): Momentum parameter (default: 3.0, must be > 2)
            learnable_step (bool): Whether step sizes are learnable
        """
        super().__init__()

        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = edge_dim + state_dim
        self.a = a  # Momentum parameter
        self.learnable_step = learnable_step

        # Feature encoder (point cloud -> features)
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Learnable step sizes and thresholds for each layer
        if learnable_step:
            self.step_sizes = nn.ParameterList([
                nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
                for _ in range(num_layers)
            ])
            self.thresholds = nn.ParameterList([
                nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
                for _ in range(num_layers)
            ])
        else:
            self.register_buffer('step_sizes', torch.ones(num_layers) * 0.1)
            self.register_buffer('thresholds', torch.ones(num_layers) * 0.01)

        # Output layer (features -> mu, lambda)
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)

    def soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Soft-thresholding operator (proximal operator for L1 norm).

        soft_threshold(x, t) = sign(x) * max(|x| - t, 0)

        Args:
            x: Input tensor
            threshold: Threshold value
        Returns:
            Soft-thresholded tensor
        """
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)

    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using FISTA unrolling.

        Args:
            point_cloud (torch.Tensor): Point cloud, shape (N, 2)

        Returns:
            mu (torch.Tensor): Dual variable, shape (E, N)
            lam (torch.Tensor): Auxiliary variable, shape (3, N)
        """
        N = point_cloud.shape[0]
        device = point_cloud.device

        # Encode features
        features = self.encoder(point_cloud)  # (N, hidden_dim)

        # Initialize variables
        x = torch.zeros(N, self.output_dim, device=device)  # Current iterate
        z = torch.zeros(N, self.output_dim, device=device)  # Momentum variable
        x_prev = x.clone()

        # FISTA iterations
        for k in range(self.num_layers):
            # Get step size and threshold
            if self.learnable_step:
                step_size = torch.clamp(self.step_sizes[k], min=0.01, max=1.0)
                threshold = torch.clamp(self.thresholds[k], min=0.001, max=0.1)
            else:
                step_size = self.step_sizes[k]
                threshold = self.thresholds[k]

            # Gradient step on momentum variable z
            # grad = -2 * (z - target), where target is from features
            target = self.output_layer(features)  # (N, E+3)
            grad = -2.0 * (z - target)
            u = z - step_size * grad

            # Proximal step (soft-thresholding)
            x_new = self.soft_threshold(u, threshold)

            # Momentum update (Nesterov acceleration)
            # alpha_k = (k + a - 1) / (k + a)
            alpha = (k + self.a - 1.0) / (k + self.a)
            z = x_new + alpha * (x_new - x_prev)

            # Update for next iteration
            x_prev = x.clone()
            x = x_new

        # Final output
        output = x  # (N, E+3)

        # Split into mu and lam
        mu = output[:, :self.edge_dim].T  # (E, N)
        lam = output[:, self.edge_dim:].T  # (3, N)

        # NO constraint enforcement - use raw output from FISTA
        # The original FISTA doesn't have constraint handling

        return mu, lam


class DeepInverseUnrolling(FISTAUnrolling):
    """
    DeepInverse-style unrolling baseline.

    This is a lightweight wrapper around the FISTA unrolling model to provide
    a stable baseline interface even when the external DeepInverse library is
    not installed.
    """

    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 num_layers: int = 8,
                 hidden_dim: int = 64,
                 a: float = 3.0,
                 learnable_step: bool = True):
        super().__init__(
            edge_dim=edge_dim,
            state_dim=state_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            a=a,
            learnable_step=learnable_step,
        )

