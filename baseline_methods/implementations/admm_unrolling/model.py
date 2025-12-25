"""
ADMM Unrolling Model Implementation

Implements algorithm unrolling using ADMM (Alternating Direction Method of Multipliers).
This version wraps the DeepInverse library's ADMM implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class ADMMUnrolling(nn.Module):
    """
    ADMM Algorithm Unrolling for dual variable prediction.
    
    ADMM (Alternating Direction Method of Multipliers) is a powerful algorithm
    for solving constrained optimization problems. It can be unrolled into a
    neural network with learnable parameters.
    
    The ADMM algorithm alternates between:
    1. x-update: minimize over x
    2. z-update: minimize over z
    3. dual-update: update dual variables
    
    Attributes:
        edge_dim (int): Number of edges (E)
        state_dim (int): State dimension (typically 3)
        num_layers (int): Number of unrolling layers (iterations)
        hidden_dim (int): Hidden dimension for feature extraction
        rho (float): Penalty parameter
    """
    
    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 num_layers: int = 8,
                 hidden_dim: int = 32,
                 rho: float = 1.0,
                 G: Optional[np.ndarray] = None,
                 h: Optional[np.ndarray] = None,
                 learnable_rho: bool = False):
        """
        Initialize ADMM Unrolling.
        
        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            num_layers (int): Number of unrolling layers (default: 8)
            hidden_dim (int): Hidden dimension (default: 32)
            rho (float): Penalty parameter (default: 1.0)
            G (np.ndarray): Edge constraint matrix
            h (np.ndarray): Edge constraint vector
            learnable_rho (bool): Whether rho is learnable
        """
        super(ADMMUnrolling, self).__init__()
        
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = edge_dim + state_dim
        self.learnable_rho = learnable_rho
        
        # Register G and h as buffers
        if G is None:
            G = np.array([
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ], dtype=np.float32)
        if h is None:
            h = np.ones(edge_dim, dtype=np.float32)
        
        self.register_buffer('G', torch.from_numpy(G).float())
        self.register_buffer('h', torch.from_numpy(h).float())
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Learnable penalty parameter
        if learnable_rho:
            self.rho_param = nn.Parameter(torch.tensor(rho, dtype=torch.float32))
        else:
            self.register_buffer('rho_param', torch.tensor(rho, dtype=torch.float32))
        
        # Learnable step sizes for each layer
        self.step_sizes = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)
    
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using ADMM unrolling.
        
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
        
        # Initialize ADMM variables
        x = torch.zeros(N, self.hidden_dim, device=device)
        z = torch.zeros(N, self.hidden_dim, device=device)
        y = torch.zeros(N, self.hidden_dim, device=device)  # dual variable
        
        # Get penalty parameter
        rho = torch.clamp(self.rho_param, min=0.1, max=10.0)
        
        # ADMM iterations
        for layer_idx in range(self.num_layers):
            step_size = torch.clamp(self.step_sizes[layer_idx], min=0.01, max=1.0)
            
            # x-update: minimize over x
            # x = argmin_x ||x - features||^2 + (rho/2) * ||x - z + y/rho||^2
            x_new = (features + rho * (z - y / rho)) / (1.0 + rho)
            x = x + step_size * (x_new - x)
            
            # z-update: minimize over z (with projection)
            # z = argmin_z ||z||^2 + (rho/2) * ||x - z + y/rho||^2
            z_new = (rho * (x + y / rho)) / (1.0 + rho)
            z = z + step_size * (z_new - z)
            
            # Dual update
            y = y + rho * (x - z)
        
        # Output layer - map from hidden_dim to output_dim
        output = self.output_layer(x)  # (N, E+3)

        # Split into mu and lam
        mu = output[:, :self.edge_dim].T  # (E, N)
        lam = output[:, self.edge_dim:].T  # (3, N)

        # No extra constraints or normalization (keep original ADMM output)
        
        return mu, lam
    
    def __call__(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make the model callable."""
        return self.forward(point_cloud)


class ADMMUnrollingWithDeepInverse(nn.Module):
    """
    ADMM Unrolling using DeepInverse library.
    
    This version uses the official DeepInverse library implementation
    for better compatibility and performance.
    """
    
    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 num_layers: int = 8,
                 hidden_dim: int = 32,
                 rho: float = 1.0,
                 G: Optional[np.ndarray] = None,
                 h: Optional[np.ndarray] = None):
        """
        Initialize ADMM Unrolling with DeepInverse.
        
        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            num_layers (int): Number of unrolling layers (default: 8)
            hidden_dim (int): Hidden dimension (default: 32)
            rho (float): Penalty parameter (default: 1.0)
            G (np.ndarray): Edge constraint matrix
            h (np.ndarray): Edge constraint vector
        """
        super(ADMMUnrollingWithDeepInverse, self).__init__()
        
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = edge_dim + state_dim
        
        # Try to import DeepInverse
        try:
            from deepinv.unfolded import ADMM as DeepInverseADMM
        except ImportError as exc:
            raise ImportError("DeepInverse is required for ADMMUnrollingWithDeepInverse.") from exc

        self.admm_net = DeepInverseADMM(
            num_layers=num_layers,
            input_dim=hidden_dim,
            output_dim=self.output_dim,
            rho=rho,
        )
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Register G and h as buffers
        if G is None:
            G = np.array([
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ], dtype=np.float32)
        if h is None:
            h = np.ones(edge_dim, dtype=np.float32)
        
        self.register_buffer('G', torch.from_numpy(G).float())
        self.register_buffer('h', torch.from_numpy(h).float())
    
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using DeepInverse ADMM.
        
        Args:
            point_cloud (torch.Tensor): Point cloud, shape (N, 2)
        
        Returns:
            mu (torch.Tensor): Dual variable, shape (E, N)
            lam (torch.Tensor): Auxiliary variable, shape (3, N)
        """
        # Encode features
        features = self.encoder(point_cloud)  # (N, hidden_dim)
        
        # Apply ADMM unrolling
        output = self.admm_net(features)  # (N, E+3)
        
        # Ensure output has correct shape
        if output.shape[1] != self.output_dim:
            # Pad or truncate
            if output.shape[1] < self.output_dim:
                padding = torch.zeros(output.shape[0], self.output_dim - output.shape[1],
                                    device=output.device, dtype=output.dtype)
                output = torch.cat([output, padding], dim=1)
            else:
                output = output[:, :self.output_dim]
        
        # Split into mu and lam
        mu = output[:, :self.edge_dim].T  # (E, N)
        lam = output[:, self.edge_dim:].T  # (3, N)
        
        # No extra constraints or normalization
        
        return mu, lam
    
    def __call__(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make the model callable."""
        return self.forward(point_cloud)

