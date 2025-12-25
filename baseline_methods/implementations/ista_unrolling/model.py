"""
ISTA Unrolling Model Implementation

Implements algorithm unrolling using ISTA (Iterative Shrinkage/Thresholding Algorithm).
This version wraps the DeepInverse library's ISTA implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class ISTAUnrolling(nn.Module):
    """
    ISTA Algorithm Unrolling for dual variable prediction.
    
    ISTA (Iterative Shrinkage/Thresholding Algorithm) is a first-order optimization
    algorithm that can be unrolled into a neural network.
    
    The unrolled network has learnable step sizes and shrinkage parameters.
    
    Attributes:
        edge_dim (int): Number of edges (E)
        state_dim (int): State dimension (typically 3)
        num_layers (int): Number of unrolling layers (iterations)
        hidden_dim (int): Hidden dimension for feature extraction
    """
    
    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 num_layers: int = 10,
                 hidden_dim: int = 32,
                 G: Optional[np.ndarray] = None,
                 h: Optional[np.ndarray] = None,
                 learnable_step: bool = True):
        """
        Initialize ISTA Unrolling.
        
        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            num_layers (int): Number of unrolling layers (default: 10)
            hidden_dim (int): Hidden dimension (default: 32)
            G (np.ndarray): Edge constraint matrix
            h (np.ndarray): Edge constraint vector
            learnable_step (bool): Whether step sizes are learnable
        """
        super(ISTAUnrolling, self).__init__()
        
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = edge_dim + state_dim
        self.learnable_step = learnable_step
        
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
        
        # Learnable step sizes for each layer
        if learnable_step:
            self.step_sizes = nn.ParameterList([
                nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
                for _ in range(num_layers)
            ])
            self.shrinkage_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
                for _ in range(num_layers)
            ])
        else:
            self.register_buffer('step_sizes', torch.ones(num_layers) * 0.1)
            self.register_buffer('shrinkage_params', torch.ones(num_layers) * 0.01)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)
    
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using ISTA unrolling.
        
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
        x = torch.zeros(N, self.hidden_dim, device=device)

        # ISTA iterations
        for layer_idx in range(self.num_layers):
            # Gradient step
            if self.learnable_step:
                step_size = torch.clamp(self.step_sizes[layer_idx], min=0.01, max=1.0)
                shrinkage = torch.clamp(self.shrinkage_params[layer_idx], min=0.001, max=0.1)
            else:
                step_size = self.step_sizes[layer_idx]
                shrinkage = self.shrinkage_params[layer_idx]

            # Gradient: -2 * (x - features)
            grad = -2.0 * (x - features)

            # Gradient step
            x = x - step_size * grad

            # Soft thresholding (shrinkage)
            x = torch.sign(x) * torch.clamp(torch.abs(x) - shrinkage, min=0.0)

        # Output layer - map from hidden_dim to output_dim
        output = self.output_layer(x)  # (N, E+3)
        
        # Split into mu and lam
        mu = output[:, :self.edge_dim].T  # (E, N)
        lam = output[:, self.edge_dim:].T  # (3, N)

        # NO constraint enforcement - use raw output from ISTA
        # The original ISTA doesn't have constraint handling

        return mu, lam
    
    def __call__(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make the model callable."""
        return self.forward(point_cloud)


class ISTAUnrollingWithDeepInverse(nn.Module):
    """
    ISTA Unrolling using DeepInverse library.
    
    This version uses the official DeepInverse library implementation
    for better compatibility and performance.
    """
    
    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 num_layers: int = 10,
                 hidden_dim: int = 32,
                 G: Optional[np.ndarray] = None,
                 h: Optional[np.ndarray] = None):
        """
        Initialize ISTA Unrolling with DeepInverse.
        
        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            num_layers (int): Number of unrolling layers (default: 10)
            hidden_dim (int): Hidden dimension (default: 32)
            G (np.ndarray): Edge constraint matrix
            h (np.ndarray): Edge constraint vector
        """
        super(ISTAUnrollingWithDeepInverse, self).__init__()
        
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = edge_dim + state_dim
        
        # Try to import DeepInverse
        try:
            from deepinv.unfolded import ISTA as DeepInverseISTA
        except ImportError as exc:
            raise ImportError("DeepInverse is required for ISTAUnrollingWithDeepInverse.") from exc

        self.ista_net = DeepInverseISTA(
            num_layers=num_layers,
            input_dim=hidden_dim,
            output_dim=self.output_dim,
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
        Forward pass using DeepInverse ISTA.
        
        Args:
            point_cloud (torch.Tensor): Point cloud, shape (N, 2)
        
        Returns:
            mu (torch.Tensor): Dual variable, shape (E, N)
            lam (torch.Tensor): Auxiliary variable, shape (3, N)
        """
        # Encode features
        features = self.encoder(point_cloud)  # (N, hidden_dim)
        
        # Apply ISTA unrolling
        output = self.ista_net(features)  # (N, E+3)
        
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

        # NO constraint enforcement - use raw output from ISTA
        # The original ISTA doesn't have constraint handling

        return mu, lam
    
    def __call__(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make the model callable."""
        return self.forward(point_cloud)

