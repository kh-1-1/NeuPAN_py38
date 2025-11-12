"""
MLP Baseline Model Implementation

A simple multi-layer perceptron that serves as a black-box baseline
for point cloud to dual variable mapping.
"""

import torch
import torch.nn as nn
from typing import Tuple


class MLPBaseline(nn.Module):
    """
    Standard Multi-Layer Perceptron baseline.
    
    Maps point clouds directly to dual variables without any structural
    constraints. This serves as a simple black-box neural network baseline.
    
    Architecture:
        Input (2) -> FC -> ReLU -> FC -> ReLU -> ... -> Output (E+3)
    
    Attributes:
        edge_dim (int): Number of edges (E)
        state_dim (int): State dimension (typically 3)
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of hidden layers
    """
    
    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 hidden_dim: int = 64,
                 num_layers: int = 4,
                 dropout: float = 0.0):
        """
        Initialize MLP Baseline.
        
        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            hidden_dim (int): Hidden layer dimension (default: 64)
            num_layers (int): Number of hidden layers (default: 4)
            dropout (float): Dropout rate (default: 0.0)
        """
        super(MLPBaseline, self).__init__()
        
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = edge_dim + state_dim
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            point_cloud (torch.Tensor): Point cloud, shape (N, 2)
        
        Returns:
            mu (torch.Tensor): Dual variable, shape (E, N)
            lam (torch.Tensor): Auxiliary variable, shape (3, N)
        """
        # point_cloud: (N, 2)
        N = point_cloud.shape[0]
        
        # Forward through MLP
        output = self.mlp(point_cloud)  # (N, E+3)
        
        # Split into mu and lam
        mu = output[:, :self.edge_dim].T  # (E, N)
        lam = output[:, self.edge_dim:].T  # (3, N)

        # NO constraint enforcement - use raw output from MLP
        # The original MLP doesn't have constraint handling

        return mu, lam
    
    def __call__(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make the model callable."""
        return self.forward(point_cloud)


class MLPBaselineWithBatchNorm(nn.Module):
    """
    MLP Baseline with Batch Normalization.
    
    Improved version with batch normalization for better training stability.
    """
    
    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 hidden_dim: int = 64,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        """
        Initialize MLP Baseline with Batch Normalization.
        
        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            hidden_dim (int): Hidden layer dimension (default: 64)
            num_layers (int): Number of hidden layers (default: 4)
            dropout (float): Dropout rate (default: 0.1)
        """
        super(MLPBaselineWithBatchNorm, self).__init__()
        
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = edge_dim + state_dim
        
        # Build MLP layers with batch norm
        layers = []
        
        # Input layer
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            point_cloud (torch.Tensor): Point cloud, shape (N, 2)
        
        Returns:
            mu (torch.Tensor): Dual variable, shape (E, N)
            lam (torch.Tensor): Auxiliary variable, shape (3, N)
        """
        # point_cloud: (N, 2)
        N = point_cloud.shape[0]
        
        # Forward through MLP
        output = self.mlp(point_cloud)  # (N, E+3)
        
        # Split into mu and lam
        mu = output[:, :self.edge_dim].T  # (E, N)
        lam = output[:, self.edge_dim:].T  # (3, N)

        # NO constraint enforcement - use raw output from MLP
        # The original MLP doesn't have constraint handling

        return mu, lam
    
    def __call__(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make the model callable."""
        return self.forward(point_cloud)

