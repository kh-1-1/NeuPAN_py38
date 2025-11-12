"""Physics-Informed Hard Projection for Dual Variable Prediction"""
import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


class PhysicsInformedHardProj(nn.Module):
    """
    Physics-Informed Hard Projection
    
    Uses hard projection layers to enforce physical constraints,
    similar to NeuPAN's approach but as a baseline comparison.
    """
    
    def __init__(self, edge_dim: int = 4, state_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        
        # Robot geometry
        self.G = torch.tensor([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float32)
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Dual variable predictor
        self.mu_predictor = nn.Linear(hidden_dim, edge_dim)
        self.lam_predictor = nn.Linear(hidden_dim, state_dim)
        
    def hard_projection_mu(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Hard projection for mu to satisfy:
        1. mu >= 0
        2. ||G^T @ mu||_2 <= 1
        
        Args:
            mu: (E, N)
        Returns:
            mu_proj: (E, N)
        """
        # Step 1: Non-negativity
        mu = torch.relu(mu)
        
        # Step 2: Dual feasibility ||G^T @ mu||_2 <= 1
        E, N = mu.shape
        device = mu.device
        G = self.G.to(device)
        
        for i in range(N):
            mu_i = mu[:, i]  # (E,)
            G_T_mu = G.T @ mu_i  # (3,)
            norm = torch.norm(G_T_mu)
            
            if norm > 1.0:
                # Project onto ||G^T @ mu|| = 1
                mu[:, i] = mu_i / (norm + 1e-8)
        
        return mu
    
    def hard_projection_lam(self, lam: torch.Tensor) -> torch.Tensor:
        """
        Hard projection for lambda to satisfy ||lambda||_2 <= 1
        
        Args:
            lam: (3, N)
        Returns:
            lam_proj: (3, N)
        """
        # L2 normalization
        lam = lam / (torch.norm(lam, dim=0, keepdim=True) + 1e-8)
        return lam
        
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_cloud: (N, 2)
        Returns:
            mu: (E, N)
            lam: (3, N)
        """
        # Encode
        features = self.encoder(point_cloud)  # (N, hidden_dim)
        
        # Predict
        mu = self.mu_predictor(features)  # (N, E)
        lam = self.lam_predictor(features)  # (N, 3)
        
        # Transpose
        mu = mu.T  # (E, N)
        lam = lam.T  # (3, N)
        
        # Hard projection
        mu = self.hard_projection_mu(mu)
        lam = self.hard_projection_lam(lam)
        
        return mu, lam


class PhysicsInformedHardProjWithKKT(nn.Module):
    """Physics-Informed Hard Projection with KKT regularization"""
    
    def __init__(self, edge_dim: int = 4, state_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        
        self.base_model = PhysicsInformedHardProj(edge_dim, state_dim, hidden_dim)
        
        # KKT residual network
        self.kkt_net = nn.Sequential(
            nn.Linear(edge_dim + state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
    def compute_kkt_residual(self, mu: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        Compute KKT residual
        
        Args:
            mu: (E, N)
            lam: (3, N)
        Returns:
            residual: (N,)
        """
        # Combine mu and lam
        combined = torch.cat([mu, lam], dim=0)  # (E+3, N)
        combined = combined.T  # (N, E+3)
        
        # Compute residual
        residual = self.kkt_net(combined).squeeze(1)  # (N,)
        
        return residual
        
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_cloud: (N, 2)
        Returns:
            mu: (E, N)
            lam: (3, N)
        """
        mu, lam = self.base_model(point_cloud)
        return mu, lam

