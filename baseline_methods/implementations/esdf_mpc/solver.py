"""ESDF-MPC Solver for Dual Variable Prediction"""
import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


class ESDFMPCSolver(nn.Module):
    """
    ESDF (Euclidean Signed Distance Field) based MPC solver
    
    Uses distance field to approximate dual variables for MPC-based navigation.
    """
    
    def __init__(self, edge_dim: int = 4, state_dim: int = 3, 
                 grid_resolution: float = 0.1):
        super().__init__()
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.grid_resolution = grid_resolution
        
        # Robot geometry (square robot)
        self.G = torch.tensor([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float32)
        
    def compute_esdf(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Compute ESDF for each point
        
        Args:
            point_cloud: (N, 2)
        Returns:
            esdf: (N,) - signed distance for each point
        """
        N = point_cloud.shape[0]
        device = point_cloud.device
        
        # Compute pairwise distances
        distances = torch.cdist(point_cloud, point_cloud)  # (N, N)
        
        # Minimum distance to other points (excluding self)
        distances = distances + torch.eye(N, device=device) * 1e10
        min_distances = torch.min(distances, dim=1)[0]  # (N,)
        
        return min_distances
        
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
        
        # Compute ESDF
        esdf = self.compute_esdf(point_cloud)  # (N,)
        
        # Compute gradients (approximate)
        gradients = torch.zeros(N, 2, device=device)
        for i in range(N):
            # Find nearest neighbor
            dists = torch.norm(point_cloud - point_cloud[i], dim=1)
            dists[i] = 1e10
            nearest_idx = torch.argmin(dists)
            
            # Gradient points away from nearest obstacle
            direction = point_cloud[i] - point_cloud[nearest_idx]
            gradients[i] = direction / (torch.norm(direction) + 1e-8)
        
        # Convert gradients to dual variables
        # mu represents edge constraints
        mu = torch.zeros(self.edge_dim, N, device=device)
        
        for i in range(N):
            grad = gradients[i]  # (2,)
            
            # Map gradient to edge constraints (no ReLU)
            # Right edge (G[0] = [1, 0, 0])
            mu[0, i] = grad[0]
            # Left edge (G[1] = [-1, 0, 0])
            mu[1, i] = -grad[0]
            # Front edge (G[2] = [0, 1, 0])
            mu[2, i] = grad[1]
            # Back edge (G[3] = [0, -1, 0])
            mu[3, i] = -grad[1]
        

        
        # Compute lambda (auxiliary variables)
        lam = torch.zeros(self.state_dim, N, device=device)
        for i in range(N):
            # Use ESDF value to modulate lambda
            lam[0, i] = gradients[i, 0]
            lam[1, i] = gradients[i, 1]
            lam[2, i] = esdf[i] / 10.0  # Scale factor
        

        
        return mu, lam


class ESDFMPCAdvanced(nn.Module):
    """Advanced ESDF-MPC with learned components"""
    
    def __init__(self, edge_dim: int = 4, state_dim: int = 3):
        super().__init__()
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        
        # Learnable feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(3, 32),  # [x, y, esdf]
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        
        # Learnable decoder
        self.decoder = nn.Linear(64, edge_dim + state_dim)
        
        self.esdf_solver = ESDFMPCSolver(edge_dim, state_dim)
        
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_cloud: (N, 2)
        Returns:
            mu: (E, N)
            lam: (3, N)
        """
        # Get ESDF
        esdf = self.esdf_solver.compute_esdf(point_cloud)  # (N,)
        
        # Combine point cloud with ESDF
        features_input = torch.cat([point_cloud, esdf.unsqueeze(1)], dim=1)  # (N, 3)
        
        # Extract features
        features = self.feature_net(features_input)  # (N, 64)
        
        # Decode
        output = self.decoder(features)  # (N, E+3)
        
        # Transpose and split
        output = output.T  # (E+3, N)
        mu = output[:self.edge_dim, :]  # (E, N)
        lam = output[self.edge_dim:, :]  # (3, N)

        # NO constraint enforcement - use raw output from ESDF-MPC
        # The original ESDF-MPC doesn't have constraint handling

        return mu, lam

