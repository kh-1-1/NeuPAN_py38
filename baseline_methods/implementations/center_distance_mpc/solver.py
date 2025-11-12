"""
Center Distance MPC Implementation

A traditional method that computes dual variables based on the distance
from each point to the center of the point cloud.

Reference: Zhou et al., "Real-time Collision Avoidance for Autonomous Vehicles", RA-L 2020
"""

import torch
import numpy as np
from typing import Tuple, Optional


class CenterDistanceMPC:
    """
    Center Distance MPC method.
    
    This traditional method approximates dual variables by computing
    the distance from each point to the center of the point cloud.
    
    The key idea:
    1. Compute center point: center = mean(point_cloud)
    2. Compute distance from each point to center
    3. Use distance to approximate dual variables
    
    Attributes:
        edge_dim (int): Number of edges (E)
        state_dim (int): State dimension (typically 3)
        G (np.ndarray): Edge constraint matrix
        h (np.ndarray): Edge constraint vector
    """
    
    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 G: Optional[np.ndarray] = None,
                 h: Optional[np.ndarray] = None,
                 distance_scale: float = 1.0):
        """
        Initialize Center Distance MPC.
        
        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            G (np.ndarray): Edge constraint matrix, shape (E, state_dim)
            h (np.ndarray): Edge constraint vector, shape (E,)
            distance_scale (float): Scaling factor for distance
        """
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.distance_scale = distance_scale
        
        # Default G and h for a square robot
        if G is None:
            self.G = np.array([
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ], dtype=np.float32)
        else:
            self.G = np.asarray(G, dtype=np.float32)
        
        if h is None:
            self.h = np.ones(self.edge_dim, dtype=np.float32)
        else:
            self.h = np.asarray(h, dtype=np.float32)
    
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dual variables using center distance method.
        
        Args:
            point_cloud (torch.Tensor): Point cloud, shape (N, 2)
        
        Returns:
            mu (torch.Tensor): Dual variable, shape (E, N)
            lam (torch.Tensor): Auxiliary variable, shape (3, N)
        """
        device = point_cloud.device
        N = point_cloud.shape[0]
        
        # Compute center point
        center = point_cloud.mean(dim=0)  # (2,)
        
        # Compute distance from each point to center
        distances = torch.norm(point_cloud - center, dim=1)  # (N,)
        
        # Normalize distances to [0, 1]
        max_dist = distances.max() + 1e-8
        normalized_distances = distances / max_dist  # (N,)
        
        # Initialize mu and lam
        mu = torch.zeros(self.edge_dim, N, device=device, dtype=torch.float32)
        lam = torch.zeros(self.state_dim, N, device=device, dtype=torch.float32)
        
        # Compute mu based on distance
        # Idea: points closer to center have smaller mu (less constraint)
        # points farther from center have larger mu (more constraint)
        for i in range(self.edge_dim):
            # Distribute distance across edges
            mu[i, :] = normalized_distances * self.distance_scale
        

        
        # Compute lam based on point position relative to center
        # lam represents the direction from center to point
        direction = point_cloud - center  # (N, 2)
        direction_norm = torch.norm(direction, dim=1, keepdim=True) + 1e-8
        direction_normalized = direction / direction_norm  # (N, 2)
        
        # Pad direction to 3D (add z-component)
        lam[0, :] = direction_normalized[:, 0]  # x-component
        lam[1, :] = direction_normalized[:, 1]  # y-component
        lam[2, :] = 0.0  # z-component (zero for 2D)
        

        
        return mu, lam
    
    def __call__(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make the solver callable."""
        return self.forward(point_cloud)


class CenterDistanceMPCAdvanced(CenterDistanceMPC):
    """
    Advanced Center Distance MPC with multiple distance metrics.
    
    This version uses multiple distance metrics (Euclidean, Manhattan, etc.)
    to compute more robust dual variables.
    """
    
    def __init__(self,
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 G: Optional[np.ndarray] = None,
                 h: Optional[np.ndarray] = None,
                 distance_scale: float = 1.0,
                 use_max_distance: bool = True):
        """
        Initialize Advanced Center Distance MPC.
        
        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            G (np.ndarray): Edge constraint matrix
            h (np.ndarray): Edge constraint vector
            distance_scale (float): Scaling factor for distance
            use_max_distance (bool): Use max distance instead of mean
        """
        super().__init__(edge_dim, state_dim, G, h, distance_scale)
        self.use_max_distance = use_max_distance
    
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dual variables using advanced center distance method.
        
        Args:
            point_cloud (torch.Tensor): Point cloud, shape (N, 2)
        
        Returns:
            mu (torch.Tensor): Dual variable, shape (E, N)
            lam (torch.Tensor): Auxiliary variable, shape (3, N)
        """
        device = point_cloud.device
        N = point_cloud.shape[0]
        
        # Compute center point
        center = point_cloud.mean(dim=0)  # (2,)
        
        # Compute multiple distance metrics
        euclidean_dist = torch.norm(point_cloud - center, dim=1)  # (N,)
        manhattan_dist = torch.abs(point_cloud - center).sum(dim=1)  # (N,)
        
        # Combine distances
        combined_dist = (euclidean_dist + manhattan_dist) / 2.0
        
        # Normalize distances
        if self.use_max_distance:
            max_dist = combined_dist.max() + 1e-8
        else:
            max_dist = combined_dist.mean() + 1e-8
        
        normalized_distances = combined_dist / max_dist
        
        # Initialize mu and lam
        mu = torch.zeros(self.edge_dim, N, device=device, dtype=torch.float32)
        lam = torch.zeros(self.state_dim, N, device=device, dtype=torch.float32)
        
        # Compute mu with non-linear scaling
        for i in range(self.edge_dim):
            mu[i, :] = torch.sqrt(normalized_distances) * self.distance_scale
        

        
        # Compute lam
        direction = point_cloud - center
        direction_norm = torch.norm(direction, dim=1, keepdim=True) + 1e-8
        direction_normalized = direction / direction_norm
        
        lam[0, :] = direction_normalized[:, 0]
        lam[1, :] = direction_normalized[:, 1]
        lam[2, :] = 0.0
        

        
        return mu, lam

