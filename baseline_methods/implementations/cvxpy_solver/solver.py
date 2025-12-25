"""
CVXPY Solver Implementation

Solves the dual variable prediction problem using convex optimization.
This serves as a ground truth baseline.
"""

import torch
import numpy as np
import cvxpy as cp
from typing import Tuple, Optional


class CVXPYSolver:
    """
    CVXPY-based convex optimization solver for dual variable prediction.
    
    This solver finds the optimal dual variables (mu, lambda) that satisfy
    the MPC constraints by solving a convex optimization problem.
    
    Attributes:
        edge_dim (int): Number of edges in robot geometry (E)
        state_dim (int): State dimension (typically 3 for 2D robots)
        G (np.ndarray): Edge constraint matrix, shape (E, state_dim)
        h (np.ndarray): Edge constraint vector, shape (E,)
        solver (str): CVXPY solver to use ('CLARABEL', 'ECOS', 'SCS')
    """
    
    def __init__(self, 
                 edge_dim: int = 4,
                 state_dim: int = 3,
                 G: Optional[np.ndarray] = None,
                 h: Optional[np.ndarray] = None,
                 solver: str = 'CLARABEL',
                 verbose: bool = False):
        """
        Initialize CVXPY Solver.
        
        Args:
            edge_dim (int): Number of edges (E)
            state_dim (int): State dimension (default: 3)
            G (np.ndarray): Edge constraint matrix, shape (E, state_dim)
            h (np.ndarray): Edge constraint vector, shape (E,)
            solver (str): CVXPY solver name
            verbose (bool): Print solver output
        """
        self.edge_dim = edge_dim
        self.state_dim = state_dim
        self.solver_name = solver
        self.verbose = verbose
        
        # Default G and h for a square robot (4 edges)
        if G is None:
            # Square robot: 4 edges
            self.G = np.array([
                [1.0, 0.0, 0.0],   # right edge
                [-1.0, 0.0, 0.0],  # left edge
                [0.0, 1.0, 0.0],   # front edge
                [0.0, -1.0, 0.0],  # back edge
            ], dtype=np.float32)
        else:
            self.G = np.asarray(G, dtype=np.float32)
        
        if h is None:
            self.h = np.ones(self.edge_dim, dtype=np.float32)
        else:
            self.h = np.asarray(h, dtype=np.float32)
    
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve for dual variables given point cloud.
        
        Args:
            point_cloud (torch.Tensor): Point cloud, shape (N, 2)
        
        Returns:
            mu (torch.Tensor): Dual variable, shape (E, N)
            lam (torch.Tensor): Auxiliary variable, shape (3, N)
        """
        device = point_cloud.device
        N = point_cloud.shape[0]
        
        # Convert to numpy
        points_np = point_cloud.cpu().numpy()  # (N, 2)
        
        # Initialize output tensors
        mu_list = []
        lam_list = []
        
        # Solve for each point independently
        for i in range(N):
            point = points_np[i]  # (2,)
            
            # Solve optimization problem for this point
            mu_opt, lam_opt = self._solve_single_point(point)
            
            mu_list.append(mu_opt)
            lam_list.append(lam_opt)
        
        # Stack results
        mu = np.stack(mu_list, axis=1)  # (E, N)
        lam = np.stack(lam_list, axis=1)  # (3, N)
        
        # Convert back to torch
        mu = torch.from_numpy(mu).to(device).float()
        lam = torch.from_numpy(lam).to(device).float()
        
        return mu, lam
    
    def _solve_single_point(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve optimization problem for a single point.
        
        Problem:
            maximize: mu^T (G p - h)
            subject to: mu >= 0
                       ||G^T @ mu||_2 <= 1
        
        Args:
            point (np.ndarray): Single point, shape (2,)
        
        Returns:
            mu (np.ndarray): Optimal mu, shape (E,)
            lam (np.ndarray): Optimal lambda, shape (3,)
        """
        # Define variables
        mu = cp.Variable(self.edge_dim)

        # Pad point to match state_dim if needed (z/theta = 0)
        if self.state_dim > 2:
            p_aug = np.zeros(self.state_dim, dtype=np.float32)
            p_aug[:2] = point[:2]
        else:
            p_aug = point[: self.state_dim]
        p_aug = p_aug.reshape((self.state_dim, 1))

        # Objective: maximize mu^T (G p - h)
        objective = cp.Maximize(mu.T @ (self.G @ p_aug - self.h.reshape(-1, 1)))
        
        # Constraints
        constraints = [
            mu >= 0,  # mu >= 0
            cp.norm(self.G.T @ mu, 2) <= 1,  # ||G^T @ mu||_2 <= 1
        ]

        # Solve problem
        problem = cp.Problem(objective, constraints)
        
        problem.solve(
            solver=getattr(cp, self.solver_name),
            verbose=self.verbose,
            max_iter=1000
        )

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"CVXPY solver status: {problem.status}")

        mu_value = mu.value
        if mu_value is None:
            raise RuntimeError("CVXPY solver returned no solution.")

        mu_opt = np.asarray(mu_value, dtype=np.float32).reshape(-1)
        # Derive lambda from mu (consistent with single-point dual relation)
        lam_opt = -self.G.T @ mu_opt

        return mu_opt, lam_opt
    
    def __call__(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make the solver callable."""
        return self.forward(point_cloud)

