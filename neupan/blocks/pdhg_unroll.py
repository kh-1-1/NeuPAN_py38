"""
PDHG-Unroll: Primal-Dual Hybrid Gradient unrolling for DUNE dual refinement.

Implements J-step PDHG iterations with hard projection to enforce:
  - μ ≥ 0
  - ||G^T μ||_2 ≤ 1

Developed for NeuPAN DF-DUNE enhancement
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import torch
import torch.nn as nn
from typing import Optional


class PDHGUnroll(nn.Module):
    """
    Vectorized PDHG unrolling for batch dual variable refinement.
    
    This module refines dual variables μ (from ObsPointNet) through J steps of
    Primal-Dual Hybrid Gradient iterations, ensuring:
      1. μ ≥ 0 (non-negativity)
      2. ||G^T μ||_2 ≤ 1 (dual feasibility)
    
    The algorithm solves:
        max_{μ≥0, ||G^T μ||≤1} μ^T (G p - h)
    
    via the saddle-point formulation with auxiliary variable y ∈ R^2.
    
    Args:
        E: int, number of edges (rows of G, typically 4-8 for robot geometry)
        J: int, number of unroll steps (default 1; recommended 1-3 for real-time)
        tau: float, primal step size (default 0.5; must satisfy τ·σ·||G||² < 1)
        sigma: float, dual step size (default 0.5)
        learnable_steps: bool, whether to make tau/sigma learnable parameters (default False)
        
    Shape:
        - Input mu: [E, N] where N is batch size (number of points)
        - Input a: [E, N] affine term (G @ p - h)
        - Input G: [E, 2] geometry matrix
        - Output: [E, N] refined dual variables
        
    Example:
        >>> pdhg = PDHGUnroll(E=4, J=2, tau=0.5, sigma=0.5)
        >>> mu_refined = pdhg(mu_init, a, G)
    """
    
    def __init__(
        self, 
        E: int, 
        J: int = 1, 
        tau: float = 0.5, 
        sigma: float = 0.5, 
        learnable_steps: bool = False
    ) -> None:
        super().__init__()
        
        self.E = E
        self.J = J
        
        # Step sizes: learnable parameters or fixed buffers
        if learnable_steps:
            self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
            self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
            self.learnable = True
        else:
            self.register_buffer('tau', torch.tensor(tau, dtype=torch.float32))
            self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
            self.learnable = False
    
    def forward(self, mu: torch.Tensor, a: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        Perform J-step PDHG refinement.
        
        Args:
            mu: [E, N] initial dual variables (from ObsPointNet output)
            a: [E, N] affine term (G @ p - h), where p is [2, N] point positions
            G: [E, 2] geometry matrix (robot convex hull representation)
            
        Returns:
            mu_refined: [E, N] refined dual variables after J PDHG steps
            
        Algorithm:
            For j = 1 to J:
                1. y^{j+1} = Proj_{||·||≤1}(y^j + σ G^T μ^j)
                2. μ^{j+1} = Proj_{≥0}(μ^j + τ (a - G y^{j+1}))
                3. μ^{j+1} = μ^{j+1} / max(1, ||G^T μ^{j+1}||_2)  [safety projection]
        """
        # Validate input shapes
        assert mu.dim() == 2 and mu.shape[0] == self.E, \
            f"mu must be [E={self.E}, N], got {mu.shape}"
        assert a.dim() == 2 and a.shape == mu.shape, \
            f"a must match mu shape {mu.shape}, got {a.shape}"
        assert G.dim() == 2 and G.shape == (self.E, 2), \
            f"G must be [E={self.E}, 2], got {G.shape}"
        
        N = mu.shape[1]
        device = mu.device
        dtype = mu.dtype
        
        # Initialize auxiliary dual variable y ∈ R^{2×N}
        y = torch.zeros(2, N, device=device, dtype=dtype)
        
        # Get step sizes (with optional clamping for learnable case)
        if self.learnable:
            tau = self.tau.clamp(0.01, 0.99)
            sigma = self.sigma.clamp(0.01, 0.99)
        else:
            tau = self.tau
            sigma = self.sigma
        
        # PDHG iterations
        for _ in range(self.J):
            # Step 1: y-update with L2 unit ball projection (column-wise)
            # y^{j+1} = (y^j + σ G^T μ^j) / max(1, ||y^j + σ G^T μ^j||_2)
            y = y + sigma * (G.t() @ mu)  # [2, N]
            y_norm = torch.norm(y, dim=0, keepdim=True)  # [1, N]
            y_norm_clamped = torch.clamp(y_norm, min=1.0)  # max(1, ||y||)
            y = y / y_norm_clamped  # project to unit ball
            
            # Step 2: μ-update with non-negativity projection
            # μ^{j+1} = max(0, μ^j + τ (a - G y^{j+1}))
            mu = mu + tau * (a - G @ y)  # [E, N]
            mu = mu.clamp(min=0.0)  # non-negativity
            
            # Step 3: Safety projection to enforce ||G^T μ||_2 ≤ 1 (column-wise)
            # This is a hard constraint to ensure dual feasibility
            v = G.t() @ mu  # [2, N]
            v_norm = torch.norm(v, dim=0, keepdim=True)  # [1, N]
            v_norm_clamped = torch.clamp(v_norm, min=1.0)  # max(1, ||v||)
            mu = mu / v_norm_clamped  # scale down if violated
        
        return mu
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        learnable_str = "learnable" if self.learnable else "fixed"
        return f"E={self.E}, J={self.J}, tau={self.tau.item():.3f}, sigma={self.sigma.item():.3f}, {learnable_str}"


class PDHGUnrollPerStep(nn.Module):
    """
    Advanced PDHG-Unroll with per-step learnable parameters.
    
    Each of the J steps has independent (τ_j, σ_j), allowing the network to learn
    adaptive step schedules (e.g., large steps early, small steps late).
    
    This is an optional extension for Stage 4 (high-performance scenarios with J≥3).
    
    Args:
        E: int, number of edges
        J: int, number of unroll steps
        tau_init: float, initial value for all τ_j (default 0.5)
        sigma_init: float, initial value for all σ_j (default 0.5)
        
    Example:
        >>> pdhg = PDHGUnrollPerStep(E=4, J=3, tau_init=0.7, sigma_init=0.7)
        >>> mu_refined = pdhg(mu_init, a, G)
    """
    
    def __init__(
        self, 
        E: int, 
        J: int, 
        tau_init: float = 0.5, 
        sigma_init: float = 0.5
    ) -> None:
        super().__init__()
        
        self.E = E
        self.J = J
        
        # Per-step learnable parameters
        self.tau_list = nn.ParameterList([
            nn.Parameter(torch.tensor(tau_init, dtype=torch.float32)) 
            for _ in range(J)
        ])
        self.sigma_list = nn.ParameterList([
            nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32)) 
            for _ in range(J)
        ])
    
    def forward(self, mu: torch.Tensor, a: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """Perform J-step PDHG with per-step parameters."""
        assert mu.dim() == 2 and mu.shape[0] == self.E
        assert a.shape == mu.shape
        assert G.shape == (self.E, 2)
        
        N = mu.shape[1]
        device = mu.device
        dtype = mu.dtype
        
        y = torch.zeros(2, N, device=device, dtype=dtype)
        
        for j in range(self.J):
            # Get step sizes for this iteration (with clamping)
            tau = self.tau_list[j].clamp(0.01, 0.99)
            sigma = self.sigma_list[j].clamp(0.01, 0.99)
            
            # PDHG steps (same as PDHGUnroll)
            y = y + sigma * (G.t() @ mu)
            y_norm = torch.norm(y, dim=0, keepdim=True).clamp(min=1.0)
            y = y / y_norm
            
            mu = mu + tau * (a - G @ y)
            mu = mu.clamp(min=0.0)
            
            v = G.t() @ mu
            v_norm = torch.norm(v, dim=0, keepdim=True).clamp(min=1.0)
            mu = mu / v_norm
        
        return mu
    
    def extra_repr(self) -> str:
        tau_vals = [f"{t.item():.3f}" for t in self.tau_list]
        sigma_vals = [f"{s.item():.3f}" for s in self.sigma_list]
        return f"E={self.E}, J={self.J}, tau={tau_vals}, sigma={sigma_vals}"

