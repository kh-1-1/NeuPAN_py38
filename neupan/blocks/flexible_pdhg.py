"""
FlexiblePDHGFront: A flexible PDHG-style unfolded front-end that directly
maps points -> dual variables mu, replacing ObsPointNet (+ optional PDHG).

Design goals
- Keep I/O identical to ObsPointNet: forward(x[N,2]) -> mu[N,E].
- Use real, fixed geometry (G, h). Do NOT learn geometry.
- Per-step: gradient-like update + optional learned-prox residual + hard projections
  to guarantee dual feasibility: mu >= 0 and ||G^T mu||_2 <= 1.

This module can fully substitute the original front by setting in train YAML:
  train:
    front: flex_pdhg
    front_J: 1           # number of unfolded steps
    front_hidden: 32     # hidden width
    front_learned: true  # enable learned-prox residual per step
    projection: hard     # keep DUNE hard projection as idempotent fallback

"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class FlexiblePDHGFront(nn.Module):
    def __init__(
        self,
        input_dim: int,
        E: int,
        G: torch.Tensor,
        h: torch.Tensor,
        hidden: int = 32,
        J: int = 1,
        se2_embed: bool = False,
        use_learned_prox: bool = True,
        residual_scale: float = 0.5,
        tau: float = 0.5,
        sigma: float = 0.5,
        use_precond: bool = False,
        learnable_steps: bool = False,
        tau_min: float = 0.05,
        tau_max: float = 0.99,
        sigma_min: float = 0.05,
        sigma_max: float = 0.99,
    ) -> None:
        super().__init__()

        self.E = int(E)
        self.J = max(1, int(J))
        self.se2_embed = bool(se2_embed)
        self.use_learned_prox = bool(use_learned_prox)
        self.residual_scale = float(residual_scale)
        self.use_precond = bool(use_precond)
        self.learnable_steps = bool(learnable_steps)

        # Register geometry as buffers to keep them on the right device and avoid grads
        base_G = G.detach().clone()
        base_h = h.detach().clone()
        self.register_buffer("G", base_G)
        self.register_buffer("h", base_h)

        # Removed row preconditioning: operate directly on G/h

        actual_in = 3 if self.se2_embed else input_dim

        # Lightweight feature encoder (kept small for real-time)
        self.encoder = nn.Sequential(
            nn.Linear(actual_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        # Initialize mu from encoded features
        self.init_mu = nn.Sequential(
            nn.Linear(hidden, self.E),
            nn.ReLU(inplace=True),  # non-negativity at init
        )

        # Auxiliary y (lambda) kept implicit in row layout [N,2]
        self.init_y = nn.Sequential(
            nn.Linear(hidden, 2),
            nn.Tanh(),  # bounded init
        )

        # Learned-prox residual head working on z = mu @ G -> [N,2]
        self.prox_head = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.E),
        )

        tau = float(tau)
        sigma = float(sigma)

        # Keep simple scalar step sizes (no clamp/learnable vectors)
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

    @staticmethod
    def _polar_embed(x: torch.Tensor) -> torch.Tensor:
        # x: [N,2] -> [N,3] = (r, cos(theta), sin(theta))
        r = torch.norm(x, dim=1, keepdim=True)
        theta = torch.atan2(x[:, 1], x[:, 0]).unsqueeze(1)
        return torch.cat([r, torch.cos(theta), torch.sin(theta)], dim=1)

    def _project_mu_row(self, mu: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """Project row-layout mu [N,E] using provided geometry G."""
        mu = mu.clamp_min(0.0)
        v = mu @ G
        v_norm = torch.norm(v, dim=1, keepdim=True)
        mu = mu / v_norm.clamp(min=1.0)
        return mu

    def _step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        tau: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One PDHG-like row-layout step with optional learned-prox."""
        y = y + sigma * (mu @ self.G)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        y = y / y_norm.clamp(min=1.0)

        a_row = x @ self.G.t() - self.h.t()

        mu = mu + tau * (a_row - (y @ self.G.t()))

        if self.use_learned_prox:
            z = mu @ self.G
            delta = self.prox_head(z)
            mu = mu + self.residual_scale * delta

        mu = self._project_mu_row(mu, self.G)
        return y, mu

    def _prepare_step_sizes(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        tau_vals = self.tau.to(device=device, dtype=dtype).expand(self.J)
        sigma_vals = self.sigma.to(device=device, dtype=dtype).expand(self.J)
        return tau_vals, sigma_vals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N,2] obstacle points in robot frame.
        Returns:
            mu: [N,E] dual variables (row layout), feasible by construction.
        """
        if x.dim() != 2 or x.shape[1] != 2:
            raise ValueError(f"Expected x shape [N,2], got {tuple(x.shape)}")

        if self.se2_embed:
            feats_in = self._polar_embed(x)
        else:
            feats_in = x

        h = self.encoder(feats_in)
        mu = self.init_mu(h)  # [N,E] non-negative init
        y = self.init_y(h)    # [N,2] bounded init

        tau_vals, sigma_vals = self._prepare_step_sizes(x.device, x.dtype)

        steps = max(1, self.J)
        for j in range(steps):
            tau_j = tau_vals[j]
            sigma_j = sigma_vals[j]
            y, mu = self._step(x, y, mu, tau_j, sigma_j)

        # Final safety projection (idempotent if feasible already)
        mu = self._project_mu_row(mu, self.G)
        return mu
