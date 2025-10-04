"""
Lightweight learned proximal head for DUNE.
Applies a tiny MLP to features derived from mu and geometry G to refine mu
before hard projection. Designed to be optional and fast.
"""
from typing import Optional
import torch
import torch.nn as nn


class ProxHead(nn.Module):
    """
    A minimal proximal refinement head.
    Given mu in R^{E x N}, computes z = (G^T mu)^T in R^{N x 2},
    passes through a tiny MLP to predict a residual delta in R^{N x E},
    and returns refined mu' = clamp_min(mu + delta^T, 0).

    Notes:
    - Final hard projection should still be applied downstream to enforce ||G^T mu||_2 <= 1.
    - This head is intentionally small to preserve real-time performance.
    """

    def __init__(self, E: int, hidden: int = 32) -> None:
        super().__init__()
        self.E = int(E)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.E),
        )

    @torch.no_grad()
    def _ensure_2d(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Ensure mu has shape [E, N]. Accepts [E], [E, 1], [E, N], [E, 1, 1].
        Returns a 2D tensor [E, N].
        """
        if mu.dim() == 1:
            return mu.unsqueeze(1)
        if mu.dim() == 2:
            return mu
        if mu.dim() == 3:
            # squeeze singleton dims at the end
            return mu.squeeze(-1)
        raise ValueError(f"Unsupported mu shape: {tuple(mu.shape)}")

    def forward(self, mu: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        Supports two layouts for mu:
        - column layout [E, N]: computes z = (G^T mu)^T with shape [N, 2]
        - row layout    [N, E]: computes z = mu @ G         with shape [N, 2]
        Returns refined mu in the SAME layout as input (clamped at 0).
        """
        device = mu.device
        mu2d = self._ensure_2d(mu)
        E = self.E

        if mu2d.shape[0] == E:  # [E, N] column layout
            z = (G.t() @ mu2d).t()             # [N, 2]
            delta = self.mlp(z)                # [N, E]
            mu_ref = (mu2d.t() + delta).t()    # [E, N]
            return mu_ref.clamp_min(0.0).to(device)

        if mu2d.shape[1] == E:  # [N, E] row layout (batch first)
            z = mu2d @ G                        # [N, 2]
            delta = self.mlp(z)                 # [N, E]
            mu_ref = (mu2d + delta)             # [N, E]
            return mu_ref.clamp_min(0.0).to(device)

        raise ValueError(f"ProxHead: unsupported mu shape {tuple(mu2d.shape)}; expected [E,N] or [N,E] with E={E}")
