
"""
ObsPointNet is a neural network structure of DUNE model. It maps each obstacle point to the latent distance feature mu.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NeuPAN planner is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuPAN planner. If not, see <https://www.gnu.org/licenses/>.
"""

import torch.nn as nn
import torch

class ObsPointNet(nn.Module):
    def __init__(self, input_dim: int = 2,  output_dim: int=4, se2_embed: bool = False) -> None:
        super(ObsPointNet, self).__init__()

        self.se2_embed = se2_embed

        hidden_dim = 32
        actual_in = 3 if se2_embed else input_dim


        self.MLP = nn.Sequential(   nn.Linear(actual_in, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, output_dim),
                                    nn.ReLU(),
                                    )

    def polar_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert Cartesian (x, y) -> polar embedding (r, cos(theta), sin(theta)).
        Expect x: [N, 2], return [N, 3].
        """
        r = torch.norm(x, dim=1, keepdim=True)
        theta = torch.atan2(x[:, 1], x[:, 0]).unsqueeze(1)
        return torch.cat([r, torch.cos(theta), torch.sin(theta)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.se2_embed:
            x = self.polar_embed(x)

        return self.MLP(x)