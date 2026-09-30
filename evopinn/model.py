"""Fully connected network with tanh activations and uniform hidden width."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """MLP with `n_layers` hidden layers of `n_neurons` neurons each.

    tanh is used because PINN residuals need second derivatives of the output:
    with ReLU the network is piecewise linear and u_xx = u_tt = 0 almost everywhere.
    """

    def __init__(self, n_in: int, n_out: int, n_layers: int, n_neurons: int):
        super().__init__()
        if n_layers < 1 or n_neurons < 1:
            raise ValueError(f"invalid architecture ({n_layers}, {n_neurons})")
        layers: list[nn.Module] = []
        width = n_in
        for _ in range(n_layers):
            layers += [nn.Linear(width, n_neurons), nn.Tanh()]
            width = n_neurons
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
