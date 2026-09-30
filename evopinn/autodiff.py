"""Derivatives of the network output with respect to its inputs (automatic differentiation)."""
from __future__ import annotations

import torch


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx for every input column, keeping the graph for higher-order derivatives."""
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]
