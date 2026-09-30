"""Common interface of the physical problems."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Problem(ABC):
    """A PINN problem: training points, loss terms, exact solution and evaluation grid.

    `loss_terms` returns the *weighted* terms, so the total loss is their plain sum.
    """

    name: str
    n_in: int
    term_names: tuple[str, ...]

    def __init__(self, params: dict):
        self.params = params
        self.device = torch.device("cpu")

    @abstractmethod
    def setup(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        """Create all tensors on `device`."""

    @abstractmethod
    def loss_terms(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Weighted loss terms (scalars) of the physics-informed loss."""

    @abstractmethod
    def exact(self, inputs: torch.Tensor) -> torch.Tensor:
        """Analytical solution at `inputs` (shape [N, n_in]) -> [N, 1]."""

    @abstractmethod
    def residual_of_exact(self, inputs: torch.Tensor) -> torch.Tensor:
        """Residual of the differential equation evaluated on the exact solution (for tests)."""

    # Evaluation --------------------------------------------------------------
    eval_inputs: torch.Tensor
    eval_exact: torch.Tensor

    @torch.no_grad()
    def relative_l2(self, model: nn.Module) -> float:
        pred = model(self.eval_inputs).float()
        ref = self.eval_exact.float()
        return float(torch.linalg.vector_norm(pred - ref) / torch.linalg.vector_norm(ref))


def mse(x: torch.Tensor) -> torch.Tensor:
    """Mean squared value, computed in float32 (safe under fp16 autocast)."""
    return (x.float() ** 2).mean()
