"""Undamped simple pendulum (small-angle approximation): x'' + w0^2 x = 0 on t in [0, 1]."""
from __future__ import annotations

import math

import torch
from torch import nn

from ..autodiff import grad
from .base import Problem, mse


class Pendulum(Problem):
    name = "pendulum"
    n_in = 1
    term_names = ("data", "physics")

    def setup(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        p = self.params
        self.device = device
        self.w0 = float(p["omega0"])
        self.A = float(p["amplitude"])
        self.delta = float(p["phase"])
        self.lam = float(p["physics_weight"])

        grid = torch.linspace(0.0, 1.0, int(p["n_grid"]), dtype=dtype).view(-1, 1)
        start, stop, step = p["data_slice"]
        self.t_data = grid[start:stop:step].to(device)
        self.x_data = self.exact(self.t_data)
        self.t_phys = torch.linspace(0.0, 1.0, int(p["n_collocation"]), dtype=dtype, device=device).view(-1, 1)
        self.t_phys.requires_grad_(True)

        self.eval_inputs = torch.linspace(0.0, 1.0, int(p["n_eval"]), dtype=dtype, device=device).view(-1, 1)
        self.eval_exact = self.exact(self.eval_inputs)

    def exact(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.A * torch.sin(self.w0 * inputs - self.delta)

    def residual_of_exact(self, inputs: torch.Tensor) -> torch.Tensor:
        t = inputs.clone().requires_grad_(True)
        x = self.exact(t)
        x_t = grad(x, t)
        x_tt = grad(x_t, t)
        return x_tt + self.w0**2 * x

    def loss_terms(self, model: nn.Module) -> dict[str, torch.Tensor]:
        data = mse(model(self.t_data) - self.x_data)
        x = model(self.t_phys)
        x_t = grad(x, self.t_phys)
        x_tt = grad(x_t, self.t_phys)
        physics = self.lam * mse(x_tt + self.w0**2 * x)
        return {"data": data, "physics": physics}


def period(w0: float) -> float:
    return 2 * math.pi / w0
