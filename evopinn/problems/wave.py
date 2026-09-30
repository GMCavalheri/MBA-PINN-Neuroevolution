"""1D wave equation / vibrating string (dimensionless): u_tt - c^2 u_xx = 0 on x, t in [0, 1]."""
from __future__ import annotations

import math

import torch
from torch import nn

from ..autodiff import grad
from .base import Problem, mse
from .grids import boundary_points, collocation_grid, eval_grid, initial_points


class Wave(Problem):
    name = "wave"
    n_in = 2
    term_names = ("pde", "ic", "ic_velocity", "bc")

    def setup(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        p = self.params
        self.device = device
        self.c = float(p["c"])
        nx, nt = int(p["nx"]), int(p["nt"])

        self.xt_f = collocation_grid(nx, nt, dtype, device).requires_grad_(True)
        self.xt_ic = initial_points(nx, dtype, device).requires_grad_(True)
        self.u_ic = torch.sin(math.pi * self.xt_ic[:, 0:1]).detach()
        self.xt_left, self.xt_right = boundary_points(nt, dtype, device)

        self.eval_inputs = eval_grid(int(p["n_eval"]), dtype, device)
        self.eval_exact = self.exact(self.eval_inputs)

    def exact(self, inputs: torch.Tensor) -> torch.Tensor:
        x, t = inputs[:, 0:1], inputs[:, 1:2]
        return torch.sin(math.pi * x) * torch.cos(self.c * math.pi * t)

    def _residual(self, u: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        du = grad(u, xt)
        u_x, u_t = du[:, 0:1], du[:, 1:2]
        u_xx = grad(u_x, xt)[:, 0:1]
        u_tt = grad(u_t, xt)[:, 1:2]
        return u_tt - self.c**2 * u_xx

    def residual_of_exact(self, inputs: torch.Tensor) -> torch.Tensor:
        xt = inputs.clone().requires_grad_(True)
        return self._residual(self.exact(xt), xt)

    def loss_terms(self, model: nn.Module) -> dict[str, torch.Tensor]:
        pde = mse(self._residual(model(self.xt_f), self.xt_f))
        u0 = model(self.xt_ic)
        ic = mse(u0 - self.u_ic)
        ic_velocity = mse(grad(u0, self.xt_ic)[:, 1:2])
        bc = mse(model(self.xt_left)) + mse(model(self.xt_right))
        return {"pde": pde, "ic": ic, "ic_velocity": ic_velocity, "bc": bc}
