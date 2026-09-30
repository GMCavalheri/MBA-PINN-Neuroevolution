"""Point sets shared by the 1D space-time problems on [0, 1] x [0, 1]."""
from __future__ import annotations

import torch


def collocation_grid(nx: int, nt: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Uniform nx x nt grid of (x, t) pairs, as in the monograph (100 x 250 = 25,000 points)."""
    x = torch.linspace(0.0, 1.0, nx, dtype=dtype)
    t = torch.linspace(0.0, 1.0, nt, dtype=dtype)
    X, T = torch.meshgrid(x, t, indexing="ij")
    return torch.stack([X.reshape(-1), T.reshape(-1)], dim=1).to(device)


def initial_points(nx: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx, dtype=dtype)
    return torch.stack([x, torch.zeros_like(x)], dim=1).to(device)


def boundary_points(nt: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0.0, 1.0, nt, dtype=dtype)
    left = torch.stack([torch.zeros_like(t), t], dim=1).to(device)
    right = torch.stack([torch.ones_like(t), t], dim=1).to(device)
    return left, right


def eval_grid(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Independent n x n evaluation grid for the relative L2 error."""
    return collocation_grid(n, n, dtype, device)
