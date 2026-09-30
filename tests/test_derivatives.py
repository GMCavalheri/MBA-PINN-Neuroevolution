import math

import torch
from torch import nn

from evopinn.autodiff import grad
from evopinn.model import MLP


def test_autograd_matches_analytic_derivatives():
    xt = torch.rand(200, 2, dtype=torch.float64, requires_grad=True)
    x, t = xt[:, 0:1], xt[:, 1:2]
    u = torch.sin(math.pi * x) * torch.exp(-t)
    du = grad(u, xt)
    u_xx = grad(du[:, 0:1], xt)[:, 0:1]
    torch.testing.assert_close(du[:, 0:1], math.pi * torch.cos(math.pi * x) * torch.exp(-t))
    torch.testing.assert_close(du[:, 1:2], -u)
    torch.testing.assert_close(u_xx, -math.pi**2 * u)


def _second_derivative(model):
    xt = torch.rand(500, 2, requires_grad=True)
    u = model(xt)
    u_x = grad(u, xt)[:, 0:1]
    return grad(u_x, xt)[:, 0:1]


def test_tanh_network_has_nonzero_second_derivative():
    torch.manual_seed(0)
    assert _second_derivative(MLP(2, 1, 3, 20)).abs().max() > 1e-4


def test_relu_network_second_derivative_vanishes():
    """Regression guard for the monograph's bug: with ReLU, u_xx is identically zero."""
    torch.manual_seed(0)
    layers = [nn.Linear(2, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 1)]
    assert _second_derivative(nn.Sequential(*layers)).abs().max() == 0.0


def test_mlp_architecture_and_uniform_width():
    model = MLP(2, 1, n_layers=4, n_neurons=7)
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    assert len(linears) == 5
    assert [l.out_features for l in linears[:-1]] == [7, 7, 7, 7]
    assert all(isinstance(m, nn.Tanh) for m in model.net if not isinstance(m, nn.Linear))
