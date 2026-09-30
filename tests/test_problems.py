import math

import pytest
import torch

from evopinn.config import load_config
from evopinn.problems import make_problem

DEV = torch.device("cpu")


def build(name, dtype=torch.float64):
    p = make_problem(load_config(name))
    p.setup(DEV, dtype)
    return p


@pytest.mark.parametrize("name", ["pendulum", "heat", "wave"])
def test_exact_solution_satisfies_equation(name):
    p = build(name)
    pts = p.eval_inputs[::97].detach()
    assert p.residual_of_exact(pts).abs().max() < 1e-8


@pytest.mark.parametrize("name", ["heat", "wave"])
def test_initial_and_boundary_conditions(name):
    p = build(name)
    torch.testing.assert_close(p.exact(p.xt_ic.detach()), p.u_ic)
    assert p.exact(p.xt_left).abs().max() < 1e-12
    assert p.exact(p.xt_right).abs().max() < 1e-12


def test_wave_initial_velocity_is_zero():
    p = build("wave")
    xt = p.xt_ic.detach().clone().requires_grad_(True)
    u = p.exact(xt)
    u_t = torch.autograd.grad(u, xt, torch.ones_like(u))[0][:, 1]
    assert u_t.abs().max() < 1e-12


def test_point_counts_match_the_monograph():
    heat = build("heat", torch.float32)
    assert heat.xt_f.shape == (100 * 250, 2)
    assert heat.xt_ic.shape == (100, 2) and heat.xt_left.shape == (250, 2)
    assert heat.eval_inputs.shape == (201 * 201, 2)
    pend = build("pendulum", torch.float32)
    assert pend.t_data.shape == (10, 1) and float(pend.t_data.max()) < 0.4
    assert pend.t_phys.shape == (30, 1)


def test_heat_decay_value():
    p = build("heat")
    u = p.exact(torch.tensor([[0.5, 1.0]], dtype=torch.float64))
    assert abs(float(u) - math.exp(-0.05 * math.pi**2)) < 1e-12


@pytest.mark.parametrize("name", ["pendulum", "heat", "wave"])
def test_loss_terms_are_finite_scalars(name):
    from evopinn.model import MLP
    p = build(name, torch.float32)
    torch.manual_seed(0)
    terms = p.loss_terms(MLP(p.n_in, 1, 2, 5))
    assert tuple(terms) == p.term_names
    assert all(t.ndim == 0 and torch.isfinite(t) for t in terms.values())
