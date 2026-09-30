import numpy as np
import torch

from evopinn.baselines import random_architecture, run_random_search
from evopinn.config import load_config
from evopinn.experiment import scale_epochs
from evopinn.ga import mutate, run_ga, sample_arch
from evopinn.problems import make_problem


def small_setup():
    cfg = scale_epochs(load_config("pendulum"), 0.01)  # 10 / 20 / 30 / 200 epochs
    problem = make_problem(cfg)
    problem.setup(torch.device("cpu"))
    return cfg, problem


def test_sample_and_mutate_respect_bounds():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        L, n = sample_arch(rng, (1, 10), (1, 20))
        assert 1 <= L <= 10 and 1 <= n <= 20
        L2, n2 = mutate((L, n), rng, [0, 1], [0, 1, 2])
        assert L2 - L in (0, 1) and n2 - n in (0, 1, 2)


def test_ga_population_selection_and_elitism():
    cfg, problem = small_setup()
    out = run_ga(problem, cfg, np.random.SeedSequence(1), amp=False)
    ga = cfg["ga"]
    assert len(out.screening) == ga["population"]
    assert len(out.selected) == ga["selected"] and len(out.descendants) == ga["selected"]
    p1 = [tuple(r.arch) for r in out.reevaluation]
    assert len(p1) == ga["population"]
    assert p1[: ga["selected"]] == out.selected          # the K selected survive unchanged
    fitness = [r.final_loss for r in out.screening]
    worst_selected = max(r.final_loss for r in out.screening if r.arch in out.selected)
    assert sum(f < worst_selected for f in fitness) <= ga["selected"]  # truncation keeps the best K
    assert out.best in p1
    assert out.epochs_used == ga["population"] * ga["epochs_screen"] + len(p1) * ga["epochs_reeval"]


def test_same_seed_same_search():
    cfg, problem = small_setup()
    a = run_ga(problem, cfg, np.random.SeedSequence(7), amp=False)
    b = run_ga(problem, cfg, np.random.SeedSequence(7), amp=False)
    assert a.best == b.best
    assert [r.final_loss for r in a.reevaluation] == [r.final_loss for r in b.reevaluation]


def test_random_search_budget_and_random_arch():
    cfg, problem = small_setup()
    rs = run_random_search(problem, cfg, np.random.SeedSequence(3), amp=False)
    assert rs.epochs_used == cfg["random_search"]["candidates"] * cfg["random_search"]["epochs"]
    assert rs.best in [r.arch for r in rs.screening]
    L, n = random_architecture(cfg, np.random.SeedSequence(3))
    assert 1 <= L <= 10 and 1 <= n <= 20


def test_full_budgets_are_equal():
    cfg = load_config("heat")
    ga, rs = cfg["ga"], cfg["random_search"]
    ga_budget = ga["population"] * ga["epochs_screen"] + ga["population"] * ga["epochs_reeval"]
    assert ga_budget == rs["candidates"] * rs["epochs"] == 30000
