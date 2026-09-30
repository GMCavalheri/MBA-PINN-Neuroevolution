import numpy as np
import pytest
import torch

from evopinn.config import load_config
from evopinn.problems import make_problem
from evopinn.seed import enable_determinism
from evopinn.train import train

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@cuda
@pytest.mark.parametrize("name", ["pendulum", "heat", "wave"])
def test_cuda_graph_reproducible_and_consistent_with_eager(name):
    enable_determinism()
    cfg = load_config(name)
    p = make_problem(cfg)
    p.setup(torch.device("cuda"))
    a = train(p, (3, 12), 100, seed=9, lr=cfg["lr"])
    b = train(p, (3, 12), 100, seed=9, lr=cfg["lr"])
    e = train(p, (3, 12), 100, seed=9, lr=cfg["lr"], cuda_graph=False)
    assert a.mode == "cuda_graph" and e.mode == "eager_cuda"
    assert np.array_equal(a.history, b.history)
    np.testing.assert_allclose(a.history, e.history, rtol=1e-3, atol=1e-6)


@cuda
def test_no_gpu_memory_leak_between_trainings():
    """Regression guard: repeated CUDA-graph trainings must not retain memory."""
    enable_determinism()
    cfg = load_config("wave")
    p = make_problem(cfg)
    p.setup(torch.device("cuda"))
    train(p, (4, 16), 10, seed=0, lr=cfg["lr"])
    before = torch.cuda.memory_allocated()
    for s in range(5):
        train(p, (4, 16), 10, seed=s, lr=cfg["lr"])
    assert torch.cuda.memory_allocated() - before < 1 * 2**20
