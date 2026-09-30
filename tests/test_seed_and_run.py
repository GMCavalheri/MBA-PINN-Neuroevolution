import json

import numpy as np
import torch

from evopinn.config import load_config
from evopinn.experiment import ARMS, run_round, scale_epochs
from evopinn.problems import make_problem
from evopinn.train import train


def test_same_seed_identical_history_on_cpu():
    cfg = load_config("heat")
    p = make_problem(cfg)
    p.setup(torch.device("cpu"))
    a = train(p, (2, 8), 30, seed=123, lr=1e-3)
    b = train(p, (2, 8), 30, seed=123, lr=1e-3)
    c = train(p, (2, 8), 30, seed=124, lr=1e-3)
    assert np.array_equal(a.history, b.history)
    assert not np.array_equal(a.history, c.history)
    assert a.history.shape == (30, 1 + len(p.term_names))


def test_smoke_round_writes_complete_record(tmp_path):
    cfg = scale_epochs(load_config("wave"), 0.002)  # a few epochs per training
    cfg["problem"].update(nx=20, nt=20, n_eval=21)
    rec = run_round(cfg, 0, tmp_path, amp=False, device=torch.device("cpu"))
    saved = json.loads((tmp_path / "run_00.json").read_text())
    assert saved["master_seed"] == 1000 and set(saved["final"]) == set(ARMS)
    for arm in ARMS:
        f = saved["final"][arm]
        assert f["epochs"] == cfg["epochs_full"] and np.isfinite(f["rel_l2"])
    hist = np.load(tmp_path / "run_00_hist.npz")
    assert hist["final_ga"].shape[0] == cfg["epochs_full"]
    models = torch.load(tmp_path / "run_00_models.pt")
    assert set(models) == set(ARMS)
    assert rec["search"]["ga"]["epochs_used"] == rec["search"]["random_search"]["epochs_used"]
