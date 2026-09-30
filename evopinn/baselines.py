"""Reference arms: a single random architecture, and random search with the GA budget."""
from __future__ import annotations

import numpy as np

from .ga import SearchOutcome, rank, sample_arch
from .problems.base import Problem
from .seed import as_int, rng_from
from .train import Arch, train


def random_architecture(cfg: dict, seq: np.random.SeedSequence) -> Arch:
    """One architecture drawn with the same bounds as the GA initial population."""
    ga = cfg["ga"]
    return sample_arch(rng_from(seq), ga["layers_range"], ga["neurons_range"])


def run_random_search(problem: Problem, cfg: dict, seq: np.random.SeedSequence, amp: bool) -> SearchOutcome:
    """Random search with the same epoch budget as the GA (default 10 x 3000 = 30000 epochs)."""
    ga, rs = cfg["ga"], cfg["random_search"]
    lr, window = float(cfg["lr"]), int(cfg["fitness_window"])
    n, epochs = int(rs["candidates"]), int(rs["epochs"])
    seq_arch, seq_train = seq.spawn(2)
    rng = rng_from(seq_arch)

    candidates = [sample_arch(rng, ga["layers_range"], ga["neurons_range"]) for _ in range(n)]
    seeds = [as_int(s) for s in seq_train.spawn(n)]
    results = [train(problem, a, epochs, s, lr, amp, window) for a, s in zip(candidates, seeds)]
    best = candidates[rank(results)[0]]
    return SearchOutcome(best=best, screening=results, reevaluation=[], selected=[],
                         descendants=[], epochs_used=n * epochs)
