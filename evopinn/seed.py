"""Seeding and determinism helpers.

Every random decision is derived from the master seed of a run through
numpy.random.SeedSequence, so any run can be reproduced in isolation.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def enable_determinism() -> None:
    """Deterministic kernels (CPU bit-exact; GPU reproducible on the same hardware/software)."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)  # also seeds every CUDA device


def spawn(seq: np.random.SeedSequence, n: int) -> list[np.random.SeedSequence]:
    return seq.spawn(n)


def as_int(seq: np.random.SeedSequence) -> int:
    """Integer seed (for torch) derived from a SeedSequence."""
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def rng_from(seq: np.random.SeedSequence) -> np.random.Generator:
    return np.random.default_rng(seq)
