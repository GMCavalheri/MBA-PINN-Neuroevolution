"""Single-generation, mutation-only Genetic Algorithm for the PINN architecture.

Faithful to the method of the article:
  1. random initial population P0 of M architectures (L, n);
  2. screening: train each for E_fit epochs, fitness = smoothed final total loss;
  3. truncation selection of the K best;
  4. each selected architecture produces one descendant: (L + dL, n + dn),
     dL in {0, 1}, dn in {0, 1, 2} (the same width change is applied to every hidden layer);
  5. elitist replacement: P1 = selected + descendants (size M);
  6. re-evaluation: train each member of P1 from scratch for E_term epochs;
  7. A* = member of P1 with the lowest fitness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problems.base import Problem
from .seed import as_int, rng_from
from .train import Arch, TrainResult, train


def sample_arch(rng: np.random.Generator, layers_range, neurons_range) -> Arch:
    L = int(rng.integers(layers_range[0], layers_range[1] + 1))
    n = int(rng.integers(neurons_range[0], neurons_range[1] + 1))
    return (L, n)


def mutate(arch: Arch, rng: np.random.Generator, d_layers, d_neurons) -> Arch:
    return (arch[0] + int(rng.choice(d_layers)), arch[1] + int(rng.choice(d_neurons)))


def rank(results: list[TrainResult]) -> list[int]:
    """Indices sorted by fitness (ties keep the original order)."""
    return sorted(range(len(results)), key=lambda i: results[i].final_loss)


@dataclass
class SearchOutcome:
    best: Arch
    screening: list[TrainResult]
    reevaluation: list[TrainResult]
    selected: list[Arch]
    descendants: list[Arch]
    epochs_used: int


def run_ga(problem: Problem, cfg: dict, seq: np.random.SeedSequence, amp: bool) -> SearchOutcome:
    ga, lr, window = cfg["ga"], float(cfg["lr"]), int(cfg["fitness_window"])
    M, K = int(ga["population"]), int(ga["selected"])
    seq_arch, seq_screen, seq_reeval = seq.spawn(3)
    rng = rng_from(seq_arch)

    population = [sample_arch(rng, ga["layers_range"], ga["neurons_range"]) for _ in range(M)]
    screen_seeds = [as_int(s) for s in seq_screen.spawn(M)]
    screening = [train(problem, a, int(ga["epochs_screen"]), s, lr, amp, window)
                 for a, s in zip(population, screen_seeds)]

    selected = [population[i] for i in rank(screening)[:K]]
    descendants = [mutate(a, rng, ga["mutation_layers"], ga["mutation_neurons"]) for a in selected]
    p1 = selected + descendants

    reeval_seeds = [as_int(s) for s in seq_reeval.spawn(len(p1))]
    reevaluation = [train(problem, a, int(ga["epochs_reeval"]), s, lr, amp, window)
                    for a, s in zip(p1, reeval_seeds)]
    best = p1[rank(reevaluation)[0]]

    epochs_used = M * int(ga["epochs_screen"]) + len(p1) * int(ga["epochs_reeval"])
    return SearchOutcome(best, screening, reevaluation, selected, descendants, epochs_used)
