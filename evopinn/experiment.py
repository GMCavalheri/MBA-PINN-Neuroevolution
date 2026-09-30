"""One experimental run: three arms, each ending with a full training of its chosen architecture.

Arms
  ga            single-generation GA (article's method), search budget 30,000 epochs
  random        one architecture drawn at random (the monograph's baseline), no search
  random_search best of 10 random architectures trained for 3,000 epochs (same budget as the GA)
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from .baselines import random_architecture, run_random_search
from .config import config_hash
from .ga import SearchOutcome, run_ga
from .problems import make_problem
from .seed import as_int, enable_determinism
from .train import TrainResult, train

ARMS = ("ga", "random", "random_search")


def environment_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "platform": platform.platform(), "device": str(device),
    }
    if device.type == "cuda":
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(device)
    else:
        info["cpu"] = platform.processor() or "unknown"
    return info


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def scale_epochs(cfg: dict, factor: float) -> dict:
    """Copy of `cfg` with every epoch count multiplied by `factor` (smoke tests only)."""
    import copy
    c = copy.deepcopy(cfg)
    s = lambda v: max(1, int(round(v * factor)))  # noqa: E731
    c["ga"]["epochs_screen"] = s(c["ga"]["epochs_screen"])
    c["ga"]["epochs_reeval"] = s(c["ga"]["epochs_reeval"])
    c["random_search"]["epochs"] = s(c["random_search"]["epochs"])
    c["epochs_full"] = s(c["epochs_full"])
    c["fitness_window"] = min(c["fitness_window"], c["ga"]["epochs_screen"])
    return c


def _search_record(outcome: SearchOutcome) -> dict:
    return {
        "best": list(outcome.best), "epochs_used": outcome.epochs_used,
        "selected": [list(a) for a in outcome.selected],
        "descendants": [list(a) for a in outcome.descendants],
        "screening": [r.summary() for r in outcome.screening],
        "reevaluation": [r.summary() for r in outcome.reevaluation],
    }


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def run_path(outdir: Path, run: int) -> Path:
    return outdir / f"run_{run:02d}.json"


def run_round(cfg: dict, run: int, outdir: Path, amp: bool, device: torch.device) -> dict:
    """Execute run `run` and save run_XX.json (+ histories .npz, final models .pt)."""
    enable_determinism()
    outdir.mkdir(parents=True, exist_ok=True)
    problem = make_problem(cfg)
    problem.setup(device)
    lr, window, e_full = float(cfg["lr"]), int(cfg["fitness_window"]), int(cfg["epochs_full"])

    master = int(cfg["master_seed_offset"]) + run
    seq_ga, seq_rand, seq_rs, seq_full = np.random.SeedSequence(master).spawn(4)
    full_seeds = dict(zip(ARMS, (as_int(s) for s in seq_full.spawn(3))))

    ga = run_ga(problem, cfg, seq_ga, amp)
    rs = run_random_search(problem, cfg, seq_rs, amp)
    chosen = {"ga": ga.best, "random": random_architecture(cfg, seq_rand), "random_search": rs.best}

    finals: dict[str, TrainResult] = {
        arm: train(problem, chosen[arm], e_full, full_seeds[arm], lr, amp, window, keep_state=True)
        for arm in ARMS
    }

    record = {
        "problem": cfg["name"], "run": run, "master_seed": master, "amp": amp,
        "config_hash": config_hash(cfg), "config": cfg, "environment": environment_info(device),
        "search": {"ga": _search_record(ga), "random_search": _search_record(rs)},
        "final": {arm: finals[arm].summary() for arm in ARMS},
    }

    stem = f"run_{run:02d}"
    hist = {f"final_{arm}": finals[arm].history for arm in ARMS}
    for name, outcome in (("ga", ga), ("random_search", rs)):
        for stage in ("screening", "reevaluation"):
            for i, r in enumerate(getattr(outcome, stage)):
                hist[f"{name}_{stage}_{i}"] = r.history[:, 0]
    def save_npz(path: Path) -> None:
        with open(path, "wb") as f:
            np.savez_compressed(f, **hist)

    _atomic_write(outdir / f"{stem}_hist.npz", save_npz)
    _atomic_write(outdir / f"{stem}_models.pt",
                  lambda p: torch.save({arm: {"arch": list(finals[arm].arch), "state_dict": finals[arm].state_dict}
                                        for arm in ARMS}, p))
    # The JSON is written last: its presence marks the run as complete.
    _atomic_write(run_path(outdir, run),
                  lambda p: p.write_text(json.dumps(record, indent=1), encoding="utf-8"))
    return record


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use scripts/run_experiment.py")
