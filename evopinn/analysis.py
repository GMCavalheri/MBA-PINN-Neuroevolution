"""Statistical analysis of the completed runs.

For each problem and each comparison (GA vs random, GA vs random search) and each metric
(relative L2 error = primary, final loss = secondary):
  * wins of the GA (strictly lower value), ties;
  * one-sided exact binomial test (H0: win probability 0.5) and Wilson 95% interval;
  * one-sided Wilcoxon signed-rank test on the paired log-ratios log(other / GA);
  * median ratio other / GA with a bootstrap 95% interval;
  * Holm correction over the family of comparisons of each metric.
Also: Spearman correlation between partial-training fitness and error (is the loss a good proxy?).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ("ga", "random", "random_search")
COMPARISONS = (("ga", "random"), ("ga", "random_search"))
METRICS = ("rel_l2", "final_loss")


def load_runs(results_dir: Path, problem: str) -> list[dict]:
    files = sorted((results_dir / problem).glob("run_[0-9][0-9].json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def bootstrap_median(x: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return (float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975)))


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """GA values `a` vs reference values `b` (lower is better)."""
    n = int(a.size)
    wins, ties = int(np.sum(a < b)), int(np.sum(a == b))
    ratio = b / a
    log_ratio = np.log(ratio)
    try:
        wilcoxon_p = float(stats.wilcoxon(log_ratio, alternative="greater").pvalue)
    except ValueError:
        wilcoxon_p = math.nan
    return {
        "n": n, "wins": wins, "ties": ties, "win_rate": wins / n,
        "wilson_95": wilson(wins, n),
        "binomial_p": float(stats.binomtest(wins, n, 0.5, alternative="greater").pvalue),
        "wilcoxon_p": wilcoxon_p,
        "median_ratio": float(np.median(ratio)),
        "median_ratio_95": bootstrap_median(ratio),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
    }


def analyse_problem(runs: list[dict]) -> dict:
    values = {m: {arm: np.array([r["final"][arm][m] for r in runs], dtype=float) for arm in ARMS}
              for m in METRICS}
    out = {"n_runs": len(runs),
           "diverged": {arm: int(sum(r["final"][arm]["diverged"] for r in runs)) for arm in ARMS},
           "median": {m: {arm: float(np.median(values[m][arm])) for arm in ARMS} for m in METRICS},
           "comparisons": {}}
    for m in METRICS:
        for a, b in COMPARISONS:
            out["comparisons"][f"{m}:{a}_vs_{b}"] = compare(values[m][a], values[m][b])

    # Is the partial-training loss a good proxy for the error? (GA screening, all runs pooled)
    fit, err = [], []
    for r in runs:
        for rec in r["search"]["ga"]["screening"]:
            if math.isfinite(rec["final_loss"]) and math.isfinite(rec["rel_l2"]):
                fit.append(rec["final_loss"])
                err.append(rec["rel_l2"])
    rho = stats.spearmanr(fit, err)
    out["proxy_spearman_screening"] = {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": len(fit)}
    final_loss = np.concatenate([values["final_loss"][arm] for arm in ARMS])
    final_err = np.concatenate([values["rel_l2"][arm] for arm in ARMS])
    rho_f = stats.spearmanr(final_loss, final_err)
    out["proxy_spearman_final"] = {"rho": float(rho_f.statistic), "p": float(rho_f.pvalue), "n": int(final_loss.size)}

    out["architectures"] = {arm: [r["final"][arm]["arch"] for r in runs] for arm in ARMS}
    out["seconds_per_run"] = float(np.median([
        sum(x["seconds"] for x in r["search"]["ga"]["screening"] + r["search"]["ga"]["reevaluation"]
            + r["search"]["random_search"]["screening"]) + sum(r["final"][arm]["seconds"] for arm in ARMS)
        for r in runs]))
    return out


def analyse(results_dir: Path, problems=("pendulum", "heat", "wave")) -> dict:
    summary = {p: analyse_problem(load_runs(results_dir, p)) for p in problems
               if (results_dir / p).exists() and load_runs(results_dir, p)}
    for m in METRICS:
        for test in ("binomial_p", "wilcoxon_p"):
            keys = [(p, f"{m}:{a}_vs_{b}") for p in summary for a, b in COMPARISONS]
            adj = holm([summary[p]["comparisons"][k][test] for p, k in keys])
            for (p, k), v in zip(keys, adj):
                summary[p]["comparisons"][k][f"{test}_holm"] = v
    return summary
