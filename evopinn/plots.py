"""Figures for the article (PDF + PNG), generated from the saved runs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .analysis import ARMS, load_runs  # noqa: E402
from .config import load_config  # noqa: E402
from .model import MLP  # noqa: E402
from .problems import make_problem  # noqa: E402

LABELS = {
    "en": {"ga": "GA", "random": "Random", "random_search": "Random search",
           "pendulum": "Pendulum", "heat": "Heat conduction", "wave": "Vibrating string",
           "l2": "Relative $L^2$ error", "epochs": "Epochs", "loss": "Total loss",
           "exact": "Exact", "pred": "PINN (GA)", "err": "|error|", "data": "Data"},
    "pt": {"ga": "AG", "random": "Aleatória", "random_search": "Busca aleatória",
           "pendulum": "Pêndulo", "heat": "Condução de calor", "wave": "Corda vibrante",
           "l2": "Erro $L^2$ relativo", "epochs": "Épocas", "loss": "Custo total",
           "exact": "Exata", "pred": "PINN (AG)", "err": "|erro|", "data": "Dados"},
}
COLORS = {"ga": "#5B21B6", "random": "#9CA3AF", "random_search": "#D97706"}
PROBLEMS = ("pendulum", "heat", "wave")

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9,
                     "savefig.bbox": "tight", "savefig.dpi": 200})


def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png")
    plt.close(fig)


def median_run(runs: list[dict], arm: str = "ga") -> dict:
    """The run whose `arm` relative L2 error is the median (lower median for even n)."""
    order = sorted(runs, key=lambda r: r["final"][arm]["rel_l2"])
    return order[(len(order) - 1) // 2]


def fig_error_boxplot(results: Path, outdir: Path, lang: str) -> None:
    L = LABELS[lang]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    rng = np.random.default_rng(0)
    for ax, prob in zip(axes, PROBLEMS):
        runs = load_runs(results, prob)
        if not runs:
            ax.set_visible(False)
            continue
        data = [[r["final"][arm]["rel_l2"] for r in runs] for arm in ARMS]
        bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True)
        for patch, arm in zip(bp["boxes"], ARMS):
            patch.set_facecolor(COLORS[arm])
            patch.set_alpha(0.35)
        for i, (vals, arm) in enumerate(zip(data, ARMS), 1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=6, color=COLORS[arm], zorder=3)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 3], [L[a] for a in ARMS], rotation=15)
        ax.set_title(L[prob])
    axes[0].set_ylabel(L["l2"])
    _save(fig, outdir, f"fig_error_boxplot_{lang}")


def fig_loss_curves(results: Path, outdir: Path, lang: str) -> None:
    L = LABELS[lang]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, prob in zip(axes, PROBLEMS):
        runs = load_runs(results, prob)
        if not runs:
            ax.set_visible(False)
            continue
        run = median_run(runs)
        hist = np.load(results / prob / f"run_{run['run']:02d}_hist.npz")
        for arm in ARMS:
            arch = run["final"][arm]["arch"]
            ax.plot(hist[f"final_{arm}"][:, 0], lw=0.8, color=COLORS[arm],
                    label=f"{L[arm]} ({arch[0]}×{arch[1]})")
        ax.set_yscale("log")
        ax.set_xlabel(L["epochs"])
        ax.set_title(f"{L[prob]} (run {run['run']})")
        ax.legend(fontsize=6, frameon=False)
    axes[0].set_ylabel(L["loss"])
    _save(fig, outdir, f"fig_loss_curves_{lang}")


def _load_model(results: Path, prob: str, run: int, arm: str, n_in: int) -> MLP:
    saved = torch.load(results / prob / f"run_{run:02d}_models.pt", map_location="cpu")[arm]
    model = MLP(n_in, 1, *saved["arch"])
    model.load_state_dict(saved["state_dict"])
    return model.eval()


@torch.no_grad()
def fig_solutions(results: Path, outdir: Path, lang: str) -> None:
    """Median GA run: pendulum curve; heat and wave exact / prediction / error maps."""
    L = LABELS[lang]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])

    prob = "pendulum"
    runs = load_runs(results, prob)
    if runs:
        run = median_run(runs)
        p = make_problem(load_config(prob))
        p.setup(torch.device("cpu"))
        model = _load_model(results, prob, run["run"], "ga", 1)
        t = p.eval_inputs
        ax = fig.add_subplot(gs[0, :])
        ax.plot(t, p.eval_exact, color="black", lw=1.0, label=L["exact"])
        ax.plot(t, model(t), color=COLORS["ga"], lw=1.0, ls="--", label=L["pred"])
        ax.scatter(p.t_data, p.x_data, s=12, color="#D97706", zorder=3, label=L["data"])
        ax.set_title(f"{L[prob]} (run {run['run']}, $L^2$ = {run['final']['ga']['rel_l2']:.2e})")
        ax.set_xlabel("t")
        ax.legend(fontsize=7, frameon=False, ncol=3)

    for row, prob in ((1, "heat"), (2, "wave")):
        runs = load_runs(results, prob)
        if not runs:
            continue
        run = median_run(runs)
        cfg = load_config(prob)
        p = make_problem(cfg)
        p.setup(torch.device("cpu"))
        model = _load_model(results, prob, run["run"], "ga", 2)
        n = int(cfg["problem"]["n_eval"])
        exact = p.eval_exact.view(n, n).numpy()
        pred = model(p.eval_inputs).view(n, n).numpy()
        extent = (0, 1, 0, 1)
        panels = ((exact, L["exact"], "RdBu_r"), (pred, L["pred"], "RdBu_r"), (np.abs(pred - exact), L["err"], "magma"))
        for col, (img, title, cmap) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            vmax = np.abs(exact).max() if col < 2 else None
            im = ax.imshow(img.T, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                           vmin=-vmax if vmax else None, vmax=vmax)
            ax.set_title(f"{L[prob]}: {title}" if col == 0 else title)
            ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("t")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    _save(fig, outdir, f"fig_solutions_{lang}")
