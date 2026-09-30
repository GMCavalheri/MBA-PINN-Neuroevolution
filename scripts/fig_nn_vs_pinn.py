"""Motivating figure: standard network (data loss only) vs PINN on the pendulum, same architecture.

As in the monograph: 3 hidden layers of 32 neurons, tanh, 10 observations with t < 0.4,
standard network trained for 1,000 epochs (Adam, lr 1e-3), PINN for 20,000 epochs (Adam, lr 1e-4,
physics weight 1e-4, 30 collocation points). Runs on the CPU (seed 2026).

  python scripts/fig_nn_vs_pinn.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from evopinn.config import load_config  # noqa: E402
from evopinn.model import MLP  # noqa: E402
from evopinn.problems import make_problem  # noqa: E402
from evopinn.seed import enable_determinism, seed_everything  # noqa: E402

SEED = 2026
TEXT = {
    "en": {"nn": "Standard network (data loss only, {e} epochs)", "pinn": "PINN (data + physics loss, {e} epochs)",
           "exact": "Exact solution", "pred": "Network prediction", "data": "Training data",
           "phys": "Collocation points"},
    "pt": {"nn": "Rede padrão (só custo dos dados, {e} épocas)", "pinn": "PINN (custo dos dados + físico, {e} épocas)",
           "exact": "Solução exata", "pred": "Predição da rede", "data": "Dados de treino",
           "phys": "Pontos de colocação"},
}


def main() -> None:
    torch.set_num_threads(2)
    enable_determinism()
    p = make_problem(load_config("pendulum"))
    p.setup(torch.device("cpu"))

    seed_everything(SEED)
    nn_model = MLP(1, 1, 3, 32)
    opt = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
    for _ in range(1000):
        opt.zero_grad()
        loss = ((nn_model(p.t_data) - p.x_data) ** 2).mean()
        loss.backward()
        opt.step()

    seed_everything(SEED)
    pinn = MLP(1, 1, 3, 32)
    opt = torch.optim.Adam(pinn.parameters(), lr=1e-4)
    for _ in range(20000):
        opt.zero_grad()
        loss = sum(p.loss_terms(pinn).values())
        loss.backward(inputs=list(pinn.parameters()))
        opt.step()

    with torch.no_grad():
        l2 = {"standard": p.relative_l2(nn_model), "pinn": p.relative_l2(pinn)}
        t = p.eval_inputs
        preds = {"standard": nn_model(t), "pinn": pinn(t)}

    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for lang, T in TEXT.items():
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True)
        for ax, key, title in ((axes[0], "standard", T["nn"].format(e="1.000" if lang == "pt" else "1,000")),
                               (axes[1], "pinn", T["pinn"].format(e="20.000" if lang == "pt" else "20,000"))):
            ax.plot(t, p.eval_exact, color="grey", lw=2, alpha=0.8, label=T["exact"])
            ax.plot(t, preds[key], color="#5B21B6", lw=1.4, label=T["pred"])
            ax.scatter(p.t_data, p.x_data, s=20, color="#D97706", zorder=3, label=T["data"])
            if key == "pinn":
                ax.scatter(p.t_phys.detach(), torch.zeros_like(p.t_phys), s=10, color="#059669",
                           alpha=0.6, label=T["phys"])
            ax.set_ylim(-1.3, 1.3)
            err = f"{l2[key]:.3f}".replace(".", "," if lang == "pt" else ".")
            ax.set_title(f"{title} — $L^2$ = {err}", fontsize=9)
        axes[1].set_xlabel("t")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
        fig.savefig(out / f"fig_nn_vs_pinn_{lang}.pdf", bbox_inches="tight")
        fig.savefig(out / f"fig_nn_vs_pinn_{lang}.png", bbox_inches="tight", dpi=200)
        plt.close(fig)
    (ROOT / "results" / "nn_vs_pinn.json").write_text(json.dumps({"seed": SEED, "rel_l2": l2}, indent=1))
    print(l2)


if __name__ == "__main__":
    main()
