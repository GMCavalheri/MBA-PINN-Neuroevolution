"""Exact solutions of the three benchmark problems (figure for the README / article).

  python scripts/fig_problems.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from evopinn.config import load_config  # noqa: E402
from evopinn.problems import make_problem  # noqa: E402


def main() -> None:
    probs = {}
    for name in ("pendulum", "heat", "wave"):
        p = make_problem(load_config(name))
        p.setup(torch.device("cpu"))
        probs[name] = p

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    p = probs["pendulum"]
    ax = axes[0]
    ax.plot(p.eval_inputs, p.eval_exact, color="black", lw=1.2, label="exact $x(t)=\\sin(20t)$")
    ax.scatter(p.t_data, p.x_data, s=16, color="#D97706", zorder=3, label="10 observations")
    ax.scatter(p.t_phys.detach(), torch.full_like(p.t_phys, -1.25), s=8, color="#059669",
               label="30 collocation points")
    ax.set_ylim(-1.45, 1.35)
    ax.set_xlabel("t")
    ax.set_title("Pendulum (ODE): $\\ddot x + \\omega_0^2 x = 0$", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    for ax, name, title in ((axes[1], "heat", "Heat (PDE): $\\alpha u_{xx} - u_t = 0$"),
                            (axes[2], "wave", "Wave (PDE): $u_{tt} - c^2 u_{xx} = 0$")):
        p = probs[name]
        n = int(load_config(name)["problem"]["n_eval"])
        u = p.eval_exact.view(n, n).numpy()
        im = ax.imshow(u.T, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    out = ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "problems.png", dpi=170, bbox_inches="tight")
    print(out / "problems.png")


if __name__ == "__main__":
    main()
