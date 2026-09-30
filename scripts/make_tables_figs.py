"""Statistics, LaTeX tables (EN/PT) and figures from the completed runs.

  python scripts/make_tables_figs.py [--results results]
Writes results/summary.json, results/tables/*.tex and results/figures/*.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evopinn.analysis import analyse  # noqa: E402
from evopinn.plots import fig_error_boxplot, fig_loss_curves, fig_solutions  # noqa: E402

NAMES = {"en": {"pendulum": "Pendulum", "heat": "Heat conduction", "wave": "Vibrating string"},
         "pt": {"pendulum": "Pêndulo", "heat": "Condução de calor", "wave": "Corda vibrante"}}


def fmt_sci(x: float, lang: str) -> str:
    if x != x:  # NaN
        return "--"
    if x >= 0.01:
        s = f"{x:.3f}"
    else:
        m, e = f"{x:.1e}".split("e")
        s = f"{m} \\times 10^{{{int(e)}}}"
    return f"${s.replace('.', '{,}')}$" if lang == "pt" else f"${s}$"


def fmt_pct(x: float, lang: str) -> str:
    s = f"{100 * x:.0f}\\%"
    return s


def fmt_num(x: float, lang: str, nd: int = 2) -> str:
    s = f"{x:.{nd}f}"
    return s.replace(".", ",") if lang == "pt" else s


def table_results(summary: dict, lang: str) -> str:
    """Median errors per arm and GA comparisons on the primary metric (relative L2)."""
    head = {"en": ("Problem", "Median rel. $L^2$ error", "GA", "Random", "Random search",
                   "GA vs random", "GA vs random search", "Wins", "$p$ (Holm)", "Ratio"),
            "pt": ("Problema", "Mediana do erro $L^2$ relativo", "AG", "Aleatória", "Busca aleat.",
                   "AG vs aleatória", "AG vs busca aleatória", "Vitórias", "$p$ (Holm)", "Razão")}[lang]
    lines = [r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
             rf" & \multicolumn{{3}}{{c}}{{{head[1]}}} & \multicolumn{{3}}{{c}}{{{head[5]}}} & \multicolumn{{3}}{{c}}{{{head[6]}}} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             rf"{head[0]} & {head[2]} & {head[3]} & {head[4]} & {head[7]} & {head[8]} & {head[9]} & {head[7]} & {head[8]} & {head[9]} \\",
             r"\midrule"]
    for prob, s in summary.items():
        med = s["median"]["rel_l2"]
        row = [NAMES[lang][prob]] + [fmt_sci(med[a], lang) for a in ("ga", "random", "random_search")]
        for ref in ("random", "random_search"):
            c = s["comparisons"][f"rel_l2:ga_vs_{ref}"]
            row += [f"{c['wins']}/{c['n']}", fmt_sci(c["binomial_p_holm"], lang), fmt_num(c["median_ratio"], lang)]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_secondary(summary: dict, lang: str) -> str:
    """Secondary metric (final loss) and Wilcoxon tests on the primary metric."""
    head = {"en": ("Problem", "Comparison", "Wins (loss)", "$p$ binomial (loss, Holm)",
                   "$p$ Wilcoxon ($L^2$, Holm)", "Wilson 95\\% ($L^2$ wins)"),
            "pt": ("Problema", "Comparação", "Vitórias (custo)", "$p$ binomial (custo, Holm)",
                   "$p$ Wilcoxon ($L^2$, Holm)", "Wilson 95\\% (vitórias $L^2$)")}[lang]
    refname = {"en": {"random": "GA vs random", "random_search": "GA vs random search"},
               "pt": {"random": "AG vs aleatória", "random_search": "AG vs busca aleatória"}}[lang]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for prob, s in summary.items():
        for ref in ("random", "random_search"):
            cl = s["comparisons"][f"final_loss:ga_vs_{ref}"]
            c2 = s["comparisons"][f"rel_l2:ga_vs_{ref}"]
            lo, hi = c2["wilson_95"]
            lines.append(" & ".join([NAMES[lang][prob], refname[ref], f"{cl['wins']}/{cl['n']}",
                                     fmt_sci(cl["binomial_p_holm"], lang), fmt_sci(c2["wilcoxon_p_holm"], lang),
                                     f"{fmt_pct(lo, lang)}--{fmt_pct(hi, lang)}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args()
    results = Path(args.results)
    summary = analyse(results)
    (results / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    tables = results / "tables"
    tables.mkdir(exist_ok=True)
    for lang in ("en", "pt"):
        (tables / f"table_results_{lang}.tex").write_text(table_results(summary, lang), encoding="utf-8")
        (tables / f"table_secondary_{lang}.tex").write_text(table_secondary(summary, lang), encoding="utf-8")
        figs = results / "figures"
        fig_error_boxplot(results, figs, lang)
        fig_loss_curves(results, figs, lang)
        fig_solutions(results, figs, lang)
    print(json.dumps({p: {k: v for k, v in s.items() if k in ("n_runs", "median", "diverged")}
                      for p, s in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
