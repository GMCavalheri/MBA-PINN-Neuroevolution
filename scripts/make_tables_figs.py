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
    """Median relative L2 error and median parameter count of each arm, and the fitness-proxy correlation."""
    head = {"en": ("Problem", "Median relative $L^2$ error", "Median parameters", "GA", "Random", "Random search",
                   "$\\rho_{\\mathrm{S}}$"),
            "pt": ("Problema", "Mediana do erro $L^2$ relativo", "Mediana de parâmetros", "AG", "Aleatória",
                   "Busca aleatória", "$\\rho_{\\mathrm{S}}$")}[lang]
    lines = [r"\begin{tabular}{l ccc ccc c}", r"\toprule",
             rf" & \multicolumn{{3}}{{c}}{{{head[1]}}} & \multicolumn{{3}}{{c}}{{{head[2]}}} & \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             rf"{head[0]} & {head[3]} & {head[4]} & {head[5]} & {head[3]} & {head[4]} & {head[5]} & {head[6]} \\",
             r"\midrule"]
    for prob, s in summary.items():
        med = s["median"]["rel_l2"]
        params = s["median_params"]
        row = ([NAMES[lang][prob]] + [fmt_sci(med[a], lang) for a in ("ga", "random", "random_search")]
               + [f"{params[a]:,.0f}".replace(",", "." if lang == "pt" else ",") for a in ("ga", "random", "random_search")]
               + [fmt_num(s["proxy_spearman_screening"]["rho"], lang)])
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_comparisons(summary: dict, lang: str) -> str:
    """Paired comparisons of the GA with each reference (primary metric: relative L2 error)."""
    head = {"en": ("Problem", "Reference", "Wins", "Wilson 95\\%", "$p_{\\mathrm{B}}$", "$p_{\\mathrm{W}}$",
                   "Median ratio [95\\% CI]", "Wins (loss)"),
            "pt": ("Problema", "Referência", "Vitórias", "Wilson 95\\%", "$p_{\\mathrm{B}}$", "$p_{\\mathrm{W}}$",
                   "Razão mediana [IC 95\\%]", "Vitórias (custo)")}[lang]
    refname = {"en": {"random": "Random", "random_search": "Random search"},
               "pt": {"random": "Aleatória", "random_search": "Busca aleatória"}}[lang]
    lines = [r"\begin{tabular}{ll cc cc c c}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    probs = list(summary)
    for i, prob in enumerate(probs):
        s = summary[prob]
        for j, ref in enumerate(("random", "random_search")):
            c = s["comparisons"][f"rel_l2:ga_vs_{ref}"]
            cl = s["comparisons"][f"final_loss:ga_vs_{ref}"]
            lo, hi = c["wilson_95"]
            rlo, rhi = c["median_ratio_95"]
            ratio = f"{fmt_num(c['median_ratio'], lang)} [{fmt_num(rlo, lang)}; {fmt_num(rhi, lang)}]" if lang == "pt" \
                else f"{fmt_num(c['median_ratio'], lang)} [{fmt_num(rlo, lang)}, {fmt_num(rhi, lang)}]"
            lines.append(" & ".join([NAMES[lang][prob] if j == 0 else "", refname[ref], f"{c['wins']}/{c['n']}",
                                     f"{fmt_pct(lo, lang)}--{fmt_pct(hi, lang)}",
                                     fmt_sci(c["binomial_p_holm"], lang), fmt_sci(c["wilcoxon_p_holm"], lang),
                                     ratio, f"{cl['wins']}/{cl['n']}"]) + r" \\")
        if i < len(probs) - 1:
            lines.append(r"\addlinespace")
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
        (tables / f"table_comparisons_{lang}.tex").write_text(table_comparisons(summary, lang), encoding="utf-8")
        figs = results / "figures"
        fig_error_boxplot(results, figs, lang)
        fig_loss_curves(results, figs, lang)
        fig_solutions(results, figs, lang)
    print(json.dumps({p: {k: v for k, v in s.items() if k in ("n_runs", "median", "diverged")}
                      for p, s in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
