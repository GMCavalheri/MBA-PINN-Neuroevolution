# Neuroevolution for the Architecture Search of Physics-Informed Neural Networks

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%20%7C%20CUDA%2012.8-ee4c2c)
![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

Code for the MBA in Artificial Intelligence and Big Data monograph (ICMC, University of São Paulo) and for the
derived article *Applying Neuroevolution to the Architecture Search of Physics-Informed Neural Networks*
(Gabriel Milanez Cavalheri and Rodrigo Colnago Contreras).

**Question.** The accuracy of a Physics-Informed Neural Network (PINN) depends on its architecture, and choosing
it by trial and error is expensive, because every candidate requires solving the differential equation. Can a
*cheap* Genetic Algorithm, guided only by short partial trainings, choose better PINN architectures than a random
choice, and does it beat random search with the same budget?

**What this repository does.** A single-generation, mutation-only Genetic Algorithm searches the number of hidden
layers and neurons per layer of a PINN, while the weights are trained by gradient descent (Adam). The search costs
about 1.5 full trainings. It is evaluated on three classical problems with exact solutions (a pendulum ODE, the
heat equation, and the wave equation), in 35 seeded runs per problem, against a random architecture and against
random search with the same epoch budget, using the relative L2 error against the exact solution.

---

## Method

![Single-generation neuroevolution of the PINN architecture](docs/images/method_flowchart.png)

An individual is an architecture `A = (L, n)`: `L` hidden layers with `n` neurons each (the same width in every
hidden layer; input and output sizes come from the problem). The fitness is the total PINN loss averaged over the
last 100 epochs of a partial training, so the search never uses the exact solution.

Each run compares three arms, all ending with a full training (20,000 epochs) of the chosen architecture:

| Arm | How the architecture is chosen | Search budget |
|---|---|---|
| `ga` | 10 random architectures × 1,000 epochs → truncation selection of the 5 best → mutation (ΔL ∈ {0,1}, Δn ∈ {0,1,2}) → 10 × 2,000 epochs → best | 30,000 epochs |
| `random` | one architecture drawn at random | none |
| `random_search` | best of 10 random architectures trained for 3,000 epochs | 30,000 epochs |

The `random` arm measures the gain of any search over no search; `random_search` isolates the contribution of the
evolutionary operators from the effect of simply evaluating more candidates.

## Benchmark problems

![Exact solutions of the three benchmark problems](docs/images/problems.png)

| Problem | Equation | Exact solution | Loss terms | Learning rate |
|---|---|---|---|---|
| `pendulum` | x'' + ω₀²x = 0, ω₀ = 20, t ∈ [0, 1] | sin(20t) | MSE of 10 observations (t < 0.4) + 10⁻⁴ · residual at 30 collocation points | 10⁻⁴ |
| `heat` | α u_xx − u_t = 0, α = 0.05 | sin(πx) e^(−απ²t) | residual (100 × 250 grid) + initial condition + boundaries | 10⁻³ |
| `wave` | u_tt − c² u_xx = 0, c = 1 | sin(πx) cos(πt) | residual + initial condition + initial velocity + boundaries | 10⁻³ |

All networks use **tanh** activations and Xavier initialization. PDE problems are dimensionless on x, t ∈ [0, 1].

## Why physics in the loss matters

With the same architecture and the same 10 observations, a standard network fits the data but cannot extrapolate,
while the PINN recovers the whole oscillation (`scripts/fig_nn_vs_pinn.py`, seed 2026):

![Standard network vs PINN on the pendulum](docs/images/nn_vs_pinn.png)

## Results

> The 105 seeded runs (35 per problem) are running. The statistics, tables and figures will be added here when they
> finish: relative L2 error of each arm, wins of the GA with binomial and Wilcoxon tests (Holm-corrected), median
> error ratios with bootstrap intervals, and whether the partial-training loss is a good proxy for the error.

## Installation

```bash
git clone https://github.com/GMCavalheri/MBA-PINN-Neuroevolution.git
cd MBA-PINN-Neuroevolution
python3 -m venv .venv
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # 26 tests (GPU tests are skipped without CUDA)
```

## Usage

```bash
.venv/bin/python scripts/benchmark_amp.py                           # fp32 vs AMP, CPU vs GPU, workers
.venv/bin/python scripts/run_experiment.py --problem heat --workers 3
.venv/bin/python scripts/run_experiment.py --problem wave --runs 0-4 # a subset of runs
.venv/bin/python scripts/make_tables_figs.py                        # statistics, LaTeX tables, figures
scripts/run_all.sh                                                  # everything, in sequence
```

Runs are resumable: a run whose `results/<problem>/run_XX.json` exists is skipped. A thermal guard pauses the
workers while the GPU is at or above 80 °C and resumes them at 72 °C (`--pause-at`, `--resume-at`).

## Reproducibility

- Every random decision of run `r` (initial population, mutations, reference architectures, weight
  initializations) is derived from the master seed `1000 + r` through `numpy.random.SeedSequence`; the derived seeds
  are stored in each `run_XX.json`.
- CPU runs are bit-exact. GPU runs use deterministic algorithms (`torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`) and are identical on the same hardware, software versions and execution mode,
  which are also recorded in the JSON.
- Configuration lives in `configs/*.yaml`; each run stores the full configuration and its hash.

## GPU performance

These networks are tiny, so training time is dominated by launching many small GPU kernels. On CUDA, the whole
training step (loss, second derivatives through `torch.autograd.grad`, backpropagation and Adam) is captured once in a
**CUDA graph** and replayed, which is 3 to 13 times faster than eager execution on an RTX 2060 Mobile. Automatic mixed
precision (fp16) was benchmarked and rejected: it was slower and increased the relative L2 error by more than 10% on
the PDE problems, which need accurate second derivatives. Runs therefore use fp32, three at a time on the GPU.

## Changes from the original monograph code

This is a clean reimplementation. The code used in the monograph had problems that are fixed here:

- The heat and wave PINNs used **ReLU**. A ReLU network is piecewise linear in its inputs, so `u_xx` and `u_tt` were
  exactly zero: the heat residual reduced to `−u_t` and the wave residual vanished. All networks now use tanh, and a
  regression test guards against it.
- The wave equation used c = 300 in t ∈ [0, 1] (150 oscillations, unresolvable on the grid); it is now dimensionless
  with c = 1.
- Winners were decided by the loss of the last epoch; they are now decided by the error against the exact solution
  (primary) and by the loss averaged over the last 100 epochs (secondary).
- No random seeds were set; now every decision is seeded. Tensors are consistently placed on the device.
- A column-label swap in the pendulum logs (the source of the "17 layers × 9 neurons" entry in the monograph's
  Table 2; the trained network had 9 layers of 17 neurons) no longer exists.
- Random search with the same budget was added as a reference, and a GPU memory leak in CUDA-graph training was
  found and fixed before the experiments.

## Repository layout

```
configs/            base.yaml + one YAML per problem
evopinn/            model, problems, training (CUDA graphs), GA, baselines, experiment, analysis, plots
scripts/            run_experiment.py, run_all.sh, benchmark_amp.py, make_tables_figs.py, figure scripts
tests/              pytest suite (derivatives, exact solutions, GA, seeds, smoke run, GPU)
docs/images/        figures used in this README
results/            benchmark.json; per-run JSON, loss histories and final weights (added after the runs)
```

## Authors

- **Gabriel Milanez Cavalheri** — MBA in AI and Big Data, ICMC, University of São Paulo
- **Prof. Dr. Rodrigo Colnago Contreras** (advisor) — Department of Science and Technology, Federal University of
  São Paulo (UNIFESP), São José dos Campos

## Citation

The article is in preparation; citation details will be added upon publication.

## License

Released under the [MIT License](LICENSE).
