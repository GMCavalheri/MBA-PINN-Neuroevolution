"""Measure speed and accuracy of fp32 (CUDA graph) vs AMP (fp16, eager) on the GPU, CPU speed, and
multi-worker throughput.

Decision rule (documented in the article): AMP is used for a problem only if it is faster AND its
relative L2 error stays within 10% of fp32 for every benchmark architecture.

  python scripts/benchmark_amp.py [--epochs 2000] [--problems pendulum heat wave]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from evopinn.config import load_config  # noqa: E402
from evopinn.experiment import environment_info  # noqa: E402
from evopinn.problems import make_problem  # noqa: E402
from evopinn.seed import enable_determinism  # noqa: E402
from evopinn.train import train  # noqa: E402

ARCHS = [(3, 20), (6, 10), (10, 20)]


def bench(problem_name: str, device: str, amp: bool, arch, epochs: int, seed: int = 0) -> dict:
    enable_determinism()
    cfg = load_config(problem_name)
    p = make_problem(cfg)
    p.setup(torch.device(device))
    torch.set_num_threads(1 if device == "cuda" else torch.get_num_threads())
    r = train(p, arch, epochs, seed, float(cfg["lr"]), amp, int(cfg["fitness_window"]))
    return {"problem": problem_name, "device": device, "amp": amp, "mode": r.mode, "arch": list(arch), "epochs": epochs,
            "ms_per_epoch": 1000 * r.seconds / epochs, "rel_l2": r.rel_l2,
            "final_loss": r.final_loss, "diverged": r.diverged}


def _worker(problem_name, device, amp, epochs, q):
    q.put(bench(problem_name, device, amp, (5, 15), epochs)["ms_per_epoch"])


def throughput(problem_name: str, device: str, amp: bool, k: int, epochs: int) -> float:
    """Epochs per second with k independent processes training concurrently."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(problem_name, device, amp, epochs, q)) for _ in range(k)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return k * epochs / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--problems", nargs="+", default=["pendulum", "heat", "wave"])
    ap.add_argument("--workers", nargs="+", type=int, default=[1, 2, 3, 4])
    args = ap.parse_args()
    out = {"environment": environment_info(torch.device("cuda")), "runs": [], "throughput": [], "decision": {}}

    for name in args.problems:
        rows = []
        for arch in ARCHS:
            for amp in (False, True):
                row = bench(name, "cuda", amp, arch, args.epochs)
                rows.append(row)
                print(f"{name:8s} {row['mode']:10s} amp={amp!s:5s} arch={arch} {row['ms_per_epoch']:.2f} ms/epoch "
                      f"L2={row['rel_l2']:.3e} loss={row['final_loss']:.3e}", flush=True)
        cpu = bench(name, "cpu", False, ARCHS[0], max(200, args.epochs // 10))
        rows.append(cpu)
        print(f"{name:8s} cpu  fp32 arch={ARCHS[0]} {cpu['ms_per_epoch']:.2f} ms/epoch", flush=True)
        out["runs"] += rows

        fp32 = [r for r in rows if r["device"] == "cuda" and not r["amp"]]
        half = [r for r in rows if r["device"] == "cuda" and r["amp"]]
        faster = sum(r["ms_per_epoch"] for r in half) < sum(r["ms_per_epoch"] for r in fp32)
        accurate = all(h["rel_l2"] <= 1.10 * f["rel_l2"] and not h["diverged"] for f, h in zip(fp32, half))
        gpu_ms = sum(r["ms_per_epoch"] for r in fp32) / len(fp32)
        device = "cuda" if gpu_ms < cpu["ms_per_epoch"] else "cpu"
        use_amp = bool(device == "cuda" and faster and accurate)
        out["decision"][name] = {"device": device, "amp": use_amp, "amp_faster": faster,
                                 "amp_within_10pct_l2": accurate, "gpu_fp32_ms": gpu_ms,
                                 "cpu_fp32_ms": cpu["ms_per_epoch"]}
        print(f"{name:8s} decision: {out['decision'][name]}", flush=True)

        dec = out["decision"][name]
        for k in args.workers:
            eps = throughput(name, dec["device"], dec["amp"], k, max(200, args.epochs // 4))
            out["throughput"].append({"problem": name, "device": dec["device"], "amp": dec["amp"],
                                      "workers": k, "epochs_per_second": eps})
            print(f"{name:8s} workers={k} {eps:.0f} epochs/s", flush=True)

    path = ROOT / "results" / "benchmark.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
