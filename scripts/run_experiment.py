"""Run the experiment for one problem.

Examples
  python scripts/run_experiment.py --problem heat                      # all 35 runs
  python scripts/run_experiment.py --problem heat --runs 0-9 --workers 3
  python scripts/run_experiment.py --problem wave --runs 0 --epoch-scale 0.01 --out results_smoke

Runs whose run_XX.json already exists are skipped, so an interrupted execution can be resumed.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evopinn.config import load_config  # noqa: E402


def parse_runs(spec: str, total: int) -> list[int]:
    if spec == "all":
        return list(range(total))
    runs: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            runs += range(int(a), int(b) + 1)
        else:
            runs.append(int(part))
    return sorted(set(runs))


def setup_logging(logfile: Path) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(processName)s] %(message)s",
                        handlers=[logging.FileHandler(logfile), logging.StreamHandler()], force=True)


def worker(problem: str, runs: list[int], outdir: str, amp: bool, device: str,
           epoch_scale: float, logfile: str) -> None:
    setup_logging(Path(logfile))
    import torch
    from evopinn.experiment import resolve_device, run_path, run_round, scale_epochs
    cfg = load_config(problem)
    if epoch_scale != 1.0:
        cfg = scale_epochs(cfg, epoch_scale)
    dev = resolve_device(device)
    if dev.type == "cuda":
        torch.set_num_threads(1)  # each GPU worker only launches kernels; avoid CPU oversubscription
    out = Path(outdir)
    for r in runs:
        if run_path(out, r).exists():
            logging.info("%s run %02d already done, skipping", problem, r)
            continue
        t0 = time.time()
        logging.info("%s run %02d started (device=%s, amp=%s)", problem, r, dev, amp)
        try:
            rec = run_round(cfg, r, out, amp, dev)
        except Exception:  # keep the worker alive; the run stays incomplete and is redone on resume
            logging.exception("%s run %02d FAILED", problem, r)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            continue
        f = rec["final"]
        logging.info("%s run %02d done in %.0fs | L2 ga=%.3e random=%.3e rs=%.3e | arch ga=%s random=%s rs=%s",
                     problem, r, time.time() - t0, f["ga"]["rel_l2"], f["random"]["rel_l2"],
                     f["random_search"]["rel_l2"], f["ga"]["arch"], f["random"]["arch"],
                     f["random_search"]["arch"])


def gpu_temperature() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return int(out.stdout.split(",")[0])
    except Exception:
        return None


def thermal_guard(procs, pause_at: int, resume_at: int, templog: Path, interval: float = 10.0) -> None:
    """Pause (SIGSTOP) the workers while the GPU is at or above `pause_at` C; resume (SIGCONT) at `resume_at` C.

    Kernels already queued finish normally; the workers simply stop launching new ones while paused.
    Every reading is appended to `templog` (time, temperature, paused).
    """
    paused = False
    new = not templog.exists()
    with open(templog, "a", encoding="utf-8") as f:
        if new:
            f.write("time,temperature_c,paused\n")
        try:
            while any(p.is_alive() for p in procs):
                t = gpu_temperature()
                if t is not None:
                    if not paused and t >= pause_at:
                        for p in procs:
                            if p.is_alive():
                                os.kill(p.pid, signal.SIGSTOP)
                        paused = True
                        logging.warning("GPU at %d C >= %d C: workers paused", t, pause_at)
                    elif paused and t <= resume_at:
                        for p in procs:
                            if p.is_alive():
                                os.kill(p.pid, signal.SIGCONT)
                        paused = False
                        logging.info("GPU at %d C <= %d C: workers resumed", t, resume_at)
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{t},{int(paused)}\n")
                    f.flush()
                time.sleep(interval)
        finally:
            for p in procs:
                if p.is_alive():
                    os.kill(p.pid, signal.SIGCONT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True, choices=["pendulum", "heat", "wave"])
    ap.add_argument("--runs", default="all", help="'all', '0-34', '0,3,7' ...")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--amp", choices=["config", "on", "off"], default="config")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: config)")
    ap.add_argument("--epoch-scale", type=float, default=1.0, help="multiply every epoch count (smoke tests)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--pause-at", type=int, default=80, help="pause workers at this GPU temperature (C); 0 disables")
    ap.add_argument("--resume-at", type=int, default=72, help="resume workers at this GPU temperature (C)")
    args = ap.parse_args()

    cfg = load_config(args.problem)
    amp = {"config": bool(cfg["amp"]), "on": True, "off": False}[args.amp]
    device = args.device or cfg["device"]
    runs = parse_runs(args.runs, int(cfg["runs"]))
    outdir = Path(args.out) / args.problem
    logfile = outdir / "log.txt"
    setup_logging(logfile)
    logging.info("problem=%s runs=%s workers=%d amp=%s device=%s scale=%s",
                 args.problem, runs, args.workers, amp, device, args.epoch_scale)

    chunks = [runs[i::args.workers] for i in range(args.workers)]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, name=f"w{i}",
                         args=(args.problem, chunk, str(outdir), amp, device, args.epoch_scale, str(logfile)))
             for i, chunk in enumerate(chunks) if chunk]
    for p in procs:
        p.start()
    if args.pause_at > 0 and device == "cuda":
        logging.info("thermal guard: pause at %d C, resume at %d C", args.pause_at, args.resume_at)
        thermal_guard(procs, args.pause_at, args.resume_at, outdir / "gpu_temperature.csv")
    for p in procs:
        p.join()
    failed = [p.name for p in procs if p.exitcode != 0]
    missing = [r for r in runs if not (outdir / f"run_{r:02d}.json").exists()]
    if failed or missing:
        logging.error("workers failed: %s | incomplete runs: %s (rerun the same command to resume)", failed, missing)
        sys.exit(1)
    logging.info("all done: %d runs complete", len(runs))


if __name__ == "__main__":
    main()
