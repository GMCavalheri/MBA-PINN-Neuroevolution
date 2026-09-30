"""Training of one PINN architecture with Adam (optionally with fp16 automatic mixed precision)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import MLP, n_parameters
from .problems.base import Problem
from .seed import seed_everything

Arch = tuple[int, int]  # (hidden layers, neurons per hidden layer)


@dataclass
class TrainResult:
    arch: Arch
    epochs: int
    seed: int
    amp: bool
    mode: str                  # 'cuda_graph', 'eager_cuda' or 'eager_cpu'
    final_loss: float          # mean total loss over the last `window` epochs (fitness)
    final_terms: dict          # mean of each loss term over the same window
    last_loss: float           # total loss of the very last epoch (the monograph's criterion)
    rel_l2: float              # relative L2 error against the exact solution
    n_params: int
    seconds: float
    diverged: bool
    history: np.ndarray = field(repr=False)  # [epochs, 1 + n_terms]: total, then each term
    state_dict: dict | None = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "arch": list(self.arch), "epochs": self.epochs, "seed": self.seed, "amp": self.amp, "mode": self.mode,
            "final_loss": self.final_loss, "final_terms": self.final_terms,
            "last_loss": self.last_loss, "rel_l2": self.rel_l2, "n_params": self.n_params,
            "seconds": self.seconds, "diverged": self.diverged,
        }


def _finite_or_inf(x: float) -> float:
    return x if math.isfinite(x) else math.inf


GRAPH_WARMUP = 3  # eager warm-up epochs required before capturing a CUDA graph
_SIDE_STREAMS: dict = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    """One warm-up stream per device, reused by every training.

    cuBLAS keeps a workspace per (handle, stream); creating a new stream for each training would
    leak that workspace (~65 MB with CUBLAS_WORKSPACE_CONFIG=:4096:8) on every call.
    """
    key = str(device)
    if key not in _SIDE_STREAMS:
        _SIDE_STREAMS[key] = torch.cuda.Stream(device=device)
    return _SIDE_STREAMS[key]


def _record(terms: dict) -> torch.Tensor:
    """[total, term_1, ..., term_k] as a float32 vector."""
    values = torch.stack([t.float() for t in terms.values()])
    return torch.cat([values.sum().view(1), values])


def _loop_eager(problem, model, optimizer, epochs, history, use_amp) -> None:
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    device_type = problem.device.type
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            terms = problem.loss_terms(model)
        rec = _record(terms)
        # inputs=...: accumulate gradients only into the weights, never into the input points
        scaler.scale(rec[0]).backward(inputs=list(model.parameters()))
        scaler.step(optimizer)
        scaler.update()
        history[epoch] = rec.detach()


def _loop_cuda_graph(problem, model, optimizer, epochs, history) -> None:
    """Capture one full training step (loss, input derivatives, backward, Adam) and replay it.

    Removes the kernel-launch overhead that dominates small PINNs. Warm-up epochs run eagerly on a
    side stream (as required by CUDA graph capture) and are part of the training.
    """
    params = list(model.parameters())

    def step() -> torch.Tensor:
        rec = _record(problem.loss_terms(model))
        # Only the weights receive gradients. Letting the input points (requires_grad=True) accumulate
        # .grad inside the capture would pin the graph's private memory pool after training (leak).
        rec[0].backward(inputs=params)
        optimizer.step()
        return rec.detach()

    warm = min(GRAPH_WARMUP, epochs)
    side = _side_stream(problem.device)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for epoch in range(warm):
            optimizer.zero_grad(set_to_none=True)
            history[epoch] = step()
    torch.cuda.current_stream().wait_stream(side)
    if warm == epochs:
        return
    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        static_rec = step()
    for epoch in range(warm, epochs):
        graph.replay()
        history[epoch].copy_(static_rec)
    del graph


def train(problem: Problem, arch: Arch, epochs: int, seed: int, lr: float,
          amp: bool = False, window: int = 100, keep_state: bool = False,
          cuda_graph: bool = True) -> TrainResult:
    """Train a fresh network `arch` for `epochs` epochs and evaluate it.

    On CUDA without AMP the training step is replayed from a CUDA graph (same algorithm, faster);
    AMP and CPU use the eager loop. Results are reproducible for a fixed mode, device and software.
    """
    device = problem.device
    use_amp = bool(amp and device.type == "cuda")
    use_graph = bool(cuda_graph and device.type == "cuda" and not use_amp)
    seed_everything(seed)
    model = MLP(problem.n_in, 1, *arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, capturable=use_graph)

    n_terms = len(problem.term_names)
    history = torch.empty((epochs, 1 + n_terms), dtype=torch.float32, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    if use_graph:
        _loop_cuda_graph(problem, model, optimizer, epochs, history)
    else:
        _loop_eager(problem, model, optimizer, epochs, history, use_amp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start

    hist = history.cpu().numpy()
    w = min(window, epochs)
    tail = hist[-w:]
    diverged = not np.all(np.isfinite(hist))
    final_loss = _finite_or_inf(float(np.mean(tail[:, 0])))
    final_terms = {name: float(np.mean(tail[:, i + 1])) for i, name in enumerate(problem.term_names)}
    model.eval()
    rel_l2 = problem.relative_l2(model)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()} if keep_state else None

    return TrainResult(
        arch=(int(arch[0]), int(arch[1])), epochs=epochs, seed=int(seed), amp=use_amp,
        mode="cuda_graph" if use_graph else f"eager_{device.type}",
        final_loss=final_loss, final_terms=final_terms,
        last_loss=_finite_or_inf(float(hist[-1, 0])), rel_l2=_finite_or_inf(rel_l2),
        n_params=n_parameters(model), seconds=seconds, diverged=diverged,
        history=hist, state_dict=state,
    )
