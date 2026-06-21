from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class LatencyResult:
    latency_ms_per_iter: float
    peak_vram_mb: float
    n_iters: int


def reset_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_vram_mb() -> float:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


@contextmanager
def cuda_timer():
    """Yields a dict that will contain elapsed_ms after the block (GPU events)."""
    import torch

    result = {"elapsed_ms": 0.0}
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        yield result
        end.record()
        torch.cuda.synchronize()
        result["elapsed_ms"] = start.elapsed_time(end)
    else:
        t0 = time.time()
        yield result
        result["elapsed_ms"] = (time.time() - t0) * 1000.0


def benchmark(fn, warmup: int = 50, measured: int = 200) -> LatencyResult:
    """Run fn() warmup then measured times; return per-iter latency + peak VRAM."""
    import torch

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    reset_peak_memory()
    with cuda_timer() as t:
        for _ in range(measured):
            fn()
    return LatencyResult(
        latency_ms_per_iter=t["elapsed_ms"] / max(measured, 1),
        peak_vram_mb=peak_vram_mb(),
        n_iters=measured,
    )
