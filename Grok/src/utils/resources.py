"""CPU / memory monitoring helpers."""
from __future__ import annotations

import time
from typing import Dict

import psutil


class ResourceMonitor:
    """Track process CPU% and peak RSS memory."""

    def __init__(self) -> None:
        self.proc = psutil.Process()
        self.peak_rss_mb = 0.0
        self._cpu_samples = []
        # Prime cpu_percent
        self.proc.cpu_percent(interval=None)

    def sample(self) -> Dict[str, float]:
        rss_mb = self.proc.memory_info().rss / (1024 * 1024)
        if rss_mb > self.peak_rss_mb:
            self.peak_rss_mb = rss_mb
        cpu = self.proc.cpu_percent(interval=None)
        self._cpu_samples.append(cpu)
        return {"cpu_percent": float(cpu), "rss_mb": float(rss_mb), "peak_rss_mb": float(self.peak_rss_mb)}

    def summary(self) -> Dict[str, float]:
        avg_cpu = float(np_mean(self._cpu_samples)) if self._cpu_samples else 0.0
        return {
            "avg_cpu_percent": avg_cpu,
            "peak_memory_mb": float(self.peak_rss_mb),
        }


def np_mean(xs):
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


class LatencyTimer:
    """Simple wall-clock latency accumulator for policy inference."""

    def __init__(self) -> None:
        self.times_ms = []

    def time_call(self, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        self.times_ms.append(dt)
        return out

    def mean_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        return float(sum(self.times_ms) / len(self.times_ms))

    def reset(self) -> None:
        self.times_ms.clear()
