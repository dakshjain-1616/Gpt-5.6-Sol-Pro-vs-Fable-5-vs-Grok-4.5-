"""Shared utilities: seeding, torch setup, validation, resource monitoring, JSONL logging."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import psutil
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config and resolve paths relative to project root."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = str((PROJECT_ROOT / rel).resolve())
    return cfg


def set_global_seeds(seed: int) -> None:
    """Seed python, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


_TORCH_CONFIGURED = False


def setup_torch(threads: int = 2):
    """Cap torch CPU threads (CPU-only node). Idempotent per process."""
    global _TORCH_CONFIGURED
    import torch

    torch.set_num_threads(threads)
    if not _TORCH_CONFIGURED:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already set / parallel work started — thread cap still applied
        _TORCH_CONFIGURED = True
    return torch


def validate_array(name: str, arr: np.ndarray, shape: tuple, lo: float = -1e6, hi: float = 1e6) -> np.ndarray:
    """Validate an observation/metric array: shape, finiteness, bounds."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: contains non-finite values")
    if arr.min() < lo or arr.max() > hi:
        raise ValueError(f"{name}: values outside [{lo}, {hi}]: min={arr.min()}, max={arr.max()}")
    return arr


class JsonlLogger:
    """Append-only structured JSONL logger."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)

    def log(self, event: str, **fields) -> None:
        rec = {"ts": round(time.time(), 3), "event": event}
        rec.update(fields)
        self._fh.write(json.dumps(rec, default=float) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class ResourceMonitor:
    """psutil-based CPU / peak-memory tracker for the current process tree."""

    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.proc.cpu_percent(None)  # prime
        self.peak_rss_mb = 0.0
        self._cpu_samples: list[float] = []

    def sample(self) -> dict:
        rss_mb = self.proc.memory_info().rss / (1024 * 1024)
        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
        cpu = self.proc.cpu_percent(None)
        if cpu > 0:
            self._cpu_samples.append(cpu)
        return {"rss_mb": round(rss_mb, 1), "cpu_percent": round(cpu, 1)}

    def summary(self) -> dict:
        self.sample()
        avg_cpu = float(np.mean(self._cpu_samples)) if self._cpu_samples else 0.0
        return {"peak_rss_mb": round(self.peak_rss_mb, 1), "avg_cpu_percent": round(avg_cpu, 1)}
