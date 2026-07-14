"""Metric aggregation and validation for evaluation results."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "scenario",
    "seed",
    "avg_wait",
    "avg_travel_time",
    "avg_queue",
    "p95_queue",
    "completed_trips",
    "throughput",
    "gridlock_duration",
    "gridlock_events",
    "policy_inference_latency_ms",
    "cpu_usage",
    "peak_memory_mb",
    "failed",
    "incomplete",
]


def validate_row(row: Dict[str, Any]) -> List[str]:
    """Return list of missing/invalid field messages."""
    errs = []
    for c in REQUIRED_COLUMNS:
        if c not in row:
            errs.append(f"missing:{c}")
    for num in (
        "avg_wait",
        "avg_travel_time",
        "avg_queue",
        "p95_queue",
        "throughput",
        "gridlock_duration",
        "policy_inference_latency_ms",
        "cpu_usage",
        "peak_memory_mb",
    ):
        if num in row:
            try:
                v = float(row[num])
                if not np.isfinite(v):
                    errs.append(f"nonfinite:{num}")
            except Exception:
                errs.append(f"not_numeric:{num}")
    return errs


def rows_to_dataframe(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            if c in ("failed", "incomplete"):
                df[c] = False
            else:
                df[c] = 0.0
    return df[REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]]


def aggregate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate per-seed metrics into overall and per-scenario summaries."""
    if df is None or len(df) == 0:
        return {
            "n_episodes": 0,
            "n_failed": 0,
            "n_incomplete": 0,
            "overall": {},
            "per_scenario": {},
        }

    numeric = [
        "avg_wait",
        "avg_travel_time",
        "avg_queue",
        "p95_queue",
        "completed_trips",
        "throughput",
        "gridlock_duration",
        "gridlock_events",
        "policy_inference_latency_ms",
        "cpu_usage",
        "peak_memory_mb",
    ]

    def _summ(sub: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n": int(len(sub))}
        for c in numeric:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors="coerce").dropna()
                out[c] = {
                    "mean": float(vals.mean()) if len(vals) else 0.0,
                    "std": float(vals.std()) if len(vals) else 0.0,
                    "min": float(vals.min()) if len(vals) else 0.0,
                    "max": float(vals.max()) if len(vals) else 0.0,
                }
        if "failed" in sub.columns:
            out["n_failed"] = int(sub["failed"].astype(bool).sum())
        if "incomplete" in sub.columns:
            out["n_incomplete"] = int(sub["incomplete"].astype(bool).sum())
        return out

    overall = _summ(df)
    per_scenario = {}
    if "scenario" in df.columns:
        for sc, sub in df.groupby("scenario"):
            per_scenario[str(sc)] = _summ(sub)

    # Flat convenience fields for viz labels
    def _mean(col: str) -> float:
        if col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").mean())

    return {
        "n_episodes": int(len(df)),
        "n_failed": int(df["failed"].astype(bool).sum()) if "failed" in df.columns else 0,
        "n_incomplete": int(df["incomplete"].astype(bool).sum())
        if "incomplete" in df.columns
        else 0,
        "overall": overall,
        "per_scenario": per_scenario,
        "label_avg_wait": _mean("avg_wait"),
        "label_avg_queue": _mean("avg_queue"),
        "label_throughput": _mean("throughput"),
        "label_gridlock_pct": 100.0
        * _mean("gridlock_duration")
        / max(_mean("avg_wait") * 0 + 360.0, 1.0),  # vs episode length default
        "algorithm": "Shared-IDQN",
    }


def clip_visual(value: float, lo: float, hi: float) -> float:
    """Clip for visualization only; does not alter stored metrics."""
    return float(min(max(value, lo), hi))


def normalize_visual(value: float, lo: float, hi: float) -> float:
    """Normalize to [0,1] after clipping to [lo,hi]."""
    v = clip_visual(value, lo, hi)
    if hi <= lo:
        return 0.0
    return (v - lo) / (hi - lo)
