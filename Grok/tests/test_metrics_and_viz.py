"""Tests for metric aggregation, image dimensions, and visual normalization."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.aggregate import (
    REQUIRED_COLUMNS,
    aggregate_metrics,
    clip_visual,
    normalize_visual,
    rows_to_dataframe,
    validate_row,
)
from src.viz.map_plot import merge_viz_aggregates, render_traffic_map


def test_required_columns_present_in_csv():
    csv_path = ROOT / "artifacts" / "metrics_per_seed.csv"
    assert csv_path.exists(), "metrics_per_seed.csv missing"
    df = pd.read_csv(csv_path)
    for c in REQUIRED_COLUMNS:
        assert c in df.columns, f"missing column {c}"
    assert len(df) >= 160, f"expected >=160 rows, got {len(df)}"
    assert df["scenario"].nunique() >= 8


def test_metric_aggregation_sanity():
    rows = [
        {
            "scenario": "normal",
            "seed": 1,
            "avg_wait": 10.0,
            "avg_travel_time": 50.0,
            "avg_queue": 2.0,
            "p95_queue": 5.0,
            "completed_trips": 100,
            "throughput": 0.5,
            "gridlock_duration": 0.0,
            "gridlock_events": 0,
            "policy_inference_latency_ms": 0.1,
            "cpu_usage": 50.0,
            "peak_memory_mb": 200.0,
            "failed": False,
            "incomplete": False,
        },
        {
            "scenario": "normal",
            "seed": 2,
            "avg_wait": 20.0,
            "avg_travel_time": 60.0,
            "avg_queue": 4.0,
            "p95_queue": 8.0,
            "completed_trips": 80,
            "throughput": 0.4,
            "gridlock_duration": 10.0,
            "gridlock_events": 1,
            "policy_inference_latency_ms": 0.2,
            "cpu_usage": 60.0,
            "peak_memory_mb": 220.0,
            "failed": False,
            "incomplete": False,
        },
        {
            "scenario": "high_demand",
            "seed": 1,
            "avg_wait": 30.0,
            "avg_travel_time": 70.0,
            "avg_queue": 6.0,
            "p95_queue": 12.0,
            "completed_trips": 50,
            "throughput": 0.2,
            "gridlock_duration": 20.0,
            "gridlock_events": 2,
            "policy_inference_latency_ms": 0.15,
            "cpu_usage": 70.0,
            "peak_memory_mb": 250.0,
            "failed": True,
            "incomplete": True,
        },
    ]
    df = rows_to_dataframe(rows)
    agg = aggregate_metrics(df)
    assert agg["n_episodes"] == 3
    assert agg["n_failed"] == 1
    assert agg["n_incomplete"] == 1
    assert abs(agg["label_avg_wait"] - 20.0) < 1e-6
    assert "normal" in agg["per_scenario"]
    assert agg["per_scenario"]["normal"]["n"] == 2
    assert abs(agg["per_scenario"]["normal"]["avg_wait"]["mean"] - 15.0) < 1e-6


def test_validate_row_missing_and_nonfinite():
    bad = {"scenario": "x", "seed": 1, "avg_wait": float("nan")}
    errs = validate_row(bad)
    assert any("missing" in e for e in errs)
    full = {c: 0.0 for c in REQUIRED_COLUMNS}
    full["scenario"] = "normal"
    full["seed"] = 1
    full["failed"] = False
    full["incomplete"] = False
    full["avg_wait"] = float("inf")
    errs2 = validate_row(full)
    assert any("nonfinite" in e for e in errs2)


def test_visual_normalization_and_clip():
    assert clip_visual(-5, 0, 100) == 0.0
    assert clip_visual(150, 0, 100) == 100.0
    assert clip_visual(50, 0, 100) == 50.0
    assert abs(normalize_visual(0, 0, 100) - 0.0) < 1e-9
    assert abs(normalize_visual(100, 0, 100) - 1.0) < 1e-9
    assert abs(normalize_visual(50, 0, 100) - 0.5) < 1e-9
    assert abs(normalize_visual(200, 0, 100) - 1.0) < 1e-9
    assert abs(normalize_visual(-10, 0, 40) - 0.0) < 1e-9
    # wait scale 0-180
    assert abs(normalize_visual(90, 0, 180) - 0.5) < 1e-9
    # queue scale 0-40
    assert abs(normalize_visual(20, 0, 40) - 0.5) < 1e-9


def test_final_png_dimensions():
    png = ROOT / "artifacts" / "final_traffic_map.png"
    assert png.exists(), "final_traffic_map.png missing"
    im = Image.open(png)
    assert im.size == (1600, 1600), f"expected 1600x1600, got {im.size}"


def test_render_deterministic(tmp_path):
    viz = {
        "lane_queue": {"L_0_0_0_1_E": 5.0, "L_0_1_0_0_W": 3.0},
        "lane_congestion": {"L_0_0_0_1_E": 40.0, "L_0_1_0_0_W": 20.0},
        "lane_flow": {"L_0_0_0_1_E": 1.0, "L_0_1_0_0_W": 0.5},
        "node_wait": {(0, 0): 10.0, (0, 1): 20.0, (1, 0): 5.0, (1, 1): 15.0},
        "gridlock_steps": 2,
        "gridlock_events": 1,
        "completed_trips": 100,
    }
    # pad remaining nodes for 4x4
    for r in range(4):
        for c in range(4):
            viz["node_wait"].setdefault((r, c), float(r + c))
    labels = {"avg_wait": 12.0, "avg_queue": 2.5, "throughput": 0.9, "gridlock_pct": 1.0}
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    render_traffic_map(viz, labels, p1, grid_size=4, width=1600, height=1600)
    render_traffic_map(viz, labels, p2, grid_size=4, width=1600, height=1600)
    im1 = Image.open(p1)
    im2 = Image.open(p2)
    assert im1.size == (1600, 1600)
    assert im2.size == (1600, 1600)
    assert list(im1.getdata()) == list(im2.getdata())


def test_merge_viz_aggregates():
    a = {
        "lane_queue": {"L1": 2.0},
        "lane_congestion": {"L1": 10.0},
        "lane_flow": {"L1": 1.0},
        "node_wait": {(0, 0): 4.0},
        "gridlock_steps": 2,
        "gridlock_events": 1,
        "completed_trips": 10,
    }
    b = {
        "lane_queue": {"L1": 6.0},
        "lane_congestion": {"L1": 30.0},
        "lane_flow": {"L1": 3.0},
        "node_wait": {(0, 0): 8.0},
        "gridlock_steps": 4,
        "gridlock_events": 1,
        "completed_trips": 20,
    }
    m = merge_viz_aggregates([a, b])
    assert abs(m["lane_queue"]["L1"] - 4.0) < 1e-9
    assert abs(m["node_wait"][(0, 0)] - 6.0) < 1e-9
    assert abs(m["gridlock_steps"] - 3.0) < 1e-9


def test_aggregate_json_exists():
    p = ROOT / "artifacts" / "metrics_aggregate.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data.get("n_episodes", 0) >= 160
    assert "overall" in data
    assert "per_scenario" in data
    assert data.get("algorithm") == "Shared-IDQN"


def test_checkpoint_exists():
    ckpt = ROOT / "checkpoints" / "idqn_shared.pt"
    assert ckpt.exists() and ckpt.stat().st_size > 1000


def test_benchmark_json():
    p = ROOT / "artifacts" / "benchmark_50k.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data.get("benchmark_steps") == 50000
    assert data.get("steps_per_sec", 0) > 0
