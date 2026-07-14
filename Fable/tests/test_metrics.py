"""Tests: metric aggregation correctness and invalid-episode handling."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import REQUIRED_METRICS, aggregate, validate_row


def make_row(scenario="normal", seed=1, status="completed", **over):
    row = {
        "scenario": scenario, "seed": seed, "status": status, "error": "",
        "avg_waiting_time": 100.0, "avg_travel_time": 200.0,
        "avg_queue_length": 2.0, "p95_queue_length": 8.0,
        "completed_trips": 500, "throughput": 1000.0,
        "gridlock_duration": 0.0, "gridlock_events": 0,
        "inference_latency_ms": 0.5, "cpu_percent": 100.0,
        "peak_memory_mb": 400.0,
    }
    row.update(over)
    return row


class TestAggregation:
    def test_known_input_means(self):
        rows = [make_row(seed=1, avg_waiting_time=100.0),
                make_row(seed=2, avg_waiting_time=200.0)]
        agg = aggregate(rows)
        assert agg["overall"]["avg_waiting_time"]["mean"] == pytest.approx(150.0)
        assert agg["overall"]["avg_waiting_time"]["std"] == pytest.approx(50.0)
        assert agg["overall"]["avg_waiting_time"]["n"] == 2
        assert agg["episodes_total"] == 2
        assert agg["episodes_completed"] == 2
        assert agg["episodes_failed"] == 0

    def test_per_scenario_split(self):
        rows = [make_row("normal", 1, avg_queue_length=1.0),
                make_row("high_demand", 2, avg_queue_length=5.0)]
        agg = aggregate(rows)
        assert agg["per_scenario"]["normal"]["metrics"]["avg_queue_length"]["mean"] == 1.0
        assert agg["per_scenario"]["high_demand"]["metrics"]["avg_queue_length"]["mean"] == 5.0

    def test_all_required_metrics_present(self):
        agg = aggregate([make_row()])
        for k in REQUIRED_METRICS:
            assert k in agg["overall"], f"missing {k} in aggregate"

    def test_empty_rows_raise(self):
        with pytest.raises(ValueError):
            aggregate([])


class TestInvalidEpisodeHandling:
    def test_failed_episode_counted_not_dropped(self):
        rows = [make_row(seed=1),
                make_row(seed=2, status="failed", error="SimCrash: boom")]
        agg = aggregate(rows)
        assert agg["episodes_total"] == 2
        assert agg["episodes_failed"] == 1
        assert agg["failed_episodes"][0]["seed"] == 2
        assert agg["failed_episodes"][0]["error"] == "SimCrash: boom"
        # numeric means only over completed
        assert agg["overall"]["avg_waiting_time"]["n"] == 1

    def test_incomplete_episode_flagged(self):
        rows = [make_row(seed=1), make_row(seed=2, status="incomplete")]
        agg = aggregate(rows)
        assert agg["episodes_failed"] == 1
        assert agg["per_scenario"]["normal"]["failed"] == 1
        assert agg["per_scenario"]["normal"]["episodes"] == 2

    def test_validate_row_catches_missing_metric(self):
        row = make_row()
        del row["throughput"]
        assert any("missing:throughput" in p for p in validate_row(row))

    def test_validate_row_catches_nonfinite(self):
        assert validate_row(make_row(avg_waiting_time=float("nan")))
        assert validate_row(make_row(avg_queue_length=-1.0))

    def test_validate_row_ok(self):
        assert validate_row(make_row()) == []
