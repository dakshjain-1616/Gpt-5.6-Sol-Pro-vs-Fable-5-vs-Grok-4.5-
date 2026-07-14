#!/usr/bin/env python3
"""Evaluate frozen Shared-IDQN checkpoint across scenarios and seeds."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.idqn import SharedIDQN
from src.env.traffic_env import TrafficEnv
from src.metrics.aggregate import REQUIRED_COLUMNS, aggregate_metrics, rows_to_dataframe, validate_row
from src.utils.config import ensure_dirs, load_config, resolve_path
from src.utils.seed import set_global_seed
from src.viz.map_plot import merge_viz_aggregates


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--scenarios", nargs="*", default=None)
    return p.parse_args()


def run_episode(env: TrafficEnv, agent: SharedIDQN, seed: int, scenario: str):
    agent.reset_latency_stats()
    obs = env.reset(seed=seed, scenario=scenario)
    done = False
    while not done:
        actions = agent.select_actions_greedy(obs)
        obs, rewards, done, info = env.step(actions)
    metrics = env.episode_metrics()
    metrics["policy_inference_latency_ms"] = agent.mean_inference_latency_ms()
    metrics["cpu_usage"] = metrics.pop("avg_cpu_percent", 0.0)
    # rename peak
    if "peak_memory_mb" not in metrics:
        metrics["peak_memory_mb"] = 0.0
    # ensure required fields
    metrics["failed"] = bool(metrics.get("failed", False))
    metrics["incomplete"] = bool(metrics.get("incomplete", False))
    return metrics, env.get_viz_data()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_global_seed(int(cfg.get("seed", 42)))

    ev = cfg.get("evaluation", {})
    n_seeds = args.n_seeds or int(ev.get("n_seeds", 20))
    seeds_start = int(ev.get("seeds_start", 1000))
    scenarios = args.scenarios or list(
        ev.get(
            "scenarios",
            [
                "normal",
                "high_demand",
                "sudden_surge",
                "uneven",
                "road_closure",
                "noisy_sensors",
                "missing_sensors",
                "partial_light_failure",
            ],
        )
    )
    ckpt = args.checkpoint or ev.get("checkpoint", "checkpoints/idqn_shared.pt")
    ckpt_path = resolve_path(ckpt)

    if args.smoke:
        n_seeds = 2
        scenarios = scenarios[:3]
        cfg["simulation"]["episode_steps"] = 60

    agent = SharedIDQN(cfg, device=cfg.get("device", "cpu"))
    if not ckpt_path.exists():
        print(f"WARNING: checkpoint missing {ckpt_path}; evaluating untrained policy")
    else:
        agent.load(ckpt_path)
        print(f"Loaded checkpoint {ckpt_path}")

    # Freeze epsilon effectively by greedy select
    agent.eps_start = 0.0
    agent.eps_end = 0.0

    rows = []
    viz_aggs = []
    t0 = time.time()
    for sc in scenarios:
        for i in range(n_seeds):
            seed = seeds_start + i
            env = TrafficEnv(cfg, scenario=sc, seed=seed)
            try:
                metrics, viz = run_episode(env, agent, seed, sc)
            except Exception as e:
                metrics = {
                    "scenario": sc,
                    "seed": seed,
                    "avg_wait": 0.0,
                    "avg_travel_time": 0.0,
                    "avg_queue": 0.0,
                    "p95_queue": 0.0,
                    "completed_trips": 0,
                    "throughput": 0.0,
                    "gridlock_duration": 0.0,
                    "gridlock_events": 0,
                    "policy_inference_latency_ms": 0.0,
                    "cpu_usage": 0.0,
                    "peak_memory_mb": 0.0,
                    "failed": True,
                    "incomplete": True,
                    "fail_reason": str(e),
                }
                viz = {}
            errs = validate_row(metrics)
            if errs:
                # fill missing
                for c in REQUIRED_COLUMNS:
                    metrics.setdefault(c, 0 if c not in ("failed", "incomplete", "scenario") else False)
                if "scenario" not in metrics:
                    metrics["scenario"] = sc
            rows.append(metrics)
            if viz:
                viz_aggs.append(viz)
            print(
                f"eval sc={sc} seed={seed} wait={metrics.get('avg_wait', 0):.2f} "
                f"q={metrics.get('avg_queue', 0):.2f} trips={metrics.get('completed_trips', 0)} "
                f"failed={metrics.get('failed')}"
            )

    df = rows_to_dataframe(rows)
    csv_path = resolve_path(cfg["paths"]["metrics_csv"])
    json_path = resolve_path(cfg["paths"]["metrics_json"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    agg = aggregate_metrics(df)
    # Better gridlock %: mean gridlock_duration / episode_steps
    ep_steps = float(cfg.get("simulation", {}).get("episode_steps", 360))
    if args.smoke:
        ep_steps = float(cfg["simulation"]["episode_steps"])
    gl_mean = float(df["gridlock_duration"].mean()) if len(df) else 0.0
    agg["label_gridlock_pct"] = 100.0 * gl_mean / max(ep_steps, 1.0)
    agg["algorithm"] = "Shared-IDQN"
    agg["n_seeds"] = n_seeds
    agg["scenarios"] = scenarios
    agg["checkpoint"] = str(ckpt_path)
    agg["elapsed_sec"] = time.time() - t0
    agg["csv"] = str(csv_path)

    # Save merged viz for visualize.py
    merged = merge_viz_aggregates(viz_aggs)
    viz_path = resolve_path(cfg["paths"]["artifact_dir"]) / "eval_viz_aggregate.json"
    # JSON-serialize node keys
    serial = {
        "lane_queue": merged["lane_queue"],
        "lane_congestion": merged["lane_congestion"],
        "lane_flow": merged["lane_flow"],
        "node_wait": {f"{k[0]},{k[1]}" if isinstance(k, tuple) else str(k): v for k, v in merged["node_wait"].items()},
        "gridlock_steps": merged["gridlock_steps"],
        "gridlock_events": merged["gridlock_events"],
        "completed_trips": merged["completed_trips"],
    }
    with open(viz_path, "w", encoding="utf-8") as f:
        json.dump(serial, f)
    agg["viz_aggregate"] = str(viz_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print("EVAL_DONE", f"rows={len(df)} csv={csv_path} json={json_path}")
    print(
        f"label wait={agg['label_avg_wait']:.2f} queue={agg['label_avg_queue']:.2f} "
        f"thr={agg['label_throughput']:.4f} gl%={agg['label_gridlock_pct']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
