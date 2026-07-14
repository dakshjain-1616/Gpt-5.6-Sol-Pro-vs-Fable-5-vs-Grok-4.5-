#!/usr/bin/env python3
"""Benchmark 50,000 SUMO simulation steps with the DQN policy in the loop.

Measures steps/sec on this node and derives the training budget written into
config.yaml (train.num_episodes) targeting the configured wall-clock window.
Output: results/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ResourceMonitor, load_config, set_global_seeds, setup_torch

TARGET_HOURS = 3.0          # aim mid-window of the 2-4h budget
TRAIN_OVERHEAD = 1.35       # measured-forward-only -> training incl. backprop margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--write-config", action="store_true",
                    help="write derived num_episodes into config.yaml")
    args = ap.parse_args()

    cfg = load_config()
    setup_torch(cfg["runtime"]["torch_threads"])
    set_global_seeds(7)

    from src.agent import DoubleDQNAgent
    from src.env import TrafficEnv
    from src.scenarios import sample_training_spec

    agent = DoubleDQNAgent(cfg, seed=7)
    env = TrafficEnv(cfg, collect_metrics=False)
    monitor = ResourceMonitor()

    sim_steps = 0
    decision_interval = cfg["env"]["decision_interval"]
    lat_samples = []
    t0 = time.perf_counter()
    ep = 0
    while sim_steps < args.steps:
        spec = sample_training_spec(ep, 999, cfg)
        obs = env.reset(spec)
        done = False
        while not done and sim_steps < args.steps:
            actions, lat = agent.act(obs, greedy=True)
            obs, _, done, _ = env.step(actions)
            sim_steps += decision_interval
            lat_samples.append(lat)
            if sim_steps % 5000 < decision_interval:
                monitor.sample()
                print(f"  {sim_steps}/{args.steps} steps "
                      f"({sim_steps/(time.perf_counter()-t0):.0f} steps/s)")
        ep += 1
    env.close()
    elapsed = time.perf_counter() - t0
    steps_per_sec = sim_steps / elapsed

    ep_len = cfg["train"]["episode_length"]
    budget_steps = int(TARGET_HOURS * 3600 * steps_per_sec / TRAIN_OVERHEAD)
    num_episodes = max(budget_steps // ep_len, 10)
    projected_h = num_episodes * ep_len / steps_per_sec * TRAIN_OVERHEAD / 3600

    import numpy as np
    res_summary = monitor.summary()
    out = {
        "benchmark_steps": sim_steps,
        "elapsed_sec": round(elapsed, 1),
        "steps_per_sec": round(steps_per_sec, 1),
        "mean_inference_latency_ms": round(1000 * float(np.mean(lat_samples)), 3),
        "p95_inference_latency_ms": round(1000 * float(np.percentile(lat_samples, 95)), 3),
        "target_hours": TARGET_HOURS,
        "train_overhead_factor": TRAIN_OVERHEAD,
        "derived_num_episodes": int(num_episodes),
        "episode_length": ep_len,
        "projected_training_hours": round(projected_h, 2),
        **res_summary,
    }
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    if args.write_config:
        import re
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        text = cfg_path.read_text()
        text = re.sub(r"num_episodes: \d+.*", f"num_episodes: {num_episodes}"
                      "            # benchmark-derived budget", text, count=1)
        cfg_path.write_text(text)
        print(f"config.yaml updated: train.num_episodes={num_episodes}")


if __name__ == "__main__":
    main()
