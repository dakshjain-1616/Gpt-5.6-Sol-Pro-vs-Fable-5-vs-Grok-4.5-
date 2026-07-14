#!/usr/bin/env python3
"""Benchmark 50k simulation steps on CPU; write artifacts/benchmark_50k.json."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.idqn import SharedIDQN
from src.env.traffic_env import TrafficEnv
from src.utils.config import ensure_dirs, load_config, resolve_path
from src.utils.seed import set_global_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--steps", type=int, default=50000)
    args = p.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_global_seed(int(cfg.get("seed", 42)))

    # Long episode for continuous stepping
    cfg = dict(cfg)
    cfg["simulation"] = dict(cfg.get("simulation", {}))
    cfg["simulation"]["episode_steps"] = args.steps + 10

    env = TrafficEnv(cfg, scenario="normal", seed=0)
    agent = SharedIDQN(cfg, device="cpu")
    obs = env.reset(seed=0, scenario="normal")

    n = args.steps
    t0 = time.time()
    for i in range(n):
        actions = agent.select_actions(obs, explore=True)
        obs, rewards, done, info = env.step(actions)
        agent.store_transition(obs, actions, rewards, obs, False)
        if i % 4 == 0:
            agent.train_step()
        if done:
            obs = env.reset(seed=i + 1, scenario="normal")
    elapsed = time.time() - t0
    sps = n / max(elapsed, 1e-9)

    # Practical budget: aim ~15-25 min training wall time on this node
    target_train_minutes = 18.0
    budget_steps = int(sps * target_train_minutes * 60.0)
    # Clamp to reasonable range
    budget_steps = int(min(max(budget_steps, 30000), 200000))
    ep_steps = int(cfg.get("simulation", {}).get("episode_steps", 360))
    # restore default episode length for budget episodes estimate
    ep_steps = int(load_config(args.config).get("simulation", {}).get("episode_steps", 360))
    budget_episodes = max(50, budget_steps // max(ep_steps, 1))

    result = {
        "benchmark_steps": n,
        "elapsed_sec": elapsed,
        "steps_per_sec": sps,
        "sim_only_note": "includes env.step + policy + occasional train_step",
        "recommended_total_env_steps": budget_steps,
        "recommended_max_episodes": budget_episodes,
        "episode_steps": ep_steps,
        "target_train_minutes": target_train_minutes,
        "device": "cpu",
        "n_intersections": 16,
    }
    out = resolve_path(cfg["paths"]["benchmark_json"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("BENCHMARK_DONE", json.dumps(result))

    # Update config training budget in place
    import yaml

    cfg_path = resolve_path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        full = yaml.safe_load(f)
    full.setdefault("training", {})
    full["training"]["total_env_steps"] = budget_steps
    full["training"]["max_episodes"] = budget_episodes
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(full, f, default_flow_style=False, sort_keys=False)
    print(f"Updated {cfg_path} total_env_steps={budget_steps} max_episodes={budget_episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
