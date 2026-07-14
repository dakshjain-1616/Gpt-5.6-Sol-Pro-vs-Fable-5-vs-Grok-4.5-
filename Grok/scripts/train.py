#!/usr/bin/env python3
"""Train Shared I-DQN traffic signal controller."""
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--resume", default=None, help="checkpoint path to resume")
    p.add_argument("--smoke", action="store_true", help="tiny run for E2E smoke")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_global_seed(int(cfg.get("seed", 42)))

    train_cfg = cfg.get("training", {})
    total_steps = args.total_steps or int(train_cfg.get("total_env_steps", 80000))
    max_episodes = args.max_episodes or int(train_cfg.get("max_episodes", 250))
    scenarios = list(train_cfg.get("scenarios", ["normal", "uneven", "surge", "variable", "random_routes"]))
    probs = list(train_cfg.get("scenario_probs", [0.3, 0.2, 0.15, 0.2, 0.15]))
    if len(probs) != len(scenarios):
        probs = [1.0 / len(scenarios)] * len(scenarios)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()

    if args.smoke:
        total_steps = 400
        max_episodes = 2
        cfg["simulation"]["episode_steps"] = 80

    agent = SharedIDQN(cfg, device=cfg.get("device", "cpu"))
    ckpt_dir = resolve_path(cfg["paths"]["checkpoint_dir"])
    final_ckpt = ckpt_dir / "idqn_shared.pt"
    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume} env_steps={agent.env_steps}")

    log_path = resolve_path(cfg["paths"]["log_dir"]) / "train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(int(cfg.get("seed", 42)) + 7)
    env_steps = 0
    episode = 0
    t0 = time.time()
    ep_returns = []

    print(f"Training Shared-IDQN total_steps={total_steps} max_episodes={max_episodes}")

    while env_steps < total_steps and episode < max_episodes:
        sc = str(rng.choice(scenarios, p=probs))
        seed = int(cfg.get("seed", 42)) + episode * 17 + 3
        env = TrafficEnv(cfg, scenario=sc, seed=seed)
        obs = env.reset(seed=seed, scenario=sc)
        done = False
        ep_ret = 0.0
        ep_loss = []
        while not done and env_steps < total_steps:
            actions = agent.select_actions(obs, explore=True)
            next_obs, rewards, done, info = env.step(actions)
            agent.store_transition(obs, actions, rewards, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                ep_loss.append(loss)
            ep_ret += float(np.mean(rewards))
            obs = next_obs
            env_steps += 1
        episode += 1
        metrics = env.episode_metrics()
        ep_returns.append(ep_ret)
        row = {
            "episode": episode,
            "env_steps": env_steps,
            "scenario": sc,
            "seed": seed,
            "ep_return": ep_ret,
            "avg_loss": float(np.mean(ep_loss)) if ep_loss else None,
            "epsilon": agent.epsilon,
            "avg_wait": metrics["avg_wait"],
            "avg_queue": metrics["avg_queue"],
            "completed_trips": metrics["completed_trips"],
            "throughput": metrics["throughput"],
            "gridlock_events": metrics["gridlock_events"],
            "train_steps": agent.train_steps,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if episode % int(train_cfg.get("log_every", 10)) == 0 or args.smoke:
            print(
                f"ep={episode} steps={env_steps} sc={sc} ret={ep_ret:.2f} "
                f"wait={metrics['avg_wait']:.2f} q={metrics['avg_queue']:.2f} "
                f"done={metrics['completed_trips']} eps={agent.epsilon:.3f} "
                f"loss={row['avg_loss']}"
            )
        if episode % int(train_cfg.get("checkpoint_every", 50)) == 0:
            agent.save(ckpt_dir / f"idqn_ep{episode}.pt")

    agent.save(final_ckpt)
    elapsed = time.time() - t0
    summary = {
        "episodes": episode,
        "env_steps": env_steps,
        "train_steps": agent.train_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": env_steps / max(elapsed, 1e-6),
        "checkpoint": str(final_ckpt),
        "final_epsilon": agent.epsilon,
        "mean_ep_return_last10": float(np.mean(ep_returns[-10:])) if ep_returns else 0.0,
    }
    with open(resolve_path(cfg["paths"]["log_dir"]) / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("TRAIN_DONE", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
