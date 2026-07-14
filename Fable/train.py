#!/usr/bin/env python3
"""Train the parameter-shared Double DQN traffic-signal controller.

Usage:
    python train.py [--episodes N] [--resume] [--config config.yaml]

Checkpoints: checkpoints/model_latest.pt (every checkpoint_every episodes,
resumable) and checkpoints/model_best.pt (best mean episode reward).
Structured JSONL training log: logs/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.utils import (JsonlLogger, ResourceMonitor, load_config,
                       set_global_seeds, setup_torch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-name", default="model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_torch(cfg["runtime"]["torch_threads"])
    seed = int(cfg["train"]["master_seed"])
    set_global_seeds(seed)

    from src.agent import DoubleDQNAgent
    from src.env import TrafficEnv
    from src.scenarios import sample_training_spec

    n_episodes = args.episodes or int(cfg["train"]["num_episodes"])
    ck_dir = Path(cfg["paths"]["checkpoints_dir"])
    latest = ck_dir / f"{args.checkpoint_name}_latest.pt"
    best_path = ck_dir / f"{args.checkpoint_name}_best.pt"

    agent = DoubleDQNAgent(cfg, seed=seed)
    n_params = agent.num_parameters()
    assert n_params < 50_000, f"model too large: {n_params} params"

    start_ep, best_reward = 0, -np.inf
    if args.resume and latest.exists():
        extra = agent.load(latest)
        start_ep = int(extra.get("episode", -1)) + 1
        best_reward = float(extra.get("best_reward", -np.inf))
        print(f"Resumed from {latest} at episode {start_ep}")

    logger = JsonlLogger(Path(cfg["paths"]["logs_dir"]) / "train.jsonl")
    monitor = ResourceMonitor()
    logger.log("train_start", episodes=n_episodes, start_episode=start_ep,
               params=n_params, seed=seed)
    print(f"Model parameters: {n_params} (<50k OK). Episodes {start_ep}..{n_episodes-1}")

    env = TrafficEnv(cfg, collect_metrics=False)
    t_start = time.time()
    try:
        for ep in range(start_ep, n_episodes):
            spec = sample_training_spec(ep, seed, cfg)
            try:
                obs = env.reset(spec)
            except Exception as exc:  # failure recovery: skip broken episode
                logger.log("episode_error", episode=ep, error=str(exc))
                env.close()
                continue
            ep_reward, ep_loss, n_loss, decisions = 0.0, 0.0, 0, 0
            done = False
            while not done:
                actions, _ = agent.act(obs, greedy=False)
                nxt, rew, done, info = env.step(actions)
                agent.buffer.add_batch(obs, actions, rew, nxt, float(done))
                loss = agent.learn()
                if loss is not None:
                    ep_loss += loss
                    n_loss += 1
                ep_reward += float(rew.mean())
                obs = nxt
                decisions += 1
            mean_loss = ep_loss / max(n_loss, 1)
            res = monitor.sample()
            logger.log("episode", episode=ep, reward=round(ep_reward, 3),
                       loss=round(mean_loss, 5), epsilon=round(agent.epsilon(), 4),
                       decisions=decisions, buffer=agent.buffer.size,
                       scenario_seed=spec.seed, demand_scale=round(spec.demand_scale, 3),
                       elapsed_s=round(time.time() - t_start, 1), **res)
            print(f"ep {ep:4d} reward {ep_reward:9.2f} loss {mean_loss:8.5f} "
                  f"eps {agent.epsilon():.3f} elapsed {time.time()-t_start:7.0f}s")

            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(best_path, {"episode": ep, "reward": ep_reward,
                                       "best_reward": best_reward})
            if (ep + 1) % int(cfg["train"]["checkpoint_every"]) == 0 or ep == n_episodes - 1:
                agent.save(latest, {"episode": ep, "best_reward": best_reward})
    finally:
        env.close()
        agent.save(latest, {"episode": n_episodes - 1, "best_reward": best_reward})
        summary = monitor.summary()
        logger.log("train_end", best_reward=round(best_reward, 3),
                   wall_clock_s=round(time.time() - t_start, 1), **summary)
        logger.close()
    print(f"Done. best_reward={best_reward:.2f} "
          f"wall={time.time()-t_start:.0f}s peak_mem={summary['peak_rss_mb']}MB")


if __name__ == "__main__":
    main()
