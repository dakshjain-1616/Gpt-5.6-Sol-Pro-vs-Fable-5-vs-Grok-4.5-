#!/usr/bin/env python3
"""Evaluate ONE trained checkpoint on 8 scenarios x N deterministic seeds.

No per-scenario retraining. Failed/incomplete episodes are recorded with a
status flag, never excluded.

Outputs:
    results/metrics.csv       one row per (scenario, seed) incl. failures
    results/aggregate.json    per-scenario + overall aggregates
    results/spatial.json      per-edge/per-TLS averages for visualization
    logs/eval.jsonl           structured episode log

Runs episodes in <=2 worker processes (libsumo allows one sim per process).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from src.metrics import REQUIRED_METRICS, aggregate, validate_row
from src.utils import JsonlLogger, load_config, set_global_seeds, setup_torch

CSV_COLUMNS = ["scenario", "seed", "status", "error"] + REQUIRED_METRICS + [
    "total_departed", "episode_time"]


def run_episode(job: tuple) -> dict:
    """Worker: run one greedy-policy episode. Returns a metrics row + spatial."""
    scenario, seed, ckpt, cfg_path = job
    cfg = load_config(cfg_path)
    setup_torch(cfg["runtime"]["torch_threads"])
    set_global_seeds(seed)

    from src.agent import DoubleDQNAgent
    from src.env import TrafficEnv
    from src.scenarios import build_eval_spec
    from src.utils import ResourceMonitor

    row = {"scenario": scenario, "seed": seed, "status": "failed", "error": ""}
    spatial = None
    try:
        agent = DoubleDQNAgent(cfg, seed=0)
        agent.load(ckpt)
        env = TrafficEnv(cfg, collect_metrics=True)
        monitor = ResourceMonitor()
        spec = build_eval_spec(scenario, seed, cfg)
        obs = env.reset(spec)
        done = False
        latencies = []
        decisions = 0
        while not done:
            actions, lat = agent.act(obs, greedy=True)
            latencies.append(lat)
            obs, _, done, _ = env.step(actions)
            decisions += 1
            if decisions % 60 == 0:
                monitor.sample()
        res = env.episode_result()
        env.close()
        summary = monitor.summary()
        row.update(res["metrics"])
        row["inference_latency_ms"] = round(1000 * float(np.mean(latencies)), 4)
        row["cpu_percent"] = summary["avg_cpu_percent"]
        row["peak_memory_mb"] = summary["peak_rss_mb"]
        # episode completeness check: sim must have reached its full length
        if res["metrics"]["episode_time"] < spec.episode_length:
            row["status"] = "incomplete"
            row["error"] = f"episode ended early at t={res['metrics']['episode_time']}"
        else:
            row["status"] = "completed"
        spatial = res["spatial"]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        try:
            env.close()
        except Exception:
            pass
    return {"row": row, "spatial": spatial}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = args.checkpoint or str(
        Path(cfg["paths"]["checkpoints_dir"]) / "model_best.pt")
    if not Path(ckpt).exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    workers = min(args.workers, cfg["runtime"]["max_concurrent_envs"])

    ev = cfg["evaluation"]
    scenario_names = args.scenarios or list(ev["scenarios"].keys())
    n_seeds = args.seeds or int(ev["seeds_per_scenario"])
    all_scenarios = list(cfg["evaluation"]["scenarios"].keys())
    jobs = []
    for sc in scenario_names:
        si = all_scenarios.index(sc)
        for k in range(n_seeds):
            seed = int(ev["seed_base"]) + si * 100 + k
            jobs.append((sc, seed, ckpt, args.config))

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(Path(cfg["paths"]["logs_dir"]) / "eval.jsonl")
    logger.log("eval_start", checkpoint=ckpt, episodes=len(jobs), workers=workers)
    print(f"Evaluating {ckpt} on {len(jobs)} episodes with {workers} workers")

    t0 = time.time()
    rows, spatials = [], []
    ctx = mp.get_context("spawn")  # fresh libsumo per worker
    with ctx.Pool(processes=workers, maxtasksperchild=8) as pool:
        for i, res in enumerate(pool.imap_unordered(run_episode, jobs)):
            row = res["row"]
            problems = validate_row(row) if row["status"] == "completed" else []
            if problems:
                row["status"] = "invalid_metrics"
                row["error"] = ";".join(problems)
            rows.append(row)
            if res["spatial"] and row["status"] == "completed":
                spatials.append(res["spatial"])
            logger.log("episode_done", i=i, scenario=row["scenario"],
                       seed=row["seed"], status=row["status"],
                       avg_wait=row.get("avg_waiting_time"),
                       elapsed=round(time.time() - t0, 1))
            if (i + 1) % 10 == 0 or i == len(jobs) - 1:
                print(f"  {i+1}/{len(jobs)} episodes done "
                      f"({time.time()-t0:.0f}s elapsed)")

    rows.sort(key=lambda r: (r["scenario"], r["seed"]))
    csv_path = results_dir / f"{args.out_prefix}metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    agg = aggregate(rows)
    agg["checkpoint"] = ckpt
    agg["algorithm"] = "Parameter-shared Double DQN"
    with open(results_dir / f"{args.out_prefix}aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

    # average spatial fields across completed episodes for visualization
    if spatials:
        edges = spatials[0]["edges"]
        sp_out = {
            "edges": edges,
            "edge_avg_occupancy": np.mean([s["edge_avg_occupancy"] for s in spatials],
                                          axis=0).tolist(),
            "edge_avg_queue": np.mean([s["edge_avg_queue"] for s in spatials],
                                      axis=0).tolist(),
            "edge_avg_vehicles": np.mean([s["edge_avg_vehicles"] for s in spatials],
                                         axis=0).tolist(),
            "tls_avg_wait": {tid: float(np.mean([s["tls_avg_wait"][tid] for s in spatials]))
                             for tid in spatials[0]["tls_avg_wait"]},
            "episodes_averaged": len(spatials),
        }
        with open(results_dir / f"{args.out_prefix}spatial.json", "w") as f:
            json.dump(sp_out, f, indent=2)

    logger.log("eval_end", episodes=len(rows),
               completed=agg["episodes_completed"], failed=agg["episodes_failed"],
               wall_clock_s=round(time.time() - t0, 1))
    logger.close()
    print(f"Wrote {csv_path} ({len(rows)} rows), aggregate.json, spatial.json in "
          f"{time.time()-t0:.0f}s. completed={agg['episodes_completed']} "
          f"failed={agg['episodes_failed']}")


if __name__ == "__main__":
    main()
