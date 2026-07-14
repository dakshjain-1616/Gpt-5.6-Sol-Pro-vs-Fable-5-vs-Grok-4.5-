#!/usr/bin/env python3
"""Replay Fable's trained Double-DQN checkpoint and record a rollout trace.

Runs the frozen policy (greedy, no training) on the exact seeds that were used
in the original evaluation, and captures per-decision-step state:
vehicle positions, signal phases, and per-edge queue counts.

Output: showcase/traces/fable_<scenario>.npz  +  showcase/traces/fable_geom.json
Must be run with Fable/venv/bin/python (needs libsumo).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FABLE = Path(__file__).resolve().parents[1] / "Fable"
sys.path.insert(0, str(FABLE))

import libsumo  # noqa: E402
import sumolib  # noqa: E402

from src.agent import DoubleDQNAgent  # noqa: E402
from src.env import TLS_IDS, TrafficEnv  # noqa: E402
from src.scenarios import build_eval_spec  # noqa: E402
from src.utils import load_config, set_global_seeds, setup_torch  # noqa: E402

SEEDS = {
    "normal": 10000, "high_demand": 10100, "sudden_surge": 10200,
    "uneven_directional": 10300, "road_closure": 10400, "noisy_sensors": 10500,
    "missing_sensors": 10600, "partial_tls_failure": 10700,
}
ORDER = ["normal", "uneven_directional", "high_demand", "sudden_surge",
         "road_closure", "noisy_sensors", "missing_sensors", "partial_tls_failure"]


def export_geometry(cfg, out: Path):
    """Road-network geometry in SUMO coordinates (identical across all runs)."""
    net = sumolib.net.readNet(cfg["paths"]["net_file"])
    edges = []
    for e in net.getEdges():
        shape = e.getShape()
        edges.append({
            "id": e.getID(),
            "shape": [[float(x), float(y)] for x, y in shape],
        })
    tls = {t: [float(v) for v in net.getNode(t).getCoord()] for t in TLS_IDS}
    xs = [p[0] for e in edges for p in e["shape"]]
    ys = [p[1] for e in edges for p in e["shape"]]
    geom = {
        "edges": edges,
        "tls": tls,
        "bounds": [min(xs), min(ys), max(xs), max(ys)],
        "edge_ids": [e["id"] for e in edges],
    }
    out.write_text(json.dumps(geom))
    return geom


def record(scenario: str, seed: int, cfg, ckpt: str, geom, outdir: Path):
    set_global_seeds(seed)
    setup_torch(cfg["runtime"]["torch_threads"])

    agent = DoubleDQNAgent(cfg, seed=0)
    agent.load(ckpt)
    env = TrafficEnv(cfg, collect_metrics=False)
    spec = build_eval_spec(scenario, seed, cfg)

    edge_ids = geom["edge_ids"]
    veh_xy, veh_off = [], [0]
    phases, failed, equeue, times, nveh, arrived = [], [], [], [], [], []

    obs = env.reset(spec)
    done = False
    while not done:
        actions, _ = agent.act(obs, greedy=True)
        obs, _, done, info = env.step(actions)

        ids = libsumo.vehicle.getIDList()
        if ids:
            pts = np.array(
                [[*libsumo.vehicle.getPosition(v), libsumo.vehicle.getSpeed(v)] for v in ids],
                dtype=np.float32)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
        veh_xy.append(pts)
        veh_off.append(veh_off[-1] + len(pts))

        phases.append([env.phase[t] for t in TLS_IDS])
        failed.append([t in env.failed_tls for t in TLS_IDS])
        equeue.append([libsumo.edge.getLastStepHaltingNumber(e) for e in edge_ids])
        times.append(info["time"])
        nveh.append(info["vehicles"])
        arrived.append(libsumo.simulation.getArrivedNumber())

    env.close()

    np.savez_compressed(
        outdir / f"fable_{scenario}.npz",
        veh=np.concatenate(veh_xy) if veh_xy else np.zeros((0, 3), np.float32),
        veh_off=np.array(veh_off, np.int64),
        phase=np.array(phases, np.int8),
        failed=np.array(failed, np.bool_),
        equeue=np.array(equeue, np.float32),
        t=np.array(times, np.float32),
        nveh=np.array(nveh, np.int32),
        arrived=np.array(arrived, np.int32),
        scenario=np.array(scenario),
        seed=np.array(seed),
    )
    print(f"  {scenario:<22} frames={len(times):4d}  peak_veh={max(nveh):4d}  "
          f"completed={int(np.sum(arrived))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(FABLE / "checkpoints" / "model_best.pt"))
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "traces"))
    args = ap.parse_args()

    import os
    os.chdir(FABLE)  # config paths are relative to the Fable root

    cfg = load_config(str(FABLE / "config.yaml"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    geom = export_geometry(cfg, outdir / "fable_geom.json")
    print(f"geometry: {len(geom['edges'])} edges, {len(geom['tls'])} intersections", flush=True)

    for sc in ORDER:
        record(sc, SEEDS[sc], cfg, args.checkpoint, geom, outdir)
    print("fable traces done", flush=True)


if __name__ == "__main__":
    main()
