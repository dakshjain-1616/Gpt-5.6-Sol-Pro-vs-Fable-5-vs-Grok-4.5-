#!/usr/bin/env python3
"""Replay Grok's trained Shared-IDQN checkpoint and record a rollout trace.

Runs the frozen policy (greedy) on the evaluated seeds and captures per-step
state: vehicle positions (reconstructed from lane geometry), signal phases,
and per-lane queue counts.

Output: showcase/traces/grok_<scenario>.npz  +  showcase/traces/grok_geom.json
Must be run with Grok/venv/bin/python.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GROK = Path(__file__).resolve().parents[1] / "Grok"
sys.path.insert(0, str(GROK))

from src.agent.idqn import SharedIDQN  # noqa: E402
from src.env.traffic_env import TrafficEnv  # noqa: E402
from src.sim.python_micro_sim import DIR_DELTA  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402

SEED = 1000
ORDER = ["normal", "uneven", "high_demand", "sudden_surge",
         "road_closure", "noisy_sensors", "missing_sensors", "partial_light_failure"]

SPACING = 300.0  # metres between adjacent intersections (matches lane_length)


def node_xy(node):
    """Grid node (row, col) -> plot coords. Row 0 is at the top."""
    r, c = node
    return np.array([c * SPACING, -r * SPACING], dtype=np.float64)


def lane_endpoints(sim, lid):
    """(start, end) coords of a lane. Entry lanes start outside the grid."""
    ln = sim.lanes[lid]
    end = node_xy(ln.to_node)
    if ln.from_node is None:
        dr, dc = DIR_DELTA[ln.direction]          # travel direction into to_node
        r, c = ln.to_node
        start = node_xy((r - dr, c - dc))         # virtual node one cell upstream
    else:
        start = node_xy(ln.from_node)
    return start, end


def export_geometry(sim, out: Path):
    lane_ids = list(sim.lanes.keys())
    lanes = []
    for lid in lane_ids:
        s, e = lane_endpoints(sim, lid)
        lanes.append({"id": lid, "start": s.tolist(), "end": e.tolist(),
                      "dir": sim.lanes[lid].direction,
                      "is_entry": bool(sim.lanes[lid].is_entry),
                      "is_exit": bool(sim.lanes[lid].is_exit)})
    nodes = {f"{r},{c}": node_xy((r, c)).tolist()
             for r in range(sim.G) for c in range(sim.G)}
    xs = [p for l in lanes for p in (l["start"][0], l["end"][0])]
    ys = [p for l in lanes for p in (l["start"][1], l["end"][1])]
    geom = {"lanes": lanes, "nodes": nodes, "lane_ids": lane_ids,
            "bounds": [min(xs), min(ys), max(xs), max(ys)]}
    out.write_text(json.dumps(geom))
    return geom


def record(scenario: str, cfg, ckpt: str, outdir: Path):
    set_global_seed(SEED)
    agent = SharedIDQN(cfg, device="cpu")
    agent.load(ckpt)
    env = TrafficEnv(cfg, scenario=scenario, seed=SEED)

    obs = env.reset(seed=SEED, scenario=scenario)
    sim = env.sim
    geom = export_geometry(sim, outdir / "grok_geom.json")
    lane_ids = geom["lane_ids"]
    lane_idx = {lid: i for i, lid in enumerate(lane_ids)}

    # Precompute unit vectors so vehicle position -> xy is a cheap lerp.
    starts = np.array([geom["lanes"][i]["start"] for i in range(len(lane_ids))])
    ends = np.array([geom["lanes"][i]["end"] for i in range(len(lane_ids))])

    veh_xy, veh_off = [], [0]
    phases, failed, lqueue, nveh, completed = [], [], [], [], []
    nodes = env.nodes

    done = False
    while not done:
        actions = agent.select_actions_greedy(obs)
        obs, _, done, info = env.step(actions)

        pts = []
        for v in sim.vehicles.values():
            i = lane_idx.get(v.lane_id)
            if i is None:
                continue
            frac = np.clip(v.position / sim.lanes[v.lane_id].length, 0.0, 1.0)
            xy = starts[i] + (ends[i] - starts[i]) * frac
            pts.append([xy[0], xy[1], v.speed])
        pts = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), np.float32)
        veh_xy.append(pts)
        veh_off.append(veh_off[-1] + len(pts))

        phases.append([sim.get_phase(n) for n in nodes])
        failed.append([n in sim.failed_lights for n in nodes])
        lqueue.append([sim.lane_queue(lid) if hasattr(sim, "lane_queue")
                       else sum(1 for vid in sim.lanes[lid].vehicles
                                if sim.vehicles[vid].speed < 0.1)
                       for lid in lane_ids])
        nveh.append(len(sim.vehicles))
        completed.append(sim.completed_trips)

    np.savez_compressed(
        outdir / f"grok_{scenario}.npz",
        veh=np.concatenate(veh_xy) if veh_xy else np.zeros((0, 3), np.float32),
        veh_off=np.array(veh_off, np.int64),
        phase=np.array(phases, np.int8),
        failed=np.array(failed, np.bool_),
        lqueue=np.array(lqueue, np.float32),
        nveh=np.array(nveh, np.int32),
        completed=np.array(completed, np.int32),
        nodes=np.array([list(n) for n in nodes], np.int32),
        scenario=np.array(scenario),
        seed=np.array(SEED),
    )
    print(f"  {scenario:<22} frames={len(nveh):4d}  peak_veh={max(nveh):4d}  "
          f"completed={completed[-1]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(GROK / "checkpoints" / "idqn_shared.pt"))
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "traces"))
    args = ap.parse_args()

    import os
    os.chdir(GROK)

    cfg = load_config(str(GROK / "configs" / "default.yaml"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for sc in ORDER:
        record(sc, cfg, args.checkpoint, outdir)
    print("grok traces done", flush=True)


if __name__ == "__main__":
    main()
