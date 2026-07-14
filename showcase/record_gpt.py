#!/usr/bin/env python3
"""Record a FIXED-TIME baseline rollout on GPT's own 4x4 network.

IMPORTANT: GPT never implemented a learned controller — no model, no training,
no checkpoint. There is no policy to replay. What this records is GPT's network
and scenarios running under SUMO's DEFAULT STATIC signal program, i.e. ordinary
fixed-time traffic lights that sense nothing and learn nothing.

This is a baseline, NOT a trained result, and is labelled as such in the video.

It uses GPT's own code as far as that code goes:
  - network.build_network()  -> its 4x4 grid (16 signals)
  - network.make_routes()    -> its per-scenario demand
  - simulator.TrafficEnv.start() -> its SUMO launch + road-closure handling
Its TrafficEnv.step() is deliberately NOT used, because it requires a control
action; we advance SUMO directly so the static program runs untouched.

Output: showcase/traces/gpt_<scenario>.npz + showcase/traces/gpt_geom.json
Run with GPT/venv/bin/python.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GPT = Path(__file__).resolve().parents[1] / "GPT"
sys.path.insert(0, str(GPT))

import sumo  # noqa: E402
import os  # noqa: E402
os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)

import sumolib  # noqa: E402
import traci  # noqa: E402

from src.network import TLS_IDS, build_network  # noqa: E402
from src.simulator import TrafficEnv  # noqa: E402

SEED = 1000
DURATION = 720          # sim seconds
SAMPLE = 2              # record every 2 s -> 360 frames, matching the other two videos

# GPT's own scenario names (src/network.py SCENARIOS)
ORDER = ["normal", "uneven_directional", "high_demand", "sudden_surge",
         "road_closure", "noisy_sensors", "missing_sensors",
         "partial_signal_failure"]

# GPT models signal failure as: every 5th intersection can only ever use phase 0.
FAILED_IDX = [i for i in range(len(TLS_IDS)) if i % 5 == 0]


def export_geometry(net_file: Path, out: Path):
    net = sumolib.net.readNet(str(net_file))
    edges = [{"id": e.getID(),
              "shape": [[float(x), float(y)] for x, y in e.getShape()]}
             for e in net.getEdges()]
    tls_order = list(TLS_IDS)
    tls = {t: [float(v) for v in net.getNode(t).getCoord()] for t in tls_order}
    xs = [p[0] for e in edges for p in e["shape"]]
    ys = [p[1] for e in edges for p in e["shape"]]
    geom = {"edges": edges, "tls": tls, "tls_order": tls_order,
            "edge_ids": [e["id"] for e in edges],
            "bounds": [min(xs), min(ys), max(xs), max(ys)]}
    out.write_text(json.dumps(geom))
    return geom


def record(scenario: str, net_file: Path, work: Path, geom, outdir: Path):
    env = TrafficEnv(net_file, work, scenario, SEED, duration=DURATION)
    env.start()   # launches SUMO, applies road_closure if that's the scenario

    edge_ids = geom["edge_ids"]
    # first/second green phase index per signal, as GPT itself defines them
    g0 = {t: env.green_phases[t][0] for t in TLS_IDS}
    g1 = {t: env.green_phases[t][1] for t in TLS_IDS}

    veh_xy, veh_off = [], [0]
    phases, failed, equeue, nveh = [], [], [], []
    last = {t: 0 for t in TLS_IDS}
    fail_mask = [i in FAILED_IDX and scenario == "partial_signal_failure"
                 for i in range(len(TLS_IDS))]

    for step in range(DURATION):
        # a "failed" signal is stuck on its first green and never cycles
        if scenario == "partial_signal_failure":
            for i in FAILED_IDX:
                traci.trafficlight.setPhase(TLS_IDS[i], g0[TLS_IDS[i]])

        traci.simulationStep()   # static program drives every other signal

        if step % SAMPLE:
            continue

        ids = traci.vehicle.getIDList()
        pts = (np.array([[*traci.vehicle.getPosition(v), traci.vehicle.getSpeed(v)]
                         for v in ids], dtype=np.float32)
               if ids else np.zeros((0, 3), np.float32))
        veh_xy.append(pts)
        veh_off.append(veh_off[-1] + len(pts))

        ph = []
        for t in TLS_IDS:
            p = traci.trafficlight.getPhase(t)
            if p == g0[t]:
                last[t] = 0
            elif p == g1[t]:
                last[t] = 1
            ph.append(last[t])          # hold through yellow/all-red
        phases.append(ph)
        failed.append(fail_mask)
        equeue.append([traci.edge.getLastStepHaltingNumber(e) for e in edge_ids])
        nveh.append(len(ids))

    env.close()

    np.savez_compressed(
        outdir / f"gpt_{scenario}.npz",
        veh=np.concatenate(veh_xy) if veh_xy else np.zeros((0, 3), np.float32),
        veh_off=np.array(veh_off, np.int64),
        phase=np.array(phases, np.int8),
        failed=np.array(failed, np.bool_),
        equeue=np.array(equeue, np.float32),
        nveh=np.array(nveh, np.int32),
        scenario=np.array(scenario),
        seed=np.array(SEED),
    )
    print(f"  {scenario:<24} frames={len(nveh):4d}  peak_veh={max(nveh):4d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "traces"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "gptwork"
    work.mkdir(parents=True, exist_ok=True)

    netconvert = str(Path(sumo.SUMO_HOME) / "bin" / "netconvert")
    net_file = build_network(work, netconvert)
    geom = export_geometry(net_file, outdir / "gpt_geom.json")
    print(f"GPT network: {len(geom['edges'])} edges, {len(geom['tls'])} signals "
          f"(fixed-time / static program — no learned controller)", flush=True)

    for sc in ORDER:
        record(sc, net_file, work, geom, outdir)
    print("gpt traces done", flush=True)


if __name__ == "__main__":
    main()
