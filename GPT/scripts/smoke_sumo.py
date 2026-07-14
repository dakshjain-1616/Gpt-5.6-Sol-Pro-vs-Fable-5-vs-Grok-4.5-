#!/usr/bin/env python3
"""Headless Eclipse SUMO/TraCI smoke test with external signal control."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)

try:
    import sumo
    os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
    import sumolib
    import traci
except ImportError as exc:
    raise SystemExit(f"Missing pinned SUMO dependencies; run setup.sh: {exc}")

net_file = OUT / "tiny.net.xml"
sumo_bin = Path(os.environ["SUMO_HOME"]) / "bin" / "sumo"
netgenerate = Path(os.environ["SUMO_HOME"]) / "bin" / "netgenerate"
subprocess.run([str(netgenerate), "--grid", "--grid.number", "3", "--grid.length", "80",
                "--tls.set", "B1", "--no-turnarounds", "true", "-o", str(net_file)],
               check=True, capture_output=True, text=True)
net = sumolib.net.readNet(str(net_file), withPrograms=True)
edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
west = min(edges, key=lambda e: (e.getFromNode().getCoord()[0], -e.getToNode().getCoord()[0]))
east_candidates = sorted(edges, key=lambda e: e.getToNode().getCoord()[0], reverse=True)
route = None
for target in east_candidates:
    candidate, _ = net.getShortestPath(west, target)
    if candidate and len(candidate) >= 3:
        route = [e.getID() for e in candidate]
        break
if route is None:
    raise RuntimeError("Could not construct a multi-edge tiny-network route")

traci.start([str(sumo_bin), "-n", str(net_file), "--no-step-log", "true", "--no-warnings", "true",
             "--time-to-teleport", "-1", "--seed", "7"])
phase_commands, progression = [], []
try:
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        raise RuntimeError("Tiny network has no traffic signals")
    traci.route.add("smoke_route", route)
    traci.vehicle.add("smoke_vehicle", "smoke_route", depart="0")
    inserted = False
    arrived = 0
    for step in range(600):
        for tls_id in tls_ids:
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
            phase_count = len(logics[0].phases)
            phase = (step // 20) % phase_count
            traci.trafficlight.setPhase(tls_id, phase)
            if step % 20 == 0:
                phase_commands.append({"step": step, "tls": tls_id, "phase": phase,
                                       "state": traci.trafficlight.getRedYellowGreenState(tls_id)})
        traci.simulationStep()
        if "smoke_vehicle" in traci.vehicle.getIDList():
            inserted = True
            if step % 10 == 0:
                progression.append({"step": step, "road": traci.vehicle.getRoadID("smoke_vehicle"),
                                    "position": round(traci.vehicle.getLanePosition("smoke_vehicle"), 3),
                                    "speed": round(traci.vehicle.getSpeed("smoke_vehicle"), 3)})
        arrived += traci.simulation.getArrivedNumber()
        if arrived:
            total_steps = step + 1
            break
    else:
        total_steps = 600
finally:
    traci.close()

version = subprocess.run([str(sumo_bin), "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
record = {"status": "PASS" if inserted and arrived > 0 and len({p["road"] for p in progression}) > 1 else "FAIL",
          "sumo_version": version, "sumo_executable": str(sumo_bin), "network": str(net_file),
          "traffic_lights": len(tls_ids), "route_edges": route, "vehicle_inserted": inserted,
          "progression": progression, "externally_issued_phases": phase_commands,
          "arrived_vehicles": arrived, "total_steps": total_steps, "clean_exit": True}
(OUT / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
if record["status"] != "PASS":
    raise SystemExit("SUMO smoke verification failed")
