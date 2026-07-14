"""Seeded traffic-demand scenario specification and SUMO route-file generation.

The 4x4 grid has 16 fringe entry edges and 16 fringe exit edges. Demand is a
per-second non-homogeneous Poisson insertion process; every draw is made from a
seeded RNG so any (scenario, seed) pair regenerates byte-identical demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Fringe stub edges by compass side (entry = into grid, exit = out of grid).
ENTRY_EDGES = {
    "W": ["left0A0", "left1A1", "left2A2", "left3A3"],
    "E": ["right0D0", "right1D1", "right2D2", "right3D3"],
    "N": ["top0A3", "top1B3", "top2C3", "top3D3"],
    "S": ["bottom0A0", "bottom1B0", "bottom2C0", "bottom3D0"],
}
EXIT_EDGES = {
    "W": ["A0left0", "A1left1", "A2left2", "A3left3"],
    "E": ["D0right0", "D1right1", "D2right2", "D3right3"],
    "N": ["A3top0", "B3top1", "C3top2", "D3top3"],
    "S": ["A0bottom0", "B0bottom1", "C0bottom2", "D0bottom3"],
}
SIDES = ["N", "S", "E", "W"]
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


@dataclass
class ScenarioSpec:
    """Full seeded description of one episode's demand + perturbations."""

    name: str
    seed: int
    episode_length: int = 1500
    demand_end_margin: int = 300
    base_rate: float = 0.45            # vehicles/second network-wide
    demand_scale: float = 1.0
    directional_bias: dict = field(default_factory=lambda: {s: 1.0 for s in SIDES})
    surge: dict | None = None          # {start, end, mult}
    rate_wave_amp: float = 0.0         # sinusoidal arrival-rate variability [0,1)
    rate_wave_period: float = 600.0
    closure: dict | None = None        # {edges: [...], time: int}
    sensor_noise_sigma: float = 0.0
    sensor_dropout_p: float = 0.0
    tls_failure: dict | None = None    # {count: int, time: int}


def rate_at(spec: ScenarioSpec, t: int) -> float:
    """Arrival rate (veh/s) at sim-second t."""
    r = spec.base_rate * spec.demand_scale
    if spec.rate_wave_amp > 0:
        r *= 1.0 + spec.rate_wave_amp * np.sin(2 * np.pi * t / spec.rate_wave_period)
    if spec.surge and spec.surge["start"] <= t < spec.surge["end"]:
        r *= spec.surge["mult"]
    return max(r, 0.0)


def generate_route_file(spec: ScenarioSpec, out_path: str | Path) -> dict:
    """Write a SUMO route file with seeded random trips. Returns demand stats."""
    rng = np.random.default_rng(spec.seed)
    side_w = np.array([spec.directional_bias.get(s, 1.0) for s in SIDES], dtype=float)
    side_p = side_w / side_w.sum()

    trips = []
    t_end = spec.episode_length - spec.demand_end_margin
    vid = 0
    for t in range(t_end):
        n = rng.poisson(rate_at(spec, t))
        for _ in range(n):
            entry_side = rng.choice(SIDES, p=side_p)
            entry = ENTRY_EDGES[entry_side][rng.integers(4)]
            # Exit side: favour crossing the grid (opposite side) but allow any.
            exit_side = OPPOSITE[entry_side] if rng.random() < 0.55 else SIDES[rng.integers(4)]
            candidates = [e for e in EXIT_EDGES[exit_side]
                          if e != entry[::-1] and not _same_stub(entry, e)]
            exit_edge = candidates[rng.integers(len(candidates))]
            trips.append((float(t) + float(rng.random()), vid, entry, exit_edge))
            vid += 1
    trips.sort(key=lambda x: x[0])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        f.write('  <vType id="car" accel="2.6" decel="4.5" length="5.0" '
                'maxSpeed="13.89" sigma="0.5"/>\n')
        for depart, i, src, dst in trips:
            f.write(f'  <trip id="v{i}" type="car" depart="{depart:.2f}" '
                    f'from="{src}" to="{dst}" departLane="best" departSpeed="max"/>\n')
        f.write("</routes>\n")
    return {"num_trips": len(trips), "route_file": str(out_path)}


def _same_stub(entry_edge: str, exit_edge: str) -> bool:
    """True if the exit edge is the reverse stub of the entry edge (u-turn trip)."""
    for side in SIDES:
        for k, e in enumerate(ENTRY_EDGES[side]):
            if e == entry_edge:
                return EXIT_EDGES[side][k] == exit_edge
    return False


def sample_training_spec(episode_idx: int, master_seed: int, cfg: dict) -> ScenarioSpec:
    """Randomized-but-seeded training scenario: varies demand scale, directional
    bias, surge windows, arrival-rate waves and (implicitly) all routes."""
    seed = master_seed * 100003 + episode_idx
    rng = np.random.default_rng(seed)
    bias = {s: 1.0 for s in SIDES}
    kind = rng.random()
    if kind < 0.30:                      # uneven directional demand
        axis = ["N", "S"] if rng.random() < 0.5 else ["E", "W"]
        for s in axis:
            bias[s] = float(rng.uniform(1.8, 3.2))
        for s in set(SIDES) - set(axis):
            bias[s] = float(rng.uniform(0.4, 0.9))
    surge = None
    if 0.30 <= kind < 0.55:              # moderate surge window
        start = int(rng.integers(300, 800))
        surge = {"start": start, "end": start + int(rng.integers(150, 350)),
                 "mult": float(rng.uniform(1.8, 2.8))}
    return ScenarioSpec(
        name=f"train_ep{episode_idx}",
        seed=seed,
        episode_length=cfg["train"]["episode_length"],
        demand_end_margin=cfg["train"]["demand_end_margin"],
        base_rate=cfg["demand"]["base_rate"],
        demand_scale=float(rng.uniform(0.7, 1.5)),
        directional_bias=bias,
        surge=surge,
        rate_wave_amp=float(rng.uniform(0.0, 0.4)),
        rate_wave_period=float(rng.uniform(400, 900)),
    )


def build_eval_spec(scenario_name: str, seed: int, cfg: dict) -> ScenarioSpec:
    """Deterministic evaluation scenario from config."""
    sc = cfg["evaluation"]["scenarios"][scenario_name]
    return ScenarioSpec(
        name=scenario_name,
        seed=seed,
        episode_length=cfg["evaluation"]["episode_length"],
        demand_end_margin=cfg["evaluation"]["demand_end_margin"],
        base_rate=cfg["demand"]["base_rate"],
        demand_scale=float(sc.get("demand_scale", 1.0)),
        directional_bias={**{s: 1.0 for s in SIDES}, **sc.get("directional_bias", {})},
        surge=sc.get("surge"),
        closure=sc.get("closure"),
        sensor_noise_sigma=float(sc.get("sensor_noise_sigma", 0.0)),
        sensor_dropout_p=float(sc.get("sensor_dropout_p", 0.0)),
        tls_failure=sc.get("tls_failure"),
    )
