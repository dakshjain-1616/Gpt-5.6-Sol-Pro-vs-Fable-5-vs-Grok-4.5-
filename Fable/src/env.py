"""Multi-intersection SUMO environment (headless, in-process via libsumo).

Controls all 16 traffic lights of the fixed 4x4 grid with a 4-phase action
space per intersection (NS-through, NS-left, EW-through, EW-left). Phase
changes insert an explicit 3 s yellow transition and respect min-green.

Only ONE TrafficEnv may be active per OS process (libsumo limitation);
parallelism is achieved with worker processes (<=2 concurrent).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import sumolib

import libsumo

from .metrics import EpisodeCollector
from .scenarios import ScenarioSpec, generate_route_file

TLS_IDS = [f"{c}{r}" for c in "ABCD" for r in range(4)]
NUM_PHASES = 4
PHASE_NAMES = ["NS_through", "NS_left", "EW_through", "EW_left"]


def _classify_link(net, junction_coord, in_lane: str, out_lane: str):
    """Return (approach_side, turn) for a controlled link using geometry."""
    jx, jy = junction_coord
    in_edge = net.getEdge(in_lane.rsplit("_", 1)[0])
    out_edge = net.getEdge(out_lane.rsplit("_", 1)[0])
    fx, fy = in_edge.getFromNode().getCoord()
    tx, ty = out_edge.getToNode().getCoord()
    dx, dy = jx - fx, jy - fy          # travel direction into junction
    ox, oy = tx - jx, ty - jy          # travel direction out of junction
    if abs(dy) >= abs(dx):
        approach = "N" if dy < 0 else "S"
    else:
        approach = "E" if dx < 0 else "W"
    ang = math.atan2(dx * oy - dy * ox, dx * ox + dy * oy)
    if abs(ang) < math.pi / 4:
        turn = "s"
    elif ang > 0:
        turn = "l"
    else:
        turn = "r"
    return approach, turn


def _build_phase_states(link_info) -> list[str]:
    """Build the 4 green-state strings for one TLS from link classification.

    Phase 0 NS-through : N/S straight+right 'G'
    Phase 1 NS-left    : N/S left 'G', N/S right permissive 'g'
    Phase 2 EW-through : E/W straight+right 'G'
    Phase 3 EW-left    : E/W left 'G', E/W right permissive 'g'
    """
    n = len(link_info)
    states = []
    for phase in range(NUM_PHASES):
        axis = ("N", "S") if phase < 2 else ("E", "W")
        want_left = phase % 2 == 1
        chars = []
        for approach, turn in link_info:
            if approach not in axis:
                chars.append("r")
            elif want_left:
                chars.append("G" if turn == "l" else ("g" if turn == "r" else "r"))
            else:
                chars.append("G" if turn in ("s", "r") else "r")
        states.append("".join(chars))
    assert all(len(s) == n for s in states)
    return states


def _yellow_state(cur: str, nxt: str) -> str:
    """Transition string: green links losing right-of-way become yellow."""
    out = []
    for c, nx in zip(cur, nxt):
        if c in "Gg" and nx not in "Gg":
            out.append("y")
        elif c in "Gg":
            out.append(c)
        else:
            out.append("r")
    return "".join(out)


class TrafficEnv:
    """Gym-style wrapper: obs [16, obs_dim], actions [16] in {0..3}."""

    def __init__(self, cfg: dict, collect_metrics: bool = True):
        self.cfg = cfg
        e = cfg["env"]
        self.decision_interval = int(e["decision_interval"])
        self.yellow_duration = int(e["yellow_duration"])
        self.min_green = int(e["min_green"])
        self.queue_norm = float(e["queue_norm"])
        self.tip_norm = float(e["time_in_phase_norm"])
        self.obs_dim = int(e["obs_dim"])
        self.pressure_coef = float(cfg["reward"]["pressure_coef"])
        self.waiting_coef = float(cfg["reward"]["waiting_coef"])
        self.net_file = cfg["paths"]["net_file"]
        self.collect_metrics = collect_metrics
        self.net = sumolib.net.readNet(self.net_file)
        self.all_edges = [ed.getID() for ed in self.net.getEdges()]
        self._built = False
        self.running = False
        self.spec: ScenarioSpec | None = None
        self.collector: EpisodeCollector | None = None

    # ------------------------------------------------------------------ setup
    def _build_tls_tables(self):
        """Requires a running simulation. Cached across episodes."""
        self.tls = {}
        for tid in TLS_IDS:
            links = libsumo.trafficlight.getControlledLinks(tid)
            in_lanes, out_lanes, info = [], [], []
            coord = self.net.getNode(tid).getCoord()
            for group in links:
                in_lane, out_lane, _via = group[0]
                info.append(_classify_link(self.net, coord, in_lane, out_lane))
                if in_lane not in in_lanes:
                    in_lanes.append(in_lane)
                if out_lane not in out_lanes:
                    out_lanes.append(out_lane)
            if len(in_lanes) != self.cfg["env"]["lanes_per_intersection"]:
                raise RuntimeError(f"{tid}: expected 8 incoming lanes, got {len(in_lanes)}")
            self.tls[tid] = {
                "in_lanes": in_lanes,
                "out_lanes": out_lanes,
                "green": _build_phase_states(info),
            }
        self._built = True

    # ------------------------------------------------------------------ reset
    def reset(self, spec: ScenarioSpec) -> np.ndarray:
        self.close()
        self.spec = spec
        routes_dir = Path(self.cfg["paths"]["routes_dir"])
        route_file = routes_dir / f"{spec.name}_{spec.seed}.rou.xml"
        generate_route_file(spec, route_file)
        args = [
            "sumo", "-n", self.net_file, "-r", str(route_file),
            "--no-step-log", "true", "--no-warnings", "true",
            "--seed", str(spec.seed % (2**31)),
            "--time-to-teleport", "300",
            "--step-length", "1.0",
        ]
        if spec.closure:  # rerouting so closures create pressure, not dead-ends
            args += ["--device.rerouting.probability", "0.5"]
        libsumo.start(args)
        self.running = True
        if not self._built:
            self._build_tls_tables()

        rng = np.random.default_rng(spec.seed + 777)
        self.noise_rng = np.random.default_rng(spec.seed + 888)
        n_lanes = self.cfg["env"]["lanes_per_intersection"]
        self.dropout_mask = np.ones((len(TLS_IDS), n_lanes), dtype=bool)
        if spec.sensor_dropout_p > 0:
            self.dropout_mask = rng.random((len(TLS_IDS), n_lanes)) >= spec.sensor_dropout_p
        self.failed_tls: set[str] = set()
        self._failure_pending = None
        if spec.tls_failure:
            k = int(spec.tls_failure["count"])
            picks = rng.choice(len(TLS_IDS), size=k, replace=False)
            self._failure_pending = {"time": int(spec.tls_failure["time"]),
                                     "tls": [TLS_IDS[i] for i in picks]}
        self._closure_pending = dict(spec.closure) if spec.closure else None

        self.phase = {tid: 0 for tid in TLS_IDS}
        self.time_in_phase = {tid: 0.0 for tid in TLS_IDS}
        for tid in TLS_IDS:
            libsumo.trafficlight.setRedYellowGreenState(tid, self.tls[tid]["green"][0])
        if self.collect_metrics:
            self.collector = EpisodeCollector(self.cfg, self.all_edges, self.tls)
        self.t = 0.0
        return self._observe()

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray):
        actions = np.asarray(actions)
        if actions.shape != (len(TLS_IDS),):
            raise ValueError(f"actions shape {actions.shape} != ({len(TLS_IDS)},)")
        if actions.min() < 0 or actions.max() >= NUM_PHASES:
            raise ValueError(f"actions out of range: {actions}")

        pending = {}
        for i, tid in enumerate(TLS_IDS):
            if tid in self.failed_tls:
                continue
            a = int(actions[i])
            if a != self.phase[tid] and self.time_in_phase[tid] >= self.min_green:
                cur = self.tls[tid]["green"][self.phase[tid]]
                nxt = self.tls[tid]["green"][a]
                libsumo.trafficlight.setRedYellowGreenState(tid, _yellow_state(cur, nxt))
                pending[tid] = a

        for k in range(self.decision_interval):
            if k == self.yellow_duration:
                self._commit_pending(pending)
                pending = {}
            self._apply_scheduled_events()
            libsumo.simulationStep()
            self.t = libsumo.simulation.getTime()
            for tid in TLS_IDS:
                self.time_in_phase[tid] += 1.0
            if self.collector is not None:
                self.collector.on_sim_step(self.t)
        if pending:  # decision_interval <= yellow_duration edge case
            self._commit_pending(pending)

        obs = self._observe()
        rewards = self._rewards()
        done = self.t >= self.spec.episode_length
        info = {"time": self.t, "vehicles": libsumo.vehicle.getIDCount(),
                "failed_tls": sorted(self.failed_tls)}
        return obs, rewards, done, info

    def _commit_pending(self, pending: dict):
        for tid, a in pending.items():
            libsumo.trafficlight.setRedYellowGreenState(tid, self.tls[tid]["green"][a])
            self.phase[tid] = a
            self.time_in_phase[tid] = 0.0

    def _apply_scheduled_events(self):
        if self._closure_pending and self.t >= self._closure_pending["time"]:
            for eid in self._closure_pending["edges"]:
                libsumo.edge.setDisallowed(eid, ["passenger"])
            self._closure_pending = None
        if self._failure_pending and self.t >= self._failure_pending["time"]:
            for tid in self._failure_pending["tls"]:
                self.failed_tls.add(tid)
                libsumo.trafficlight.setProgram(tid, "0")  # fixed-time fallback
            self._failure_pending = None

    # ----------------------------------------------------------- observations
    def _observe(self) -> np.ndarray:
        obs = np.zeros((len(TLS_IDS), self.obs_dim), dtype=np.float32)
        for i, tid in enumerate(TLS_IDS):
            lanes = self.tls[tid]["in_lanes"]
            q = np.array([libsumo.lane.getLastStepHaltingNumber(l) for l in lanes],
                         dtype=np.float32)
            occ = np.array([libsumo.lane.getLastStepOccupancy(l) for l in lanes],
                           dtype=np.float32)
            q = np.clip(q / self.queue_norm, 0.0, 2.0)
            occ = np.clip(occ, 0.0, 1.0)
            if self.spec.sensor_noise_sigma > 0:
                q += self.noise_rng.normal(0, self.spec.sensor_noise_sigma, q.shape)
                occ += self.noise_rng.normal(0, self.spec.sensor_noise_sigma, occ.shape)
                q, occ = np.clip(q, 0, 2), np.clip(occ, 0, 1)
            mask = self.dropout_mask[i]
            q, occ = q * mask, occ * mask
            ph = np.zeros(NUM_PHASES, dtype=np.float32)
            ph[self.phase[tid]] = 1.0
            tip = min(self.time_in_phase[tid] / self.tip_norm, 2.0)
            obs[i] = np.concatenate([q, occ, ph, [tip]])
        if not np.all(np.isfinite(obs)):
            raise ValueError("non-finite observation")
        return obs

    def _rewards(self) -> np.ndarray:
        r = np.zeros(len(TLS_IDS), dtype=np.float32)
        for i, tid in enumerate(TLS_IDS):
            inc = sum(libsumo.lane.getLastStepHaltingNumber(l)
                      for l in self.tls[tid]["in_lanes"])
            out = sum(libsumo.lane.getLastStepHaltingNumber(l)
                      for l in self.tls[tid]["out_lanes"])
            pressure = inc - out
            r[i] = -(self.pressure_coef * pressure + self.waiting_coef * inc)
        return r

    # ------------------------------------------------------------------ close
    def episode_result(self) -> dict:
        if self.collector is None:
            return {}
        return self.collector.finalize(self.t)

    def close(self):
        if self.running:
            try:
                libsumo.close()
            except Exception:
                pass
            self.running = False
