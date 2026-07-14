"""
TrafficEnv: Gym-like multi-agent wrapper around the pure-Python micro simulator.

Provides local observations, discrete phase actions, max-pressure-inspired rewards,
8 evaluation scenario modes, metric collection, and resource monitoring.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.sim.python_micro_sim import DIRS, PythonMicroSim
from src.utils.resources import ResourceMonitor


SCENARIOS = (
    "normal",
    "high_demand",
    "sudden_surge",
    "uneven",
    "road_closure",
    "noisy_sensors",
    "missing_sensors",
    "partial_light_failure",
    # training-only aliases
    "surge",
    "variable",
    "random_routes",
)


class TrafficEnv:
    """Multi-agent traffic signal control environment (16 intersections)."""

    def __init__(self, cfg: Dict[str, Any], scenario: str = "normal", seed: int = 0) -> None:
        self.cfg = cfg
        self.scenario = scenario
        self.base_seed = int(seed)
        net = cfg.get("network", {})
        sim_cfg = cfg.get("simulation", {})
        dem = cfg.get("demand", {})
        self.G = int(net.get("grid_size", 4))
        self.n_agents = self.G * self.G
        self.nodes = [(r, c) for r in range(self.G) for c in range(self.G)]
        self.episode_steps = int(sim_cfg.get("episode_steps", 360))
        self.yellow_time = int(sim_cfg.get("yellow_time", 3))
        self.min_green = int(sim_cfg.get("min_green", 5))
        self.max_green = int(sim_cfg.get("max_green", 60))
        self.n_phases = int(cfg.get("phases", {}).get("n_phases", 2))
        self.obs_dim = int(cfg.get("agent", {}).get("obs_dim", 24))
        self.n_actions = int(cfg.get("agent", {}).get("n_actions", 2))

        self.base_spawn = float(dem.get("base_spawn_rate", 0.08))
        self.high_spawn = float(dem.get("high_spawn_rate", 0.18))
        self.surge_mult = float(dem.get("surge_multiplier", 3.0))
        self.surge_start = int(dem.get("surge_start", 120))
        self.surge_duration = int(dem.get("surge_duration", 60))

        rw = cfg.get("reward", {})
        self.w_pressure = float(rw.get("pressure_weight", 1.0))
        self.w_throughput = float(rw.get("throughput_weight", 0.5))
        self.w_wait = float(rw.get("wait_weight", 0.1))
        self.w_gridlock = float(rw.get("gridlock_weight", 2.0))
        self.gl_thresh = float(rw.get("gridlock_queue_threshold", 25))

        self.closed_edges: List[str] = []
        self.failed_lights: List[Tuple[int, int]] = []
        self.sensor_noise_std = 0.0
        self.missing_sensor_mask: Optional[np.ndarray] = None
        self.uneven_bias = 1.0
        self.variable_rate = False

        self._apply_scenario_static(scenario)

        self.sim = PythonMicroSim(
            grid_size=self.G,
            lane_length=float(net.get("lane_length", 300.0)),
            free_speed=float(net.get("free_speed", 13.0)),
            max_vehicles_per_lane=int(net.get("max_vehicles_per_lane", 40)),
            dt=float(sim_cfg.get("dt", 1.0)),
            seed=self.base_seed,
            spawn_rate=self.base_spawn,
            closed_edges=self.closed_edges,
            failed_lights=self.failed_lights,
        )

        self.t = 0
        self.done = False
        self.failed = False
        self.fail_reason = ""
        self.prev_completed = 0
        self.prev_pressure = {n: 0.0 for n in self.nodes}
        self.action_elapsed = {n: 0 for n in self.nodes}
        self.monitor = ResourceMonitor()

        # Episode metric accumulators
        self.metric_hist: Dict[str, List[float]] = {
            "avg_wait": [],
            "avg_queue": [],
            "p95_queue": [],
            "throughput": [],
            "n_vehicles": [],
        }
        self.travel_times: List[float] = []

    def _apply_scenario_static(self, scenario: str) -> None:
        s = scenario
        self.closed_edges = []
        self.failed_lights = []
        self.sensor_noise_std = 0.0
        self.missing_sensor_mask = None
        self.uneven_bias = 1.0
        self.variable_rate = False
        self.spawn_rate = self.base_spawn

        if s in ("normal", "random_routes"):
            self.spawn_rate = self.base_spawn
        elif s in ("high_demand",):
            self.spawn_rate = self.high_spawn
        elif s in ("sudden_surge", "surge"):
            self.spawn_rate = self.base_spawn
        elif s in ("uneven",):
            self.spawn_rate = self.base_spawn
            self.uneven_bias = 2.5
        elif s in ("road_closure",):
            self.spawn_rate = self.base_spawn
            # Close a central eastbound link
            self.closed_edges = ["L_1_1_1_2_E", "L_1_2_1_1_W"]
        elif s in ("noisy_sensors",):
            self.spawn_rate = self.base_spawn
            self.sensor_noise_std = 0.15
        elif s in ("missing_sensors",):
            self.spawn_rate = self.base_spawn
            # Mask half of observation dims for some agents (applied in obs)
            rng = np.random.RandomState(self.base_seed + 99)
            self.missing_sensor_mask = rng.rand(self.n_agents, self.obs_dim) > 0.35
        elif s in ("partial_light_failure",):
            self.spawn_rate = self.base_spawn
            self.failed_lights = [(1, 1), (2, 2)]
        elif s in ("variable",):
            self.spawn_rate = self.base_spawn
            self.variable_rate = True
        else:
            self.spawn_rate = self.base_spawn

    def _current_spawn_rate(self) -> float:
        s = self.scenario
        rate = self.spawn_rate
        if s in ("sudden_surge", "surge"):
            if self.surge_start <= self.t < self.surge_start + self.surge_duration:
                rate = self.base_spawn * self.surge_mult
            else:
                rate = self.base_spawn
        if self.variable_rate:
            # sinusoidal variation
            rate = self.base_spawn * (0.6 + 0.8 * (0.5 + 0.5 * np.sin(self.t / 40.0)))
        if self.uneven_bias != 1.0:
            # overall slightly higher; directional bias handled via entry selection in sim spawn
            # (approx: boost rate)
            rate = rate * 1.15
        return float(rate)

    # ------------------------------------------------------------------ API
    def reset(self, seed: Optional[int] = None, scenario: Optional[str] = None) -> np.ndarray:
        if scenario is not None:
            self.scenario = scenario
            self._apply_scenario_static(scenario)
            # rebuild sim if closures/failures changed
            net = self.cfg.get("network", {})
            sim_cfg = self.cfg.get("simulation", {})
            self.sim = PythonMicroSim(
                grid_size=self.G,
                lane_length=float(net.get("lane_length", 300.0)),
                free_speed=float(net.get("free_speed", 13.0)),
                max_vehicles_per_lane=int(net.get("max_vehicles_per_lane", 40)),
                dt=float(sim_cfg.get("dt", 1.0)),
                seed=int(seed if seed is not None else self.base_seed),
                spawn_rate=self.spawn_rate,
                closed_edges=self.closed_edges,
                failed_lights=self.failed_lights,
            )
        if seed is not None:
            self.base_seed = int(seed)
        self.sim.reset(seed=self.base_seed, spawn_rate=self.spawn_rate)
        self.t = 0
        self.done = False
        self.failed = False
        self.fail_reason = ""
        self.prev_completed = 0
        self.prev_pressure = {n: 0.0 for n in self.nodes}
        self.action_elapsed = {n: 0 for n in self.nodes}
        self.monitor = ResourceMonitor()
        self.metric_hist = {k: [] for k in self.metric_hist}
        self.travel_times = []
        return self._get_obs()

    def _get_local_obs(self, node: Tuple[int, int]) -> np.ndarray:
        """Build local observation vector for one intersection (obs_dim floats)."""
        sim = self.sim
        feats: List[float] = []
        # Incoming lane queues + counts + waits (4 directions) = 12
        for d_idx, d in enumerate(DIRS):
            lids = []
            for lid in sim._incoming_lanes(node):
                if sim.lanes[lid].direction == d:
                    lids.append(lid)
            if lids:
                lid = lids[0]
                q = sim.lane_queue_length(lid) / max(sim.max_veh, 1)
                n = sim.lane_vehicle_count(lid) / max(sim.max_veh, 1)
                w = min(sim.lane_waiting_time(lid) / 100.0, 2.0)
            else:
                q = n = w = 0.0
            feats.extend([q, n, w])
        # Phase one-hot + elapsed (normalized) + yellow flag = 2 + 1 + 1 = 4
        phase = sim.get_phase(node)
        phase_oh = [0.0] * self.n_phases
        phase_oh[int(phase) % self.n_phases] = 1.0
        feats.extend(phase_oh)
        feats.append(min(sim.phase_elapsed[node] / max(self.max_green, 1), 1.5))
        feats.append(1.0 if sim.is_yellow(node) else 0.0)
        # Neighbor pressure summary (4 neighbors) = 4
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < self.G and 0 <= nc < self.G:
                p = sim.intersection_pressure((nr, nc)) / (4.0 * sim.max_veh)
            else:
                p = 0.0
            feats.append(p)
        # Own pressure + n_vehicles global norm + time frac = 3
        own_p = sim.intersection_pressure(node) / (4.0 * sim.max_veh)
        feats.append(own_p)
        feats.append(min(len(sim.vehicles) / 200.0, 2.0))
        feats.append(self.t / max(self.episode_steps, 1))

        obs = np.asarray(feats, dtype=np.float32)
        if obs.shape[0] < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))
        elif obs.shape[0] > self.obs_dim:
            obs = obs[: self.obs_dim]

        # Sensor noise / missing
        if self.sensor_noise_std > 0:
            noise = np.random.RandomState(self.base_seed + self.t + hash(node) % 10000).randn(
                self.obs_dim
            ).astype(np.float32)
            obs = obs + self.sensor_noise_std * noise
        if self.missing_sensor_mask is not None:
            idx = self.nodes.index(node)
            mask = self.missing_sensor_mask[idx]
            obs = obs * mask.astype(np.float32)
        return np.clip(obs, -5.0, 5.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """Return stacked observations shape (n_agents, obs_dim)."""
        return np.stack([self._get_local_obs(n) for n in self.nodes], axis=0)

    def _reward_for_node(self, node: Tuple[int, int], throughput_delta: float, gridlock: bool) -> float:
        pressure = self.sim.intersection_pressure(node)
        # Max-pressure inspired: negative pressure, reward throughput, penalize waits/gridlock
        wait_sum = 0.0
        for lid in self.sim._incoming_lanes(node):
            wait_sum += self.sim.lane_waiting_time(lid)
        r = (
            -self.w_pressure * (pressure / 10.0)
            + self.w_throughput * throughput_delta
            - self.w_wait * (wait_sum / 50.0)
            - (self.w_gridlock if gridlock else 0.0)
        )
        # Small bonus for pressure reduction
        prev = self.prev_pressure[node]
        r += 0.2 * (prev - pressure) / 10.0
        self.prev_pressure[node] = pressure
        return float(np.clip(r, -20.0, 20.0))

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        actions: array shape (n_agents,) int phase selections.
        Returns: obs, rewards, done, info
        """
        if self.done:
            raise RuntimeError("Episode done; call reset()")

        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.shape[0] != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {actions.shape[0]}")

        # Apply actions with min-green / yellow constraints
        for i, node in enumerate(self.nodes):
            if node in self.sim.failed_lights:
                continue
            a = int(actions[i]) % self.n_actions
            cur = self.sim.get_phase(node)
            elapsed = self.sim.phase_elapsed[node]
            if self.sim.is_yellow(node):
                continue
            if a != cur:
                if elapsed >= self.min_green:
                    self.sim.set_tl_phase(node, a, yellow_time=self.yellow_time)
            else:
                # Force switch if max green exceeded
                if elapsed >= self.max_green:
                    self.sim.set_tl_phase(node, 1 - cur, yellow_time=self.yellow_time)

        rate = self._current_spawn_rate()
        try:
            stats = self.sim.step(spawn_rate=rate)
        except Exception as e:
            self.failed = True
            self.fail_reason = str(e)
            self.done = True
            obs = self._get_obs()
            rewards = np.zeros(self.n_agents, dtype=np.float32)
            return obs, rewards, True, self._info(failed=True)

        self.t += 1
        completed_delta = stats["completed_trips"] - self.prev_completed
        self.prev_completed = stats["completed_trips"]
        thr_share = completed_delta / max(self.n_agents, 1)
        gridlock = stats["gridlock_steps"] > 0 and self.sim._was_gridlocked

        rewards = np.array(
            [self._reward_for_node(n, thr_share, gridlock) for n in self.nodes],
            dtype=np.float32,
        )

        # Metrics
        self.metric_hist["avg_wait"].append(stats["avg_wait"])
        self.metric_hist["avg_queue"].append(stats["avg_queue"])
        self.metric_hist["p95_queue"].append(stats["p95_queue"])
        self.metric_hist["throughput"].append(float(completed_delta))
        self.metric_hist["n_vehicles"].append(float(stats["n_vehicles"]))
        self.monitor.sample()

        if self.t >= self.episode_steps:
            self.done = True
        # Hard fail: total gridlock for long period
        if stats["gridlock_steps"] > self.episode_steps * 0.5 and stats["n_vehicles"] > 50:
            # still complete episode but mark degraded
            pass

        obs = self._get_obs()
        return obs, rewards, self.done, self._info(failed=False)

    def _info(self, failed: bool = False) -> Dict[str, Any]:
        stats = self.sim.get_global_stats()
        res = self.monitor.summary()
        info = {
            "t": self.t,
            "scenario": self.scenario,
            "seed": self.base_seed,
            "failed": failed or self.failed,
            "fail_reason": self.fail_reason,
            "stats": stats,
            "resources": res,
            "completed_trips": stats["completed_trips"],
            "gridlock_steps": stats["gridlock_steps"],
            "gridlock_events": stats["gridlock_events"],
        }
        return info

    def episode_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics for the finished (or current) episode."""
        stats = self.sim.get_global_stats()
        res = self.monitor.summary()
        waits = self.metric_hist["avg_wait"]
        queues = self.metric_hist["avg_queue"]
        p95s = self.metric_hist["p95_queue"]
        thr = self.metric_hist["throughput"]

        avg_travel = 0.0
        if stats["completed_trips"] > 0:
            avg_travel = stats.get("time", self.t)  # fallback
            # better estimate from sim totals
            if self.sim.completed_trips > 0:
                avg_travel = self.sim.total_travel_time / self.sim.completed_trips

        gl_duration = float(stats["gridlock_steps"])
        gl_pct = 100.0 * gl_duration / max(self.t, 1)

        incomplete = not self.done or self.failed
        return {
            "scenario": self.scenario,
            "seed": self.base_seed,
            "steps": self.t,
            "avg_wait": float(np.mean(waits)) if waits else 0.0,
            "avg_travel_time": float(avg_travel),
            "avg_queue": float(np.mean(queues)) if queues else 0.0,
            "p95_queue": float(np.mean(p95s)) if p95s else 0.0,
            "completed_trips": int(stats["completed_trips"]),
            "throughput": float(stats["completed_trips"]) / max(self.t, 1),
            "gridlock_duration": gl_duration,
            "gridlock_events": int(stats["gridlock_events"]),
            "gridlock_pct": gl_pct,
            "peak_memory_mb": float(res.get("peak_memory_mb", 0.0)),
            "avg_cpu_percent": float(res.get("avg_cpu_percent", 0.0)),
            "failed": bool(self.failed),
            "incomplete": bool(incomplete and self.failed),
            "fail_reason": self.fail_reason,
            "spawned": int(stats["spawned"]),
            "final_vehicles": int(stats["n_vehicles"]),
        }

    def get_viz_data(self) -> Dict[str, Any]:
        return self.sim.get_viz_aggregates()
