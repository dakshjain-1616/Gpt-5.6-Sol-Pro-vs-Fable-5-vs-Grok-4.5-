"""Episode metric collection and cross-episode aggregation.

Definitions (documented in REPORT.md):
- avg_waiting_time  : total halted vehicle-seconds / vehicles that entered the
                      network (seconds per vehicle).
- avg_travel_time   : mean insertion->arrival time of completed trips (s).
- avg_queue_length  : mean halted-vehicle count per monitored incoming lane,
                      averaged over all sim steps (vehicles).
- p95_queue_length  : 95th percentile over all (step, lane) halted counts.
- completed_trips   : number of vehicles that reached their destination.
- throughput        : completed trips per hour of simulated time.
- gridlock          : sustained interval (>= min_duration s) where >=
                      min_vehicles are present and network mean speed <
                      speed_threshold m/s. Reported as total duration (s) and
                      event count.
"""
from __future__ import annotations

import numpy as np

import libsumo

REQUIRED_METRICS = [
    "avg_waiting_time", "avg_travel_time", "avg_queue_length",
    "p95_queue_length", "completed_trips", "throughput",
    "gridlock_duration", "gridlock_events", "inference_latency_ms",
    "cpu_percent", "peak_memory_mb",
]


class EpisodeCollector:
    """Accumulates per-step statistics during one episode (eval/benchmark)."""

    def __init__(self, cfg: dict, edges: list[str], tls: dict):
        self.g = cfg["gridlock"]
        self.edges = edges
        self.tls = tls
        self.depart_times: dict[str, float] = {}
        self.travel_times: list[float] = []
        self.total_departed = 0
        self.total_halted_sec = 0.0
        self.queue_samples: list[np.ndarray] = []
        self._stall_run = 0.0
        self.gridlock_duration = 0.0
        self.gridlock_events = 0
        n_e = len(edges)
        self.edge_occ_sum = np.zeros(n_e)
        self.edge_queue_sum = np.zeros(n_e)
        self.edge_veh_sum = np.zeros(n_e)
        self.tls_wait_sum = {tid: 0.0 for tid in tls}
        self.steps = 0
        self.mono_lanes = [l for tid in tls for l in tls[tid]["in_lanes"]]

    def on_sim_step(self, t: float):
        self.steps += 1
        for vid in libsumo.simulation.getDepartedIDList():
            self.depart_times[vid] = t
            self.total_departed += 1
        for vid in libsumo.simulation.getArrivedIDList():
            d = self.depart_times.pop(vid, None)
            if d is not None:
                self.travel_times.append(t - d)

        lane_q = np.array([libsumo.lane.getLastStepHaltingNumber(l)
                           for l in self.mono_lanes], dtype=np.float32)
        self.queue_samples.append(lane_q)
        total_halted = 0
        veh_total, speed_weighted = 0, 0.0
        for j, e in enumerate(self.edges):
            n = libsumo.edge.getLastStepVehicleNumber(e)
            h = libsumo.edge.getLastStepHaltingNumber(e)
            occ = libsumo.edge.getLastStepOccupancy(e)
            self.edge_veh_sum[j] += n
            self.edge_queue_sum[j] += h
            self.edge_occ_sum[j] += occ
            total_halted += h
            veh_total += n
            if n > 0:
                speed_weighted += n * libsumo.edge.getLastStepMeanSpeed(e)
        self.total_halted_sec += total_halted
        mean_speed = speed_weighted / veh_total if veh_total > 0 else np.inf
        for tid in self.tls:
            w = sum(libsumo.lane.getWaitingTime(l) for l in self.tls[tid]["in_lanes"])
            h = sum(libsumo.lane.getLastStepHaltingNumber(l)
                    for l in self.tls[tid]["in_lanes"])
            self.tls_wait_sum[tid] += w / max(h, 1)

        # gridlock run tracking
        if veh_total >= self.g["min_vehicles"] and mean_speed < self.g["speed_threshold"]:
            self._stall_run += 1.0
        else:
            self._close_stall_run()

    def _close_stall_run(self):
        if self._stall_run >= self.g["min_duration"]:
            self.gridlock_events += 1
            self.gridlock_duration += self._stall_run
        self._stall_run = 0.0

    def finalize(self, episode_time: float) -> dict:
        self._close_stall_run()
        q = np.concatenate(self.queue_samples) if self.queue_samples else np.zeros(1)
        steps = max(self.steps, 1)
        hours = max(episode_time / 3600.0, 1e-9)
        m = {
            "avg_waiting_time": self.total_halted_sec / max(self.total_departed, 1),
            "avg_travel_time": float(np.mean(self.travel_times)) if self.travel_times else 0.0,
            "avg_queue_length": float(q.mean()),
            "p95_queue_length": float(np.percentile(q, 95)),
            "completed_trips": len(self.travel_times),
            "throughput": len(self.travel_times) / hours,
            "gridlock_duration": self.gridlock_duration,
            "gridlock_events": self.gridlock_events,
            "total_departed": self.total_departed,
            "episode_time": episode_time,
        }
        spatial = {
            "edges": self.edges,
            "edge_avg_occupancy": (self.edge_occ_sum / steps).tolist(),
            "edge_avg_queue": (self.edge_queue_sum / steps).tolist(),
            "edge_avg_vehicles": (self.edge_veh_sum / steps).tolist(),
            "tls_avg_wait": {tid: v / steps for tid, v in self.tls_wait_sum.items()},
        }
        return {"metrics": m, "spatial": spatial}


def validate_row(row: dict) -> list[str]:
    """Return list of problems with a metrics row (empty = valid)."""
    problems = []
    for k in REQUIRED_METRICS:
        if k not in row:
            problems.append(f"missing:{k}")
        else:
            try:
                v = float(row[k])
                if not np.isfinite(v) or v < 0:
                    problems.append(f"invalid:{k}={row[k]}")
            except (TypeError, ValueError):
                problems.append(f"non-numeric:{k}={row[k]}")
    if "status" not in row:
        problems.append("missing:status")
    return problems


def aggregate(rows: list[dict]) -> dict:
    """Aggregate per-episode rows into per-scenario and overall statistics.

    Failed/incomplete episodes are COUNTED and reported, and excluded only
    from the numeric means (their metrics may be partial); they are never
    dropped from the output.
    """
    if not rows:
        raise ValueError("no rows to aggregate")
    scenarios = sorted({r["scenario"] for r in rows})
    out = {"per_scenario": {}, "overall": {}, "episodes_total": len(rows)}
    ok_rows = [r for r in rows if r.get("status") == "completed"]
    out["episodes_completed"] = len(ok_rows)
    out["episodes_failed"] = len(rows) - len(ok_rows)
    out["failed_episodes"] = [
        {"scenario": r["scenario"], "seed": r["seed"], "status": r.get("status"),
         "error": r.get("error", "")}
        for r in rows if r.get("status") != "completed"
    ]

    def stats(sub: list[dict]) -> dict:
        agg = {}
        for k in REQUIRED_METRICS:
            vals = [float(r[k]) for r in sub if k in r and r[k] is not None
                    and np.isfinite(float(r[k]))]
            agg[k] = {"mean": float(np.mean(vals)) if vals else None,
                      "std": float(np.std(vals)) if vals else None,
                      "n": len(vals)}
        return agg

    for sc in scenarios:
        sub_all = [r for r in rows if r["scenario"] == sc]
        sub_ok = [r for r in sub_all if r.get("status") == "completed"]
        out["per_scenario"][sc] = {
            "episodes": len(sub_all),
            "completed": len(sub_ok),
            "failed": len(sub_all) - len(sub_ok),
            "metrics": stats(sub_ok),
        }
    out["overall"] = stats(ok_rows)
    return out
