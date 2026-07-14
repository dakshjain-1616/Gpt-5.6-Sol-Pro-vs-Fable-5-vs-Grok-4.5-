"""
Pure-Python microscopic traffic simulator for a fixed 4x4 signalized grid.

Open-source component of this repository. Provides a CityFlow-like interface:
lane queues, waiting times, phase control, vehicle spawning/routing, and
deterministic seeding. No live rendering.

Network layout (intersections labeled by (r,c) with r,c in 0..G-1):
  - Bidirectional edges between adjacent intersections
  - Boundary edges act as entry/exit
  - Each intersection has 4 approaches: N, E, S, W (missing on edges)
  - Each approach has one incoming lane (aggregated for speed)
  - 2 phases: NS green (N+S through), EW green (E+W through)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

DIRS = ("N", "E", "S", "W")
DIR_DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}
# Phase 0: N and S green; Phase 1: E and W green
PHASE_GREEN_DIRS = {0: ("N", "S"), 1: ("E", "W")}


@dataclass
class Vehicle:
    vid: int
    lane_id: str
    position: float  # meters from lane start (upstream)
    speed: float
    route: List[str]  # remaining lane ids including current
    wait_time: float = 0.0
    travel_time: float = 0.0
    entered: bool = True


@dataclass
class Lane:
    lane_id: str
    length: float
    from_node: Optional[Tuple[int, int]]  # None = external source
    to_node: Tuple[int, int]
    direction: str  # direction of travel into to_node
    vehicles: List[int] = field(default_factory=list)  # vehicle ids ordered upstream->downstream
    is_entry: bool = False
    is_exit: bool = False


class PythonMicroSim:
    """Deterministic microscopic grid simulator."""

    def __init__(
        self,
        grid_size: int = 4,
        lane_length: float = 300.0,
        free_speed: float = 13.0,
        max_vehicles_per_lane: int = 40,
        dt: float = 1.0,
        seed: int = 0,
        spawn_rate: float = 0.08,
        closed_edges: Optional[List[str]] = None,
        failed_lights: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.G = int(grid_size)
        self.lane_length = float(lane_length)
        self.free_speed = float(free_speed)
        self.max_veh = int(max_vehicles_per_lane)
        self.dt = float(dt)
        self.spawn_rate = float(spawn_rate)
        self.closed_edges = set(closed_edges or [])
        self.failed_lights = set(tuple(x) for x in (failed_lights or []))

        self.rng = np.random.RandomState(int(seed))
        self.time = 0.0
        self.step_count = 0
        self._next_vid = 1

        self.lanes: Dict[str, Lane] = {}
        self.vehicles: Dict[int, Vehicle] = {}
        # Intersection phase: (r,c) -> phase index
        self.phases: Dict[Tuple[int, int], int] = {}
        self.phase_elapsed: Dict[Tuple[int, int], int] = {}
        self.yellow_remaining: Dict[Tuple[int, int], int] = {}
        self.pending_phase: Dict[Tuple[int, int], Optional[int]] = {}

        # Stats
        self.completed_trips = 0
        self.total_travel_time = 0.0
        self.total_wait_time = 0.0
        self.spawned = 0
        self.throughput_this_step = 0
        self.gridlock_steps = 0
        self.gridlock_events = 0
        self._was_gridlocked = False

        # Aggregates for visualization (per lane / node)
        self.lane_queue_sum: Dict[str, float] = {}
        self.lane_cong_sum: Dict[str, float] = {}
        self.lane_flow_sum: Dict[str, float] = {}
        self.node_wait_sum: Dict[Tuple[int, int], float] = {}
        self.agg_steps = 0

        self._build_network()
        self._reset_stats_buffers()

    # ------------------------------------------------------------------ network
    def _lane_id(self, fr: Optional[Tuple[int, int]], to: Tuple[int, int], d: str) -> str:
        if fr is None:
            return f"entry_{to[0]}_{to[1]}_{d}"
        return f"L_{fr[0]}_{fr[1]}_{to[0]}_{to[1]}_{d}"

    def _build_network(self) -> None:
        G = self.G
        # Internal bidirectional edges
        for r in range(G):
            for c in range(G):
                self.phases[(r, c)] = 0
                self.phase_elapsed[(r, c)] = 0
                self.yellow_remaining[(r, c)] = 0
                self.pending_phase[(r, c)] = None
                # neighbors
                for d, (dr, dc) in DIR_DELTA.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < G and 0 <= nc < G:
                        lid = self._lane_id((r, c), (nr, nc), d)
                        if lid in self.closed_edges:
                            continue
                        self.lanes[lid] = Lane(
                            lane_id=lid,
                            length=self.lane_length,
                            from_node=(r, c),
                            to_node=(nr, nc),
                            direction=d,
                        )
                # Entry lanes from outside into boundary nodes
                for d, (dr, dc) in DIR_DELTA.items():
                    nr, nc = r - dr, c - dc  # source outside
                    if not (0 <= nr < G and 0 <= nc < G):
                        # external entry into (r,c) from direction d
                        lid = self._lane_id(None, (r, c), d)
                        if lid in self.closed_edges:
                            continue
                        self.lanes[lid] = Lane(
                            lane_id=lid,
                            length=self.lane_length,
                            from_node=None,
                            to_node=(r, c),
                            direction=d,
                            is_entry=True,
                        )
                # Exit sinks: virtual exit lanes leaving boundary
                for d, (dr, dc) in DIR_DELTA.items():
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < G and 0 <= nc < G):
                        lid = f"exit_{r}_{c}_{d}"
                        self.lanes[lid] = Lane(
                            lane_id=lid,
                            length=self.lane_length,
                            from_node=(r, c),
                            to_node=(r, c),  # sink
                            direction=d,
                            is_exit=True,
                        )

        self.entry_lanes = [lid for lid, ln in self.lanes.items() if ln.is_entry]
        self.exit_lanes = [lid for lid, ln in self.lanes.items() if ln.is_exit]
        self.internal_lanes = [
            lid for lid, ln in self.lanes.items() if not ln.is_entry and not ln.is_exit
        ]

    def _reset_stats_buffers(self) -> None:
        self.lane_queue_sum = {lid: 0.0 for lid in self.lanes}
        self.lane_cong_sum = {lid: 0.0 for lid in self.lanes}
        self.lane_flow_sum = {lid: 0.0 for lid in self.lanes}
        self.node_wait_sum = {(r, c): 0.0 for r in range(self.G) for c in range(self.G)}
        self.agg_steps = 0

    # ------------------------------------------------------------------ control
    def set_tl_phase(self, node: Tuple[int, int], phase: int, yellow_time: int = 3) -> None:
        """Request phase change; yellow transition if switching."""
        if node in self.failed_lights:
            return  # stuck light
        phase = int(phase) % 2
        if self.yellow_remaining.get(node, 0) > 0:
            self.pending_phase[node] = phase
            return
        if phase == self.phases[node]:
            return
        if yellow_time > 0:
            self.yellow_remaining[node] = int(yellow_time)
            self.pending_phase[node] = phase
        else:
            self.phases[node] = phase
            self.phase_elapsed[node] = 0

    def get_phase(self, node: Tuple[int, int]) -> int:
        return self.phases[node]

    def is_yellow(self, node: Tuple[int, int]) -> bool:
        return self.yellow_remaining.get(node, 0) > 0

    # ------------------------------------------------------------------ queries
    def _incoming_lanes(self, node: Tuple[int, int]) -> List[str]:
        r, c = node
        out = []
        for d in DIRS:
            # lane arriving at node from direction d
            # either internal from neighbor or entry
            dr, dc = DIR_DELTA[d]
            fr = (r - dr, c - dc)
            if 0 <= fr[0] < self.G and 0 <= fr[1] < self.G:
                lid = self._lane_id(fr, node, d)
            else:
                lid = self._lane_id(None, node, d)
            if lid in self.lanes:
                out.append(lid)
        return out

    def lane_vehicle_count(self, lane_id: str) -> int:
        if lane_id not in self.lanes:
            return 0
        return len(self.lanes[lane_id].vehicles)

    def lane_queue_length(self, lane_id: str, speed_thresh: float = 0.5) -> int:
        if lane_id not in self.lanes:
            return 0
        q = 0
        for vid in self.lanes[lane_id].vehicles:
            v = self.vehicles[vid]
            if v.speed < speed_thresh:
                q += 1
        return q

    def lane_waiting_time(self, lane_id: str) -> float:
        if lane_id not in self.lanes:
            return 0.0
        return float(sum(self.vehicles[vid].wait_time for vid in self.lanes[lane_id].vehicles))

    def intersection_pressure(self, node: Tuple[int, int]) -> float:
        """Max-pressure style: sum of incoming queues."""
        return float(sum(self.lane_queue_length(lid) for lid in self._incoming_lanes(node)))

    def get_global_stats(self) -> Dict:
        queues = [self.lane_queue_length(lid) for lid in self.lanes if not self.lanes[lid].is_exit]
        waits = [self.vehicles[vid].wait_time for vid in self.vehicles]
        return {
            "n_vehicles": len(self.vehicles),
            "completed_trips": self.completed_trips,
            "spawned": self.spawned,
            "avg_queue": float(np.mean(queues)) if queues else 0.0,
            "p95_queue": float(np.percentile(queues, 95)) if queues else 0.0,
            "avg_wait": float(np.mean(waits)) if waits else 0.0,
            "total_wait": float(sum(waits)) if waits else 0.0,
            "gridlock_steps": self.gridlock_steps,
            "gridlock_events": self.gridlock_events,
            "throughput_step": self.throughput_this_step,
            "time": self.time,
            "step": self.step_count,
        }

    # ------------------------------------------------------------------ routing
    def _random_route_from(self, entry_lane_id: str, max_hops: int = 12) -> List[str]:
        """Build a multi-intersection route starting from an entry lane."""
        lane = self.lanes[entry_lane_id]
        route = [entry_lane_id]
        node = lane.to_node
        visited_nodes = {node}
        hops = 0
        while hops < max_hops:
            # Prefer continuing straight, else turn, eventually exit
            candidates = []
            for d, (dr, dc) in DIR_DELTA.items():
                nr, nc = node[0] + dr, node[1] + dc
                if 0 <= nr < self.G and 0 <= nc < self.G:
                    lid = self._lane_id(node, (nr, nc), d)
                    if lid in self.lanes and lid not in self.closed_edges:
                        candidates.append((lid, (nr, nc), False))
                else:
                    lid = f"exit_{node[0]}_{node[1]}_{d}"
                    if lid in self.lanes:
                        candidates.append((lid, node, True))
            if not candidates:
                break
            # Bias toward unvisited and exits after a few hops
            weights = []
            for lid, nxt, is_exit in candidates:
                w = 1.0
                if is_exit:
                    w = 0.5 + 0.4 * hops
                elif nxt in visited_nodes:
                    w = 0.2
                else:
                    w = 1.5
                weights.append(w)
            weights = np.array(weights, dtype=np.float64)
            weights /= weights.sum()
            idx = int(self.rng.choice(len(candidates), p=weights))
            lid, nxt, is_exit = candidates[idx]
            route.append(lid)
            if is_exit:
                break
            visited_nodes.add(nxt)
            node = nxt
            hops += 1
        return route

    # ------------------------------------------------------------------ dynamics
    def _spawn(self, rate_override: Optional[float] = None) -> None:
        rate = self.spawn_rate if rate_override is None else rate_override
        for lid in self.entry_lanes:
            if lid in self.closed_edges:
                continue
            lane = self.lanes[lid]
            if len(lane.vehicles) >= self.max_veh:
                continue
            # Headway: only spawn if first vehicle is far enough
            if lane.vehicles:
                lead = self.vehicles[lane.vehicles[0]]
                if lead.position < 12.0:
                    continue
            if self.rng.rand() < rate:
                route = self._random_route_from(lid)
                if len(route) < 2:
                    continue
                vid = self._next_vid
                self._next_vid += 1
                self.vehicles[vid] = Vehicle(
                    vid=vid,
                    lane_id=lid,
                    position=2.0,
                    speed=self.free_speed * 0.5,
                    route=route,
                )
                lane.vehicles.insert(0, vid)  # upstream end
                self.spawned += 1

    def _can_pass(self, node: Tuple[int, int], direction: str) -> bool:
        """Whether a vehicle arriving from `direction` may enter the intersection."""
        if self.yellow_remaining.get(node, 0) > 0:
            return False
        if node in self.failed_lights:
            # Failed light stuck on phase 0 (NS) forever
            return direction in PHASE_GREEN_DIRS[0]
        phase = self.phases[node]
        return direction in PHASE_GREEN_DIRS[phase]

    def _downstream_lane(self, veh: Vehicle) -> Optional[str]:
        if len(veh.route) < 2:
            return None
        return veh.route[1]

    def _move_vehicles(self) -> None:
        self.throughput_this_step = 0
        # Process lanes: move vehicles from downstream to upstream order
        # Snapshot lane vehicle lists
        lane_ids = list(self.lanes.keys())
        # Sort so we process near intersections first (downstream first within lane)
        for lid in lane_ids:
            lane = self.lanes[lid]
            if lane.is_exit:
                # Drain exit lane immediately
                for vid in list(lane.vehicles):
                    self._complete_vehicle(vid)
                lane.vehicles.clear()
                continue

            # Vehicles ordered upstream(index0) -> downstream(end)
            # Process from downstream end so space frees up
            new_order = list(lane.vehicles)
            n = len(new_order)
            for i in range(n - 1, -1, -1):
                vid = new_order[i]
                if vid not in self.vehicles:
                    continue
                veh = self.vehicles[vid]
                # Leader position
                if i < n - 1:
                    lead_vid = new_order[i + 1]
                    if lead_vid in self.vehicles:
                        lead_pos = self.vehicles[lead_vid].position
                    else:
                        lead_pos = lane.length + 100
                else:
                    lead_pos = lane.length + 100  # no leader in lane

                gap = lead_pos - veh.position - 7.0  # vehicle length + min gap
                # Desired speed
                v_des = self.free_speed
                if gap < 2.0:
                    v_des = 0.0
                elif gap < 15.0:
                    v_des = self.free_speed * max(0.0, (gap - 2.0) / 13.0)

                # Approaching stop line
                dist_to_stop = lane.length - veh.position
                at_stop = dist_to_stop < 15.0 and i == n - 1

                if at_stop and not lane.is_exit:
                    node = lane.to_node
                    if not self._can_pass(node, lane.direction):
                        # Must stop at stop line
                        if dist_to_stop < 8.0:
                            v_des = 0.0
                        else:
                            v_des = min(v_des, max(0.0, (dist_to_stop - 5.0) / self.dt))
                    else:
                        # Check downstream capacity
                        next_lid = self._downstream_lane(veh)
                        if next_lid is None:
                            v_des = 0.0
                        elif next_lid not in self.lanes:
                            v_des = 0.0
                        else:
                            next_lane = self.lanes[next_lid]
                            if len(next_lane.vehicles) >= self.max_veh:
                                v_des = 0.0
                            elif next_lane.vehicles:
                                # space at start of next lane
                                first = self.vehicles[next_lane.vehicles[0]]
                                if first.position < 10.0:
                                    v_des = 0.0

                veh.speed = v_des
                new_pos = veh.position + veh.speed * self.dt
                veh.travel_time += self.dt
                if veh.speed < 0.5:
                    veh.wait_time += self.dt

                # Cross intersection?
                if new_pos >= lane.length and not lane.is_exit:
                    node = lane.to_node
                    if self._can_pass(node, lane.direction) and veh.speed > 0.1:
                        next_lid = self._downstream_lane(veh)
                        if next_lid and next_lid in self.lanes:
                            next_lane = self.lanes[next_lid]
                            if len(next_lane.vehicles) < self.max_veh:
                                # Transfer
                                overflow = new_pos - lane.length
                                # remove from current
                                # mark for transfer
                                veh.lane_id = next_lid
                                veh.position = min(overflow, 5.0)
                                veh.route = veh.route[1:]
                                next_lane.vehicles.insert(0, vid)
                                new_order[i] = None  # remove later
                                self.lane_flow_sum[lid] = self.lane_flow_sum.get(lid, 0.0) + 1.0
                                if next_lane.is_exit:
                                    self._complete_vehicle(vid)
                                    next_lane.vehicles = [x for x in next_lane.vehicles if x != vid]
                                continue
                    # Can't cross — clamp at stop line
                    new_pos = min(new_pos, lane.length - 0.5)
                    veh.speed = 0.0
                    veh.wait_time += 0.0  # already counted

                veh.position = max(0.0, new_pos)

            # compact
            lane.vehicles = [v for v in new_order if v is not None and v in self.vehicles]

    def _complete_vehicle(self, vid: int) -> None:
        if vid not in self.vehicles:
            return
        veh = self.vehicles.pop(vid)
        self.completed_trips += 1
        self.total_travel_time += veh.travel_time
        self.total_wait_time += veh.wait_time
        self.throughput_this_step += 1
        # remove from any lane list
        for lane in self.lanes.values():
            if vid in lane.vehicles:
                lane.vehicles.remove(vid)

    def _update_phases(self) -> None:
        for node in list(self.phases.keys()):
            if self.yellow_remaining[node] > 0:
                self.yellow_remaining[node] -= 1
                if self.yellow_remaining[node] <= 0:
                    pend = self.pending_phase[node]
                    if pend is not None:
                        self.phases[node] = pend
                        self.pending_phase[node] = None
                    self.phase_elapsed[node] = 0
            else:
                self.phase_elapsed[node] += 1

    def _check_gridlock(self) -> None:
        # Gridlock if many lanes near capacity and low throughput
        n_jammed = 0
        total = 0
        for lid, lane in self.lanes.items():
            if lane.is_exit:
                continue
            total += 1
            if len(lane.vehicles) >= self.max_veh * 0.85:
                n_jammed += 1
        jammed_frac = n_jammed / max(total, 1)
        is_gl = jammed_frac > 0.35 and self.throughput_this_step == 0 and len(self.vehicles) > 20
        if is_gl:
            self.gridlock_steps += 1
            if not self._was_gridlocked:
                self.gridlock_events += 1
            self._was_gridlocked = True
        else:
            self._was_gridlocked = False

    def _accumulate_viz(self) -> None:
        self.agg_steps += 1
        for lid, lane in self.lanes.items():
            q = self.lane_queue_length(lid)
            n = len(lane.vehicles)
            cong = 100.0 * n / max(self.max_veh, 1)
            self.lane_queue_sum[lid] = self.lane_queue_sum.get(lid, 0.0) + q
            self.lane_cong_sum[lid] = self.lane_cong_sum.get(lid, 0.0) + cong
        for r in range(self.G):
            for c in range(self.G):
                node = (r, c)
                waits = []
                for lid in self._incoming_lanes(node):
                    for vid in self.lanes[lid].vehicles:
                        waits.append(self.vehicles[vid].wait_time)
                avg_w = float(np.mean(waits)) if waits else 0.0
                self.node_wait_sum[node] = self.node_wait_sum.get(node, 0.0) + avg_w

    def step(self, spawn_rate: Optional[float] = None) -> Dict:
        """Advance simulation one timestep. Returns global stats."""
        self._update_phases()
        self._spawn(rate_override=spawn_rate)
        self._move_vehicles()
        self._check_gridlock()
        self._accumulate_viz()
        self.time += self.dt
        self.step_count += 1
        return self.get_global_stats()

    def reset(self, seed: Optional[int] = None, spawn_rate: Optional[float] = None) -> None:
        if seed is not None:
            self.rng = np.random.RandomState(int(seed))
        if spawn_rate is not None:
            self.spawn_rate = float(spawn_rate)
        self.time = 0.0
        self.step_count = 0
        self._next_vid = 1
        self.vehicles.clear()
        for lane in self.lanes.values():
            lane.vehicles.clear()
        for node in self.phases:
            self.phases[node] = 0
            self.phase_elapsed[node] = 0
            self.yellow_remaining[node] = 0
            self.pending_phase[node] = None
        self.completed_trips = 0
        self.total_travel_time = 0.0
        self.total_wait_time = 0.0
        self.spawned = 0
        self.throughput_this_step = 0
        self.gridlock_steps = 0
        self.gridlock_events = 0
        self._was_gridlocked = False
        self._reset_stats_buffers()

    def get_viz_aggregates(self) -> Dict:
        """Time-averaged queues/congestion/wait for visualization."""
        steps = max(self.agg_steps, 1)
        lane_q = {k: v / steps for k, v in self.lane_queue_sum.items()}
        lane_c = {k: v / steps for k, v in self.lane_cong_sum.items()}
        lane_f = {k: v / steps for k, v in self.lane_flow_sum.items()}
        node_w = {k: v / steps for k, v in self.node_wait_sum.items()}
        return {
            "lane_queue": lane_q,
            "lane_congestion": lane_c,
            "lane_flow": lane_f,
            "node_wait": node_w,
            "steps": steps,
            "gridlock_steps": self.gridlock_steps,
            "gridlock_events": self.gridlock_events,
            "completed_trips": self.completed_trips,
        }
