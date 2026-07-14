# Technical Report: Shared Multi-Agent Independent DQN for Adaptive Traffic-Signal Control

## 1. Algorithm and Rationale

This system implements **Shared Multi-Agent Independent DQN (Shared I-DQN)** with a **max-pressure-inspired reward** for adaptive traffic-signal control on a fixed 4×4 urban grid (16 signalized intersections).

**Why this approach**
- Discrete phase selection matches traffic-light control naturally.
- Parameter sharing across 16 agents yields a tiny MLP that trains and infers efficiently on CPU-only hardware.
- Independent learners with local observations avoid the combinatorial explosion of a centralized joint action space (4^16).
- Experience replay and a target network provide stable off-policy learning without GPU.
- Max-pressure-style rewards connect the learning signal to established traffic-engineering objectives (reduce queue imbalance, maintain throughput, penalize gridlock).

**Rejected alternatives (briefly)**
- Graph-communication MARL (e.g., CoLight-style GAT): higher compute and memory risk on a 3–4 vCPU node.
- PPO/A2C continuous control: poorer sample efficiency for discrete phases under a tight CPU budget.
- Fully centralized DQN: intractable action space.
- Evolutionary search: too many simulation rollouts for the wall-time budget.

No LLM, VLM, external AI agent, or hosted model API is used.

## 2. Simulator

CityFlow was attempted (`pip install cityflow`) but no distribution was available on this host. The system therefore uses an **open-source pure-Python microscopic simulator** (`src/sim/python_micro_sim.py`) with a CityFlow-like control API:

- Fixed 4×4 grid, bidirectional internal edges, boundary entry/exit lanes.
- 16 controlled intersections; 2 non-conflicting phases (NS green, EW green) with yellow transitions.
- Vehicle-level car-following with gap-based speed, stop-line logic, multi-hop randomized routes.
- Deterministic `numpy.RandomState` seeding; no live rendering.
- Scenario hooks: spawn-rate changes, closed edges, failed lights stuck on a fixed phase.

Verified: 1000 steps without crash; ~110–150 env steps/sec including policy and light training on this node.

## 3. Observation, Action, Reward

### Observation (per intersection, dim = 24)
Local features only (partial observability by design):
- For each of 4 approaches: normalized queue, vehicle count, waiting time (12 dims).
- Phase one-hot (2), phase elapsed (1), yellow flag (1).
- Neighbor pressure summary for 4 neighbors (4).
- Own pressure, global vehicle load, episode time fraction (3).

Sensor-noise and missing-sensor scenarios corrupt or zero subsets of this vector at evaluation time.

### Action
Discrete phase index ∈ {0, 1} (NS / EW). Environment enforces:
- Minimum green time before a switch is accepted.
- Yellow interval on phase change.
- Maximum green force-switch to avoid permanent starvation.
- Failed lights ignore agent commands.

### Reward (per agent, each step)
```
r = -w_p * pressure/10
    + w_t * throughput_share
    - w_w * local_wait/50
    - w_g * 1[gridlock]
    + 0.2 * (prev_pressure - pressure)/10
```
with defaults `w_p=1.0`, `w_t=0.5`, `w_w=0.1`, `w_g=2.0`. Pressure is the sum of incoming queues (max-pressure style). Rewards are clipped to [-20, 20].

## 4. Architecture

**Q-network (shared across all intersections)**
```
Linear(obs_dim=24 → 64) → ReLU → Linear(64 → 64) → ReLU → Linear(64 → n_actions=2)
```
- Target network with hard copy every 200 gradient steps.
- Replay buffer capacity 50,000 transitions (each agent step stored independently).
- Optimizer: Adam, lr = 5e-4; Smooth L1 loss; grad clip 10.
- ε-greedy exploration: 1.0 → 0.05 over 30,000 env steps.

Checkpoint format: PyTorch state dict including Q, target, optimizer, train/env step counters, RNG state (`checkpoints/idqn_shared.pt`).

## 5. Training

**Mixture (generalization, not memorization)**  
Episodes sample scenarios with probabilities:
- normal 0.30, uneven 0.20, surge 0.15, variable arrivals 0.20, random_routes 0.15.

**Budget (from 50k-step benchmark)**  
- Benchmark: 50,000 steps in ~442 s ≈ **113 steps/sec** (env + policy + occasional train).
- Practical budget written to config: **122,057 env steps / 339 episodes** (~18 minutes target; actual full train ~837 s ≈ 14 minutes at ~146 steps/sec).
- Single concurrent env (≤2 allowed); CPU device only.

**Logging / resume**  
- Structured JSONL per episode (`logs/train.jsonl`), console log, `logs/train_summary.json`.
- Intermediate checkpoints every 50 episodes; final `checkpoints/idqn_shared.pt`.
- `scripts/train.py --resume <path>` supported.

## 6. Compute Optimizations

- Tiny shared MLP (~few tens of KB of parameters).
- Vectorized numpy observations; batched Q forward over 16 agents.
- No rendering in the training loop; matplotlib Agg only for final PNG.
- Hard target updates (no soft Polyak overhead).
- Episode length 360 s (sim seconds) balances signal and wall time.
- Two-phase lights (not 4-phase) reduce action complexity for CPU throughput.

## 7. Evaluation Methodology

- **Single frozen checkpoint** — no per-scenario fine-tuning.
- **8 scenarios × 20 deterministic seeds** (seeds 1000–1019) = **160 episodes**.
- Scenarios: normal, high_demand, sudden_surge, uneven, road_closure, noisy_sensors, missing_sensors, partial_light_failure.
- Metrics per seed/scenario (CSV + JSON aggregate):  
  avg wait, avg travel time, avg queue, 95th pct queue, completed trips, throughput, gridlock duration, gridlock events, policy inference latency (ms), CPU usage, peak memory (MB), failed/incomplete flags.
- Invalid episodes: exceptions caught, row recorded with `failed=True`, `incomplete=True`, `fail_reason` set; metrics zero-filled for schema stability.

**Headline aggregates (this run)**  
- 160 episodes, 0 failed, 0 incomplete.
- Overall mean avg wait ≈ 7.7 s, mean avg queue ≈ 0.30 veh, mean throughput ≈ 1.09 trips/step.
- Harder scenarios (missing sensors, partial light failure) show elevated wait/queue as expected; normal/high_demand maintain high completed trips.

## 8. Visualization

Exactly one PNG: `artifacts/final_traffic_map.png` (1600×1600), time-aggregated over all eval scenarios/seeds:

| Encoding | Meaning | Fixed scale (clip visuals only) |
|----------|---------|----------------------------------|
| Road color | Avg congestion | 0–100% |
| Road thickness | Avg queue | 0–40 veh |
| Node marker size | Avg wait | 0–180 s |
| Arrows | Dominant flow | — |
| Labels | Algorithm, avg wait, avg queue, throughput, gridlock % | gridlock 0–100% |

Dark neutral background, fixed top-down view, centered 4×4 network. Raw metrics in CSV/JSON are **not** clipped. Two successive `visualize.py` runs produce byte-identical PNGs.

## 9. Failure Cases and Robustness

- **Missing sensors**: large observation dropout → higher wait (~13 s) and queues; policy still completes trips.
- **Partial light failure**: two intersections stuck on NS → local congestion and wait (~19 s); network remains functional via alternate routes.
- **Noisy sensors**: mild degradation vs normal.
- **Road closure**: closed central link; throughput remains near normal via rerouting in the micro-sim.
- **Gridlock**: rare under trained policy on this demand range (gridlock % ≈ 0 in aggregate); reward and max-green force-switch mitigate permanent lock.
- Episode hard failures (sim exceptions) are recorded rather than crashing the eval loop.

## 10. Resource Usage

| Stage | Wall time (approx) | Notes |
|-------|--------------------|-------|
| Benchmark 50k steps | ~7.4 min | 113 steps/sec |
| Full training 122k steps | ~14 min | 146 steps/sec, peak RSS modest (CPU torch) |
| Full eval 160 episodes | ~same order as train fraction | greedy inference |
| Peak memory | tracked per episode via psutil | reported in metrics CSV |
| Inference latency | per-agent ms in CSV | shared MLP, batch of 16 |

Hardware: 4-core CPU, ~15.6 GB RAM, no GPU.

## 11. Limitations

- Pure-Python micro-sim is simplified (single lane per approach, 2 phases, no turning pockets, no pedestrian phases); absolute numbers are not calibrated to a real city.
- Independent DQN ignores explicit multi-agent credit assignment and communication; coordination is only via neighbor pressure features and shared weights.
- Demand models are synthetic (Bernoulli/sinusoidal spawns); real OD matrices and signal timing standards are not used.
- Training budget is wall-time constrained; longer training or prioritized replay might further improve hard scenarios.
- Visualization aggregates all scenarios into one map; scenario-specific maps are not produced.
- No comparison against fixed-time, actuated, or other RL controllers is reported (per project scope).

## 12. Reproducibility

- Global seeds in `configs/default.yaml`; eval seeds fixed 1000–1019.
- Checkpoint + config + scripts suffice to regenerate metrics and PNG.
- Tests cover metric aggregation, PNG dimensions (1600×1600), and visual normalization/clipping.
