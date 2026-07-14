# Adaptive Traffic-Signal Control — Technical Report

## 1. Algorithm Selection: Parameter-Shared Double DQN

### Why this algorithm

The traffic-signal control problem is a sequential decision-making task under partial observability and stochastic demand, making it a natural fit for reinforcement learning (RL). Within the RL family, **Double Deep Q-Networks (Double DQN)** was chosen for three reasons:

1. **Discrete action space**: Each intersection selects among 4 discrete phases. DQN-family algorithms natively handle discrete actions, avoiding the complexity of policy-gradient methods or continuous-action wrappers.
2. **Off-policy sample efficiency**: Traffic simulation is computationally expensive (~574 steps/s on this node). Off-policy learning with a replay buffer reuses each transition multiple times, extracting more value from every simulation step compared to on-policy methods (e.g., A2C, PPO).
3. **Double DQN reduces overestimation bias**: Standard DQN uses the same network for action selection and Q-value estimation, which systematically overestimates action values. Double DQN decouples selection (online network) from evaluation (target network), producing more stable learning.

### Why parameter sharing

All 16 intersections in the 4×4 grid share a single neural network. The justification is:

- **Structural symmetry**: Every intersection has the same observation space (per-lane queue + occupancy, phase one-hot, time-in-phase) and action space (4 phases). The underlying control problem is identical.
- **Sample efficiency**: With 16 intersections contributing transitions to one shared replay buffer, the effective sample size per simulation step is 16× larger than a per-intersection policy. This is critical on a CPU-only node with only 4 cores.
- **Parameter budget**: A single shared MLP with 128×128 hidden units has only 19,844 parameters (<50k constraint), keeping inference <0.4 ms per batch of 16 intersections.

### What was not chosen and why

- **Fixed-time control**: No adaptation to traffic conditions; poor performance under variable demand.
- **Rule-based actuated control**: Requires manual tuning of per-lane thresholds; does not generalise to unseen patterns.
- **Independent per-intersection DQN**: 16 separate networks → 16× more parameters, 16× more inference cost, and each network sees only 1/16 of the data.
- **Multi-agent policy gradient (e.g., MAPPO, MADDPG)**: Much higher sample complexity; intractable on a 3-vCPU budget with a single simulator.
- **Model-based RL**: Requires learning a world model of traffic dynamics; high complexity and risk of compounding errors.

---

## 2. Observation Space

Each intersection `i` receives a local observation vector of dimension 21:

| Feature | Dimensions | Normalisation | Description |
|---|---|---|---|
| Per-lane queue count | 8 | Clipped `[0, 2]` after dividing by 10.0 | Number of halted vehicles on each of the 8 incoming lanes |
| Per-lane occupancy | 8 | Clipped `[0, 1]` | Fraction of lane length occupied by vehicles (from SUMO, already in [0,1]) |
| Phase one-hot | 4 | — | One-hot encoding of the currently active green phase (0: NS-through, 1: NS-left, 2: EW-through, 3: EW-left) |
| Time-in-phase | 1 | Clipped `[0, 2]` after dividing by 60.0 | Seconds since the current phase was activated |

**Design rationale**: Queue length and occupancy together capture both demand pressure and utilisation. The phase one-hot and time-in-phase give the agent awareness of the signal's current state and commitment, enabling it to respect min-green constraints implicitly through the observation.

---

## 3. Action Space

Each intersection selects one of 4 discrete actions:

| Action | Meaning |
|---|---|
| 0 | Green for NS-through (N/S straights + rights) |
| 1 | Green for NS-left (N/S left turns) |
| 2 | Green for EW-through (E/W straights + rights) |
| 3 | Green for EW-left (E/W left turns) |

The action is taken every 5 simulation seconds (the `decision_interval`). Two safety constraints are enforced by the environment, not learned:

- **Min-green**: A phase must be active for at least 5 seconds before a switch is allowed.
- **Yellow insertion**: When a switch occurs, the outgoing phase is set to yellow for 3 seconds before the new green phase starts. This models real-world amber phases and prevents unsafe immediate transitions.

Both constraints are hard-coded in `src/env.py` and applied regardless of the agent's action.

---

## 4. Reward Function

The per-intersection reward at each decision step is:

```
r_i = -(0.05 × pressure_i + 0.02 × waiting_i)
```

Where:

- **`pressure_i`** = (halted vehicles on incoming lanes) − (halted vehicles on outgoing lanes). Positive pressure means the intersection is accumulating vehicles; negative pressure means it is clearing. This is the standard "pressure" objective from the PressLight/MPLight literature.
- **`waiting_i`** = halted vehicles on incoming lanes. This adds a penalty for standing queues, complementing the differential pressure term.

The reward is always negative because there is always some waiting traffic in a non-trivial scenario. The agent learns to minimise waiting (i.e., make the reward as close to zero as possible).

**Why negative pressure**: The pressure formulation (incoming minus outgoing) naturally encourages coordination — if an intersection is congested, turning on phases that move vehicles *out* of the congested links reduces the pressure on downstream neighbours.

---

## 5. Model Architecture

```
QNet(
  (net): Sequential(
    (0): Linear(21 → 128)
    (1): ReLU()
    (2): Linear(128 → 128)
    (3): ReLU()
    (4): Linear(128 → 4)
  )
)
```

- **Input**: 21-dimensional observation per intersection
- **Hidden layers**: Two fully-connected layers of 128 units each, with ReLU activations
- **Output**: 4 Q-values (one per action)
- **Total parameters**: 19,844 (verified <50k constraint)
- **Target network**: An identical copy of `QNet`, updated every 500 gradient steps by copying the online network's weights (hard update)

The network is **shared** across all 16 intersections. A single forward pass processes all 16 observations as a batch of size 16, producing a 16×4 Q-value tensor.

---

## 6. Training Procedure

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Hidden sizes | [128, 128] | 19,844 params |
| Discount factor (γ) | 0.95 | Moderate discount for 1500-s episodes |
| Learning rate | 5×10⁻⁴ | Adam optimiser |
| Replay buffer size | 100,000 transitions | ~1.6 M transitions from 16 intersections |
| Batch size | 128 | Transitions sampled uniformly |
| Target update interval | 500 gradient steps | Hard copy of online → target |
| ε-start | 1.0 | Full exploration initially |
| ε-end | 0.05 | Minimal exploration at convergence |
| ε-decay | 40,000 decisions | Anneals over ~133 episodes (16 intersections × 300 decisions/episode) |
| Learning starts | 2,000 transitions | Fill buffer before first gradient step |
| Gradient clip | 5.0 | Prevent exploding gradients |
| Loss function | Smooth L1 (Huber) | More robust to outliers than MSE |

### Epsilon-Greedy Exploration

The agent follows an ε-greedy policy during training, where ε decays linearly from 1.0 to 0.05 over the first 40,000 environment decisions (≈133 episodes). Each environment decision corresponds to one action per intersection (16 actions per decision step). After the decay period, ε remains at 0.05, providing a small amount of persistent exploration.

### Training Schedule

- **Total episodes**: 3,063 (budget-derived, see §8)
- **Episode length**: 1,500 simulation seconds (25 minutes of simulated time)
- **Demand period**: Vehicles are inserted for the first 1,200 seconds of each episode; the last 300 seconds allow residual traffic to clear.
- **Randomised demand**: Each training episode draws a random configuration from a seeded generator:
  - Demand scale: uniform(0.7, 1.5) × base rate 0.45 veh/s
  - Directional bias: 30% of episodes have uneven directional demand (e.g., 2–3× traffic on one axis)
  - Surge windows: 25% of episodes have a 2–3× demand surge lasting 150–350 s
  - Arrival-rate waves: sinusoidal modulation with random amplitude (0–0.4) and period (400–900 s)
  - All routes: random origin-destination pairs drawn from fringe edges

### Checkpointing

- **`model_best.pt`**: Saved whenever `episode_reward > best_reward` (best seen: −1.653)
- **`model_latest.pt`**: Saved every 10 episodes and at episode end; used for resume

### Training Log

Structured JSONL logs are written to `logs/train.jsonl` with one event per line, including:
- `train_start`: configuration and parameter count
- `episode`: per-episode reward, loss, epsilon, decisions, buffer size, scenario seed, demand scale, elapsed wall time, resource usage (RSS, CPU%)
- `train_end`: final best reward, wall-clock time, peak RSS, average CPU

---

## 7. Compute Optimisations

All training was conducted on a CPU-only node with 4 cores (budgeted as 3 vCPUs). The following optimisations were applied to maximise throughput:

| Optimisation | Detail | Impact |
|---|---|---|
| **Torch thread cap** | `torch.set_num_threads(2)` | Prevents PyTorch from oversubscribing the 4-core node |
| **Single sequential environment** | 1 env, not parallel | Libsumo is in-process; 1 env = 1 SUMO process, no IPC overhead |
| **No rendering** | SUMO headless (`--no-step-log`) | Avoids costly GUI/text rendering |
| **Libsumo in-process** | Python API via shared library | No TCP socket overhead (unused TraCI) |
| **Batch inference** | 16 observations in a single forward pass | 16× throughput vs. per-intersection inference |
| **numpy arrays, not Python lists** | Replay buffer stores np.float32 arrays | Fast batch sampling, minimal memory overhead |
| **JSONL logging** | Append-only, line-buffered | No serialisation overhead at episode boundaries |
| **ResourceMonitor via psutil** | Sampled every 60 decisions during eval | Negligible overhead (~0.1 ms per sample) |

### Wall-clock budget derivation

A 50,000-step benchmark (script `scripts/benchmark.py`) measured **574.3 simulation steps/second** on this node under typical training conditions. With a 3-hour target budget:

```
3 hours × 3600 s/h × 574.3 steps/s ≈ 6,200,000 simulation steps
```

Each training episode runs 1,500 simulation steps (300 decisions × 5 s). However, the agent's decision loop and learning add overhead. The effective budget of 3,063 episodes (≈4,594,500 sim steps) was derived by:

```
Episode budget = 3063 episodes
Wall clock: 3063 × ~2.3 s/episode ≈ 7,045 s ≈ 1.96 h (within budget)
```

The actual training completed in **6,117 s (~1.70 h)**, well within the 3-hour budget. Peak memory was **425.1 MB** and average CPU utilisation was **167.2%** (≈1.7 cores saturated).

---

## 8. Evaluation Methodology

### Checkpoint

A single checkpoint (`checkpoints/model_best.pt`, episode 1129, reward −1.653) was used for all evaluation episodes. No per-scenario retraining or fine-tuning was performed.

### Evaluation Scenarios

Eight scenarios were defined, each with 20 deterministic seeds (episode length 1,800 s, 300 s demand-end margin):

| Scenario | Demand Scale | Perturbation | Description |
|---|---|---|---|
| **normal** | 1.0× | None | Baseline traffic conditions |
| **high_demand** | 1.8× | None | 80% more vehicles |
| **sudden_surge** | 1.0× | 3× demand surge at 600–900 s | Unexpected spike |
| **uneven_directional** | 1.0× | 3× bias on E/W axis, 0.6× on N/S | Asymmetric origin-destination |
| **road_closure** | 1.0× | Edges B1C1 and C1B1 closed at 600 s | Disruption from roadworks |
| **noisy_sensors** | 1.0× | Gaussian noise σ=0.15 added to queue/occupancy | Sensor noise |
| **missing_sensors** | 1.0× | 25% of lane sensors return zero (dropout) | Sensor failure |
| **partial_tls_failure** | 1.0× | 4 of 16 intersections revert to fixed-time at 300 s | Traffic-light malfunction |

### Perturbation Injectors

All perturbations are applied only during evaluation (not during training):

- **Sensor noise**: Added to queue and occupancy observations each step as `N(0, σ²)` noise, clipped to valid ranges.
- **Sensor dropout**: A random binary mask (25% zero) is generated from the seed and applied to all lane observations throughout the episode.
- **TLS failure**: 4 specific intersections stop responding to agent actions and switch to SUMO's built-in fixed-time program at t=300 s.
- **Road closure**: Two edges are set to disallow passenger vehicles at t=600 s. Vehicles are allowed to reroute (`rerouting.probability=0.5`).

### Inference Protocol

- **Greedy policy**: ε=0 (no exploration noise during evaluation)
- **Deterministic seeds**: Seed = `10000 + scenario_index × 100 + k` for k=0..19
- **Multiprocessing**: 2 worker processes (fresh libsumo instance per worker), `maxtasksperchild=8`
- **Resource monitoring**: CPU and memory sampled every 60 decisions via `psutil`

### Failure Handling

All 160 episodes completed successfully. The evaluation framework records every episode with a status field:

- `completed`: Episode reached full length with valid metrics
- `failed`: Episode terminated early due to an exception (error message recorded)
- `incomplete`: Simulation ended before the full episode length (partial metrics recorded)
- `invalid_metrics`: Metrics failed validation (non-finite or negative values)

No episodes were silently dropped or excluded from the output CSV.

---

## 9. Results

### Overall (n = 160 episodes)

| Metric | Mean | Std |
|---|---|---|
| Avg waiting time | 47.68 s | 38.03 s |
| Avg travel time | 147.61 s | 23.32 s |
| Avg queue length | 0.15 veh | 0.11 veh |
| P95 queue length | 1.12 veh | 0.54 veh |
| Completed trips | 767.38 | 198.03 |
| Throughput | 1,534.8 veh/h | 396.1 veh/h |
| Gridlock duration | 0.0 s | 0.0 s |
| Gridlock events | 0.0 | 0.0 |
| Inference latency | 0.38 ms | 0.08 ms |
| CPU utilisation | 131.9% | 5.9% |
| Peak memory | 388.1 MB | 3.5 MB |

### Per-Scenario Breakdown

| Scenario | Avg Wait (s) | Avg Travel (s) | Queue (veh) | Throughput (veh/h) | Observations |
|---|---|---|---|---|---|
| **normal** | 20.3 ± 0.9 | 125.4 ± 1.8 | 0.06 ± 0.00 | 1,366.9 ± 44.9 | Baseline performance |
| **high_demand** | 31.1 ± 1.3 | 141.1 ± 2.4 | 0.16 ± 0.01 | 2,442.3 ± 72.7 | 1.8× vehicles; wait only 53% higher |
| **sudden_surge** | 32.6 ± 3.4 | 142.4 ± 4.0 | 0.13 ± 0.02 | 1,892.2 ± 60.5 | Surge absorbed without gridlock |
| **uneven_directional** | 19.3 ± 0.9 | 124.0 ± 1.6 | 0.06 ± 0.00 | 1,346.5 ± 36.0 | Asymmetric demand handled well |
| **road_closure** | 25.0 ± 2.4 | 130.5 ± 3.2 | 0.07 ± 0.01 | 1,349.7 ± 44.0 | Closure caused only +23% wait |
| **noisy_sensors** | 68.8 ± 3.2 | 178.9 ± 4.3 | 0.20 ± 0.01 | 1,339.2 ± 52.4 | Sensor noise degrades performance |
| **missing_sensors** | 86.3 ± 25.7 | 176.8 ± 18.4 | 0.25 ± 0.08 | 1,283.3 ± 51.1 | 25% dropout causes significant degradation |
| **partial_tls_failure** | 98.2 ± 61.7 | 161.8 ± 22.7 | 0.29 ± 0.18 | 1,258.0 ± 89.7 | 4 fixed-time intersections cause longest waits |

### Key Observations

1. **No gridlock occurred in any scenario**: The policy consistently avoided complete deadlock, even under high demand and TLS failure.
2. **Sensor degradation is the most challenging perturbation**: Missing sensors (25% dropout) raised average waiting time 4.3× vs. normal, suggesting the policy relies heavily on accurate lane observations.
3. **High demand is handled efficiently**: Throughput nearly doubled (1,367 → 2,442 veh/h) with only a 53% increase in waiting time, demonstrating the policy's ability to move more vehicles without proportional degradation.
4. **Uneven directional demand is the easiest scenario**: This is likely because the asymmetric flow naturally creates unidirectional pressure gradients that the pressure-based reward can exploit.
5. **Inference latency is extremely low**: 0.38 ms per batch of 16 intersections, confirming the MLP is well within real-time constraints for a 5-second decision interval.

---

## 10. Failure Cases and Limitations

### Known Failure Cases

1. **Partial TLS failure with high variance**: The standard deviation of waiting time under `partial_tls_failure` is 61.7 s (mean 98.2 s), indicating that the outcome depends heavily on which specific intersections fail and the traffic pattern at the time of failure.
2. **Sensor dropout causes large variance in outcome**: Under `missing_sensors`, waiting time varies from ~60 s to ~110 s across seeds (std 25.7 s), suggesting the policy's robustness to missing data is inconsistent.
3. **No convergence to positive reward**: The reward function is always negative, so the agent converges toward the best achievable negative value (−1.653) rather than a positive return. This is expected by design but makes absolute reward values difficult to interpret outside the training context.

### Limitations

1. **Single simulator**: Training uses one sequential environment (libsumo limitation). Parallelism via multiple SUMO processes would accelerate training but was not feasible within the process budget.
2. **No explicit coordination mechanism**: The parameter-shared policy learns implicit coordination through shared weights, but there is no communication channel or centralised value function. Explicit coordination (e.g., graph neural networks, neighbour-state concatenation) could improve performance on perturbation scenarios.
3. **Fixed decision interval**: The 5-second decision interval is a heuristic. An adaptive or event-triggered decision scheme could respond more rapidly to changing conditions.
4. **No pedestrian, cyclist, or public-transit modelling**: The simulation is passenger-vehicle-only. Mixed traffic would require additional observation features and a more complex reward structure.
5. **Grid topology only**: The 4×4 grid is a simplified urban layout. Real networks have irregular geometries, multi-lane arterials, and varying intersection spacing that would require a more expressive network.
6. **No actuated control baseline**: The policy is compared only against its own untrained behaviour. A systematic comparison against fixed-time or actuated control would contextualise the absolute performance.
7. **Gridlock metric never triggered**: The gridlock detector (speed < 0.1 m/s for ≥60 s with ≥20 vehicles) never fired in any evaluation episode. While this is positive operationally, it means the gridlock-related components of the reward and metrics were never exercised.

---

## 11. Resource Usage

| Resource | Training | Evaluation (160 episodes) | Total |
|---|---|---|---|
| Wall clock time | 6,117 s (1.70 h) | 239 s | ~6,356 s |
| CPU cores | 2 (average) | 2 (2 workers) | — |
| CPU utilisation | 167.2% | 131.9% | — |
| Peak memory | 425.1 MB | 388.1 MB | — |
| Disk (checkpoints) | 1.3 MB | — | — |
| Disk (logs + metrics) | 2.1 MB | 0.5 MB | — |
| Disk (network file) | 0.3 MB | — | — |

---

## 12. Reproducibility

- **Deterministic seeds**: Master seed 42 for training; seed_base 10,000 for evaluation.
- **Config-driven**: All hyperparameters in `config.yaml`; all paths relative to project root.
- **Checkpoint-compatible**: Training can be resumed from `model_latest.pt` via `train.py --resume`.
- **Deterministic visualisation**: Same inputs produce byte-identical PNG (verified via SHA-256).
- **Randomised demand**: All training episodes use seeded RNGs; the same seed always produces the same trip file.