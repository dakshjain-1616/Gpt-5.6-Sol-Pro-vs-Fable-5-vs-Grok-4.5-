# Adaptive Traffic-Signal Control ML System

## Goal
Design, train, evaluate, and visualize a complete CPU-only ML traffic-signal controller on a fixed 4×4 road network (16 intersections), producing all required metrics artifacts and one deterministic 1600×1600 PNG summary image.

## Research Summary
- **CityFlow** is a microscopic multi-agent traffic simulator designed for RL traffic-signal control; faster than SUMO; Python API (`Engine`, `set_tl_phase`, lane vehicle counts, waiting times). Build requires CMake + C++ + Boost + pybind11; can be fragile on some hosts.
- **PressLight / CoLight** style multi-agent RL is standard for network-level TSC: local observations, pressure-based rewards, parameter sharing.
- **Independent DQN with shared weights** is the best fit for 3-vCPU / no-GPU: small MLP Q-network, experience replay, target network, low memory, no continuous-control overhead.
- **SUMO + sumo-rl** is a viable alternative but heavier (TraCI latency, GUI deps). Prefer CityFlow; ship a pure-Python microscopic fallback with identical Gym-like API if CityFlow cannot be built, so the ML stack remains unchanged.
- Hardware: 4 cores, ~15.6 GB RAM, no GPU, torch 2.12 + numpy available; matplotlib/pandas/psutil need install.

## Approach
**Algorithm: Shared Multi-Agent Independent DQN (I-DQN) with max-pressure-inspired reward**

Why this approach:
- Proven for multi-intersection TSC (PressLight lineage)
- Parameter sharing across 16 agents → tiny model, fast CPU inference
- Discrete phase actions match traffic-light control naturally
- Replay buffer + target net trains stably without GPU
- Robust to partial observability and sensor noise via local features + dropout-like noise injection in training

**Rejected alternatives:**
- Full MARL with communication graphs (CoLight GAT): higher compute, risk of OOM on 3 vCPU
- PPO/A2C continuous: slower sample efficiency for discrete phases on CPU
- Evolutionary methods: too many simulation rollouts for budget
- Centralized single agent over all lights: huge action space (4^16)

**Simulator strategy:**
1. Attempt CityFlow build/install
2. If install fails, use built-in pure-Python microscopic simulator (`src/sim/python_micro_sim.py`) with car-following, queues, bidirectional 4×4 grid, multi-route flows — same observation/action interface

**Network:** Fixed 4×4 grid, 16 signalized intersections, bidirectional edges, multiple OD pairs, configurable demand via flow generators.

**Observation (per intersection, local):**
- Incoming lane queue lengths (normalized)
- Incoming lane waiting times / vehicle counts
- Current phase one-hot + elapsed phase time
- Neighbor pressure summary (optional fixed-size vector)
- Dim ≈ 20–32 floats

**Action:** Discrete phase selection among legal non-conflicting phases (typically 4 phases: NS green, EW green, NS left, EW left — simplified to 2–4 phases for CPU speed). Min green / yellow transition enforced in env wrapper.

**Reward (per agent, pressure-style):**
`r = - (sum incoming pressure) + β * throughput_delta - γ * gridlock_penalty`
where pressure ≈ queue_in − queue_out on competing movements. Global episode metrics still logged separately.

**Model:** Shared MLP Q-network: Linear(obs→64)→ReLU→Linear(64→64)→ReLU→Linear(64→n_actions). Target network soft/hard update. ε-greedy exploration decaying.

**Training mixture:** Curriculum over normal / uneven / surge / variable arrival / randomized routes. Max 2 concurrent envs. Checkpoint + resume.

**Budget:** Benchmark 50k sim steps first; set training steps from measured steps/sec (target complete run within practical wall time on 3 vCPU).

**Evaluation:** 8 scenarios × ≥20 deterministic seeds, single frozen checkpoint, no per-scenario fine-tune. Log all required metrics including failed episodes.

**Visualization:** One deterministic 1600×1600 PNG from aggregated eval metrics: road color=congestion, thickness=queue, node size=wait, arrows=flow, labels for algorithm + key metrics.

## Subtasks
1. **Project scaffold & dependencies** — `requirements.txt`, `configs/default.yaml`, package layout under `src/`, README setup instructions; install torch-cpu-friendly deps (numpy, matplotlib, pandas, pyyaml, psutil, pytest).
2. **Simulator backend** — CityFlow install attempt + 4×4 `roadnet.json` / flow generators; pure-Python microscopic fallback with identical `TrafficEnv` interface; deterministic seeds; no live rendering.
3. **Environment wrapper** — observations, legal actions, phase timing, reward, scenario modes (normal, high, surge, uneven, road_closure, noisy_sensors, missing_sensors, partial_light_failure), metric collectors, resource monitors (CPU/mem).
4. **I-DQN agent** — shared Q-network, replay buffer, target updates, checkpoint save/load/resume, inference latency measurement.
5. **Minimal E2E smoke** — 1 short episode: sim → obs → action → train step → checkpoint → eval stub → PNG stub; fix blockers.
6. **Benchmark 50k steps** — measure steps/sec on node; write `artifacts/benchmark_50k.json`; choose training budget (episodes/steps) written into config.
7. **Full training** — mixed scenarios, logging, checkpoints to `checkpoints/`, structured logs under `logs/`.
8. **Full evaluation** — ≥20 seeds × 8 scenarios; CSV per-seed metrics; JSON aggregates; record incomplete/failed episodes.
9. **Visualization** — `scripts/visualize.py` → single deterministic `artifacts/final_traffic_map.png` (1600×1600) meeting fixed visual spec.
10. **Tests & report** — tests for metric aggregation, image dimensions, visual normalization; `REPORT.md` covering algorithm, spaces, reward, architecture, training, compute opts, eval, failures, resources, limitations.
11. **Final packaging** — verify all deliverables exist and paths are documented in README.

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/home/azureuser/VisualComp/Grok/requirements.txt` | Dependencies |
| `/home/azureuser/VisualComp/Grok/README.md` | Setup & run instructions |
| `/home/azureuser/VisualComp/Grok/configs/default.yaml` | Reproducible config |
| `/home/azureuser/VisualComp/Grok/src/**` | Full source (sim, env, agent, metrics, viz) |
| `/home/azureuser/VisualComp/Grok/scripts/train.py` | Training entrypoint |
| `/home/azureuser/VisualComp/Grok/scripts/evaluate.py` | Evaluation entrypoint |
| `/home/azureuser/VisualComp/Grok/scripts/visualize.py` | Final PNG generator |
| `/home/azureuser/VisualComp/Grok/scripts/benchmark.py` | 50k-step benchmark |
| `/home/azureuser/VisualComp/Grok/checkpoints/idqn_shared.pt` | Trained model |
| `/home/azureuser/VisualComp/Grok/artifacts/final_traffic_map.png` | 1600×1600 summary PNG |
| `/home/azureuser/VisualComp/Grok/artifacts/metrics_per_seed.csv` | Per-seed/scenario metrics |
| `/home/azureuser/VisualComp/Grok/artifacts/metrics_aggregate.json` | Aggregated metrics |
| `/home/azureuser/VisualComp/Grok/artifacts/benchmark_50k.json` | Benchmark results |
| `/home/azureuser/VisualComp/Grok/logs/` | Structured execution logs |
| `/home/azureuser/VisualComp/Grok/tests/` | Aggregation / image / normalization tests |
| `/home/azureuser/VisualComp/Grok/REPORT.md` | Technical report |

## Evaluation Criteria
- Trained checkpoint loads and runs inference on all 16 intersections
- Evaluation covers all 8 scenarios with ≥20 seeds each
- CSV + JSON metrics include all required fields; failed episodes recorded
- `final_traffic_map.png` is exactly 1600×1600, deterministic, with required visual encodings and labels
- No LLM/VLM/hosted API in the solution
- Training/eval stay within CPU constraints (≤2 concurrent envs, no live render)
- Tests pass for metric aggregation, image size, visual normalization

## Notes
- Seeds fixed in config for reproducibility
- Clip visual scales per spec; keep raw metric values unclipped in files
- Prefer complete reliable system over large model
- If CityFlow builds: use it; else pure-Python micro sim is acceptable open-source component of this repo
- Do not compare against other algorithms in the report
