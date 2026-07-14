# Adaptive Traffic-Signal Control via Parameter-Shared Deep RL (CPU-only)

## Goal
A complete, reproducible ML system that trains a traffic-signal controller for a fixed 4×4 signalized grid in a microscopic traffic simulator, evaluates it across 8 scenarios × ≥20 deterministic seeds without retraining, records all required metrics, and renders exactly one deterministic 1600×1600 PNG summarizing system behaviour.

## Research Summary
- **Simulator**: SUMO is pip-installable on Linux via `eclipse-sumo` (simulator binaries + tools) and `libsumo` (in-process Python API, Python ≥3.9) — verified on PyPI (actively released). `libsumo` avoids TraCI TCP overhead, giving much faster headless stepping — important for the 50k-step benchmark and CPU-only training. CityFlow was considered but requires building C++ from source (cmake + pybind11), a higher-risk path on this node; rejected on reliability grounds.
- **Network generation**: SUMO's bundled `netgenerate` creates a signalized N×N grid directly (`--grid --grid.number=4 --grid.length=... --grid.attach-length=...` with fringe entry/exit stubs), giving 16 controlled intersections, bidirectional roads, and multiple entry/exit points with identical geometry across all runs.
- **Algorithm**: Parameter-shared DQN with a pressure-style reward (in the spirit of PressLight/MPLight) is an established, CPU-friendly approach to multi-intersection signal control: one small shared network controls all 16 intersections from local observations, so model size and inference cost stay tiny and experience from all intersections trains a single policy.

## Approach
**Multi-agent RL with full parameter sharing (Double DQN, custom PyTorch-CPU implementation).**

- **Environment**: SUMO headless via `libsumo`. Fixed 4×4 grid net file generated once by `netgenerate` and committed to the repo (identical geometry everywhere). Traffic demand generated programmatically per episode from a seeded scenario spec (route files written per episode).
- **Observation (per intersection, local)**: per-incoming-lane halted-vehicle count (queue) and occupancy, normalized; current phase one-hot; normalized time-in-phase. Fixed-size vector (grid intersections are homogeneous 4-way).
- **Action (per intersection)**: choose the next green phase from the 4 standard phases (NS-through, NS-left, EW-through, EW-left) every Δ=5 s decision step; enforced min-green and automatic 3 s yellow insertion on phase change (safety/realism guard).
- **Reward (per intersection)**: negative intersection pressure (sum of incoming minus outgoing lane queues), plus a small penalty on waiting-vehicle count — encourages congestion/wait reduction and throughput without global signals.
- **Model**: shared MLP ≈ (obs_dim → 128 → 128 → 4), <50k parameters. Double DQN, uniform replay buffer, ε-greedy decay, target network. `torch.set_num_threads(2)`; ≤2 SUMO envs concurrently (train with 1–2 sequential/parallel envs only).
- **Training scenarios**: each episode samples a seeded generator config — demand scale (normal / moderate surge windows), directional bias (uneven N-S vs E-W splits), variable Poisson arrival rates, randomized origin-destination routes crossing multiple intersections. No fixed pattern memorization.
- **Evaluation** (greedy policy, single trained checkpoint, no per-scenario retraining): 8 scenarios × 20 deterministic seeds:
  1. normal, 2. high demand, 3. sudden surge, 4. uneven directional,
  5. road closure (close edges mid-network via API), 6. noisy sensors (seeded Gaussian noise on obs), 7. missing sensors (seeded lane-observation dropout → zeros), 8. partial light failure (subset of intersections stuck on fixed-time fallback / frozen program).
- **Metrics per (scenario, seed)**: avg waiting time, avg travel time, avg queue length, p95 queue length, completed trips, throughput (veh/h), gridlock duration & event count (gridlock = network-wide sustained stall detection), policy inference latency (ms/decision), CPU %, peak RSS memory. Failed/incomplete episodes recorded with status flag — never silently dropped.
- **Visualization**: single deterministic matplotlib PNG, 1600×1600 px (figsize 16×16 in @ dpi=100), dark neutral background, top-down fixed geometry read from the net file, centred; road colour = avg congestion (fixed 0–100% scale), road width = avg queue (fixed 0–40 veh scale), intersection marker size = avg wait (fixed 0–180 s scale), directional arrows = dominant flow, bottleneck/gridlock zone highlighting (0–100% scale); text block with algorithm name + headline metrics. Values clipped visually only; raw values untouched in CSV/JSON. No other charts, no rescaling, no branding.
- **Reliability**: global seed control (Python/NumPy/torch/SUMO), YAML config, periodic checkpointing + resume, input/metric validation, per-episode failure recovery (exception → recorded failed episode, run continues), psutil-based CPU/memory monitoring, structured JSONL logs, pytest suite (metric aggregation, image dimensions = 1600×1600, visual normalization/clipping, invalid-episode handling).

## Subtasks
1. Environment setup: create venv; install `eclipse-sumo`, `libsumo`, `torch` (CPU wheel), `numpy`, `pandas`, `matplotlib`, `psutil`, `pyyaml`, `pytest`; verify `import libsumo` works and `netgenerate` binary is callable. Expected output: working venv + `requirements.txt` (verify: a 10-step toy SUMO simulation runs headless via libsumo).
2. Fixed 4×4 grid network: generate `network/grid4x4.net.xml` with netgenerate (16 signalized intersections, bidirectional 2-lane roads, fringe entry/exit stubs); write seeded scenario/route generator (`src/scenarios.py`) producing route files for normal / uneven / surge / variable-rate / randomized-route demand. Expected output: net file + generator (verify: SUMO loads net + a generated route file and vehicles complete trips crossing multiple intersections).
3. Gym-style multi-intersection env wrapper (`src/env.py`): libsumo lifecycle, per-intersection observations, phase-action application with min-green + yellow insertion, pressure reward, episode metrics accumulation, input validation. Expected output: env module (verify: random-policy episode runs end-to-end, obs shapes/ranges validated, metrics non-trivial and varying).
4. Shared Double-DQN agent + training loop (`src/agent.py`, `train.py`): replay buffer, target net, ε-decay, checkpoint save/resume, structured JSONL logging, psutil monitoring, YAML config (`config.yaml`), torch threads capped at 2, ≤2 concurrent envs. Expected output: trainable pipeline (verify: short smoke-train run of a few episodes shows loss computed, checkpoint written, resume works).
5. Benchmark 50,000 simulation steps on this node (`scripts/benchmark.py`); record steps/sec and set the concrete training budget (episodes × steps) in `config.yaml` from measured speed, targeting a practical wall-clock (~2–4 h max). Expected output: `results/benchmark.json` + budget written to config (verify: file contains measured steps/sec > 0 and derived budget).
6. Minimal end-to-end pass: tiny training run → checkpoint → 1-seed evaluation → draft PNG, proving the full pipeline before full training. Expected output: draft artifacts (verify: PNG exists at 1600×1600, metrics CSV row present, no crashes).
7. Full training run with the benchmarked budget over mixed randomized scenarios; monitor reward trend; save best + final checkpoints. Expected output: `checkpoints/model_best.pt` (verify: training curves in logs show clear improvement of episode return / reduction in waiting time vs. initial random policy).
8. Evaluation harness (`evaluate.py`): 8 scenarios × 20 deterministic seeds, greedy policy from the single trained checkpoint, per-episode perturbation injectors (closure / noise / dropout / light failure), full metric capture incl. latency, CPU, peak memory, failure statuses. Expected output: `results/metrics.csv` (160 rows incl. any failed episodes) + `results/aggregate.json` (verify: row count = 160, all required metric columns present, failed episodes flagged not dropped, values vary across scenarios).
9. Visualization script (`visualize.py`): render the single final 1600×1600 PNG per fixed spec from `results/` only; deterministic (re-run → byte-identical or pixel-identical image). Expected output: `results/final_visualization.png` (verify: PIL check exactly 1600×1600, re-render produces identical image hash, fixed scales applied with clipping).
10. Test suite (`tests/`): metric aggregation correctness, invalid/incomplete-episode handling, final-image dimensions, visual normalization/clipping bounds. Expected output: pytest suite (verify: all tests pass in the venv).
11. Technical report (`REPORT.md`) covering algorithm choice + justification, observation/action/reward design, architecture, training procedure, compute optimizations, evaluation methodology, failure cases, resource usage, limitations — no comparisons/rankings against other systems; plus `README.md` setup instructions. Expected output: report + readme (verify: all required report sections present, numbers consistent with `results/aggregate.json`).

## Deliverables
| File Path | Description |
|-----------|-------------|
| /home/azureuser/VisualComp/Fable/requirements.txt | Pinned dependency file |
| /home/azureuser/VisualComp/Fable/README.md | Setup + usage instructions |
| /home/azureuser/VisualComp/Fable/config.yaml | Reproducible configuration (seeds, budget, scales) |
| /home/azureuser/VisualComp/Fable/network/grid4x4.net.xml | Fixed 4×4 SUMO network |
| /home/azureuser/VisualComp/Fable/src/ | Env, agent, scenarios, metrics, utils |
| /home/azureuser/VisualComp/Fable/train.py | Training script (checkpoint + resume) |
| /home/azureuser/VisualComp/Fable/evaluate.py | Evaluation script (8 scenarios × 20 seeds) |
| /home/azureuser/VisualComp/Fable/visualize.py | Deterministic final-PNG renderer |
| /home/azureuser/VisualComp/Fable/scripts/benchmark.py | 50k-step simulation benchmark |
| /home/azureuser/VisualComp/Fable/checkpoints/model_best.pt | Trained model checkpoint |
| /home/azureuser/VisualComp/Fable/results/metrics.csv | Per-seed, per-scenario metrics (incl. failures) |
| /home/azureuser/VisualComp/Fable/results/aggregate.json | Aggregated metrics |
| /home/azureuser/VisualComp/Fable/results/final_visualization.png | The single 1600×1600 final image |
| /home/azureuser/VisualComp/Fable/logs/ | Structured JSONL execution logs |
| /home/azureuser/VisualComp/Fable/tests/ | Pytest suite |
| /home/azureuser/VisualComp/Fable/REPORT.md | Technical report |

## Evaluation Criteria
- Single checkpoint evaluated on all 8 scenarios × 20 seeds (160 episodes) with zero per-scenario retraining.
- `metrics.csv` has 160 rows, every required metric column, and explicit status for failed episodes.
- Trained policy shows clear improvement over its own untrained/random starting behaviour on avg waiting time and queue length (no external comparisons).
- Final PNG is exactly 1600×1600, deterministic across re-renders, and satisfies every element of the fixed visual spec.
- All tests pass; training respects ≤2 concurrent envs, ≤2 torch threads, no live rendering, bounded memory.

## Notes
- Hardware: 4 cores visible but treat as 3-vCPU budget; no GPU — torch CPU wheel only.
- SUMO runs strictly headless (`sumo` binary via libsumo, never `sumo-gui`).
- Gridlock definition must be explicit in code + report (e.g., sustained interval where network mean speed < threshold and no trip completions).
- Perturbations (noise/dropout/light failure) are applied at evaluation time only, seeded deterministically.
- Training budget is decided from the measured 50k-step benchmark, not assumed.
