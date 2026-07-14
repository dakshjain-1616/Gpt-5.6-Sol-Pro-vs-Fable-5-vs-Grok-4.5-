# Adaptive Traffic-Signal Control (Shared I-DQN)

CPU-only multi-agent RL system for adaptive traffic lights on a fixed **4×4** grid (16 intersections).  
Algorithm: **Shared Multi-Agent Independent DQN** with max-pressure-inspired reward.  
Simulator: pure-Python microscopic engine (CityFlow was unavailable on this host).

## Setup

```bash
cd /home/azureuser/VisualComp/Grok
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

## Project layout

```
configs/default.yaml     # reproducible hyperparameters & paths
src/sim/                 # pure-Python 4x4 micro simulator
src/env/                 # TrafficEnv (obs/action/reward, 8 scenarios)
src/agent/               # Shared I-DQN
src/metrics/             # aggregation & validation
src/viz/                 # 1600x1600 map renderer
scripts/train.py         # training entrypoint
scripts/evaluate.py      # 8 scenarios × N seeds
scripts/visualize.py     # final PNG
scripts/benchmark.py     # 50k-step CPU benchmark + budget update
tests/                   # metric / image / normalization tests
checkpoints/idqn_shared.pt
artifacts/               # metrics CSV/JSON, PNG, benchmark
logs/                    # train/eval structured logs
REPORT.md                # technical report
```

## Quick start

```bash
source venv/bin/activate
export PYTHONPATH=.

# Optional smoke (seconds)
python scripts/train.py --smoke
python scripts/evaluate.py --smoke
python scripts/visualize.py

# Benchmark 50k steps and write training budget into config
python scripts/benchmark.py --steps 50000

# Full train (uses configs/default.yaml budget)
python scripts/train.py
# Resume:
# python scripts/train.py --resume checkpoints/idqn_shared.pt

# Full evaluation (8 scenarios × 20 seeds)
python scripts/evaluate.py --checkpoint checkpoints/idqn_shared.pt

# Deterministic 1600×1600 map
python scripts/visualize.py

# Tests
pytest -q tests/
```

## Deliverables

| Path | Description |
|------|-------------|
| `checkpoints/idqn_shared.pt` | Trained shared I-DQN weights |
| `artifacts/metrics_per_seed.csv` | Per scenario/seed metrics |
| `artifacts/metrics_aggregate.json` | Aggregated metrics |
| `artifacts/final_traffic_map.png` | 1600×1600 summary map |
| `artifacts/benchmark_50k.json` | CPU benchmark + recommended budget |
| `logs/` | Training/eval logs |
| `REPORT.md` | Algorithm, spaces, training, eval, limitations |

## Evaluation scenarios

normal, high_demand, sudden_surge, uneven, road_closure, noisy_sensors, missing_sensors, partial_light_failure — each with ≥20 deterministic seeds; single frozen checkpoint.

## Notes

- No GPU required; max 1–2 concurrent envs.
- No LLM/VLM/hosted API in the control loop.
- Visual scales are clipped for display only; CSV/JSON store raw values.
- See `REPORT.md` for full methodology.
