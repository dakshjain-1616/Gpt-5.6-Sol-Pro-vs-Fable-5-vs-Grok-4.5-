# CPU Adaptive Traffic Signal Control

A reproducible CPU-only Eclipse SUMO microscopic simulation and parameter-shared DQN controller for a fixed 4×4 bidirectional network (16 controlled intersections).

## Setup

```bash
cd /home/azureuser/VisualComp/GPT
bash setup.sh
```

The setup creates `venv`, installs exact versions from `requirements.txt`, and runs a headless tiny-network verification. Never use `sumo-gui`; all workflows are headless.

## Workflow

```bash
source venv/bin/activate
python scripts/run_pipeline.py build
python scripts/run_pipeline.py minimal
python scripts/run_pipeline.py benchmark --steps 50000
python scripts/run_pipeline.py train
python scripts/run_pipeline.py evaluate
python scripts/run_pipeline.py render
python scripts/validate_artifacts.py
pytest -q
```

All random seeds, fixed geometry, training budget, observation/reward bounds, eight scenarios, and visual normalization ranges are in `config/config.yaml`. The simulator is externally controlled through TraCI. Training and evaluation use one simulator process at a time (the implementation never exceeds the two-environment limit). See `REPORT.md` after the completed run.
