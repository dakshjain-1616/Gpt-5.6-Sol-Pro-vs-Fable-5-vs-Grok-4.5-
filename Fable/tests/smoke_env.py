"""Smoke test: random-policy episode through TrafficEnv (short)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.env import TrafficEnv, TLS_IDS, NUM_PHASES
from src.scenarios import ScenarioSpec
from src.utils import load_config, set_global_seeds

cfg = load_config()
set_global_seeds(1)
spec = ScenarioSpec(name="smoke", seed=123, episode_length=300, demand_end_margin=60,
                    base_rate=cfg["demand"]["base_rate"], demand_scale=1.2)
env = TrafficEnv(cfg)
obs = env.reset(spec)
print("obs shape:", obs.shape, "min/max:", obs.min(), obs.max())
assert obs.shape == (16, cfg["env"]["obs_dim"])

rng = np.random.default_rng(0)
tot_r = np.zeros(16)
done = False
steps = 0
phase_changes = 0
last_phases = dict(env.phase)
while not done:
    actions = rng.integers(0, NUM_PHASES, size=16)
    obs, r, done, info = env.step(actions)
    assert np.all(np.isfinite(r)), "non-finite reward"
    tot_r += r
    steps += 1
    for tid in TLS_IDS:
        if env.phase[tid] != last_phases[tid]:
            phase_changes += 1
    last_phases = dict(env.phase)

res = env.episode_result()
env.close()
m = res["metrics"]
print(f"decisions={steps} phase_changes={phase_changes}")
print(f"mean reward per intersection: {tot_r.mean():.3f}")
print("metrics:", {k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()})
assert steps > 0 and phase_changes > 0
assert m["total_departed"] > 0, "no vehicles departed"
assert m["completed_trips"] > 0, "no trips completed"
assert m["avg_queue_length"] >= 0
print("SMOKE ENV OK")
