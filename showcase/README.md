# Showcase — three implementations of the traffic-signal brief

Everything here is derived from the three runs' **own** artifacts. No metric is
invented, and no number is copied between runs.

## What each run actually produced

| | Fable | Grok | GPT |
|---|---|---|---|
| Status | complete | complete | **incomplete** |
| Algorithm | Parameter-shared Double DQN | Shared multi-agent I-DQN (max-pressure reward) | never implemented |
| Simulator | SUMO 1.27 via `libsumo` | custom pure-Python micro-sim (CityFlow install failed) | SUMO, smoke test only |
| Network | 4×4, 16 signals | 4×4, 16 signals | 1 intersection (tiny test net) |
| Episode | 1800 s, 5 s decisions | 360 steps, 1 s decisions | 26 steps |
| Model | 19,844 params (MLP 128×128), 4 phases | 5,890 params (MLP 64), 2 phases | no checkpoint |
| Episodes evaluated | 160 / 160, 0 failed | 160 / 160, 0 failed | 0 |
| Final 1600×1600 PNG | yes | yes | no |

GPT stopped after verifying it could drive SUMO on a one-intersection network.
There is no model, no evaluation, and nothing to visualize for it.

## The comparability problem (read this before reading any chart)

Fable and Grok ran on **different simulators**. SUMO's car-following is far
stricter than Grok's hand-rolled micro-sim, and their units differ
(throughput is veh/h vs veh/step; episodes are 1800 s vs 360 steps). Their raw
waiting times and throughputs are therefore **not on a common axis**, and a
head-to-head "who won" ranking would be meaningless.

Two things *are* legitimately comparable and are the only things compared here:

1. **Self-relative degradation** — how much each policy degrades under each
   stress scenario relative to its *own* normal-traffic baseline. The simulator
   cancels out of the ratio.
2. **Resource cost** — inference latency and peak memory, measured on the same
   3-vCPU node.

Both models independently agree on which scenarios are hard: partial signal
failure (4.8× / 4.5× the normal waiting time) and missing sensors (4.3× / 3.1×)
dominate; road closure and uneven demand are nearly free (≈1.0–1.2×).

## Artifacts

| File | What it is |
|---|---|
| `out/comparison.png` | 1600×2000 comparison infographic across all three runs |
| `out/fable_rollout.{mp4,mov,webm,gif}` | 72 s video: Fable's **trained policy** driving all 8 eval scenarios |
| `out/grok_rollout.{mp4,mov,webm,gif}` | 72 s video: Grok's **trained policy**, same 8 scenarios |
| `out/gpt_rollout.{mp4,mov,webm,gif}` | 72 s video: **NOT a trained policy** — see below |
| `traces/*.npz` | Recorded rollout state (vehicle positions, phases, queues) |

### The GPT video is a baseline, not a result

GPT has **no model to record**. It contains no `torch` import, no Q-network, no
replay buffer, no training loop and no checkpoint; its README instructs you to
run `run_pipeline.py train / evaluate / render` and read `REPORT.md`, none of
which exist. There is no policy whose behaviour could be shown.

What its `network.py` *does* do — verifiably — is build the full 4×4 grid
(16 signals) and generate valid multi-hop routes. So `gpt_rollout.*` shows
**GPT's own network running under SUMO's default static signal program**:
ordinary fixed-time lights that sense nothing and learn nothing. The video is
labelled `NO LEARNED CONTROLLER` in red on every frame.

It is included for two honest reasons: it shows how far that run actually got,
and it is a useful un-learned reference point. Two things are visible in it that
the metrics alone don't convey — every signal cycles in **lockstep** (the static
program has no offsets or coordination), and under partial signal failure the
fixed timer lets vehicle count climb from ~32 to ~87 as queues lock up behind
the stuck signals.

It must never be presented as GPT's trained result. There isn't one.

Four video formats are provided because container/codec support varies by
viewer; they are the same 1920×1080 content. The GIF is downscaled to 960 px
and 12 fps so it plays anywhere.

## How the videos were made

`record_fable.py` / `record_grok.py` reload each **frozen checkpoint** and replay
it greedily (no training, no fine-tuning) on the exact seeds that appear in that
run's evaluation CSV, capturing per-step vehicle positions, signal phases and
per-road queues. `render_video.py` turns a trace into video.

The replay is deterministic: Grok's `normal` seed-1000 replay completes 311
trips, matching the 311 recorded in `Grok/artifacts/metrics_per_seed.csv`.

Reading the video:

- **road colour** — live queue on that road
- **vehicle dot** — red = stopped, pale = moving at free speed
- **node teal / amber** — which axis currently has green (N/S vs E/W)
- **red ring** — that signal has failed (partial-failure scenario)
- **left panel** — live state, plus the metrics that scenario actually scored
  across all 20 evaluation seeds

Congestion colour is auto-scaled to each model's own data. The brief's mandated
fixed scale (0–40 vehicles) is ~100× larger than the queues these policies
actually produce (~0.2–0.5 veh/road), which is why both runs' spec-compliant
final PNGs look almost empty. Those original PNGs are left untouched; the
auto-scaling applies only to the artifacts in this folder.

## Reproducing

```bash
# record rollouts (each run needs its own venv)
Fable/venv/bin/python showcase/record_fable.py
Grok/venv/bin/python  showcase/record_grok.py

# render
Fable/venv/bin/python showcase/render_video.py --model fable
Grok/venv/bin/python  showcase/render_video.py --model grok
Fable/venv/bin/python showcase/infographic.py
```
