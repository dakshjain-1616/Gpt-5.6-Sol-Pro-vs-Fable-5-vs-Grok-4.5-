# Three Agents, One Brief — Comparison Report

Fable, Grok and GPT were each given the same task, the same machine, and no help:

> Train a reinforcement-learning controller for a 4×4 grid of 16 traffic signals. Evaluate it across
> eight stress scenarios. Show that it holds up.

This report covers what each one actually built, what the numbers say, and — importantly — which
numbers you are not allowed to compare. Every figure here is read from the runs' own artifacts.
Nothing is copied between runs and nothing is invented.

**Contents**

1. [What happened](#1-what-happened)
2. [What each model built](#2-what-each-model-built)
3. [Design, side by side](#3-design-side-by-side)
4. [Results](#4-results)
5. [What the numbers actually tell us](#5-what-the-numbers-actually-tell-us)
6. [Caveats — the part that matters most](#6-caveats--the-part-that-matters-most)
7. [Appendix: scenario definitions](#7-appendix-scenario-definitions)

---

## 1. What happened

Two of the three finished. One did not.

| | **Fable** | **Grok** | **GPT** |
|---|---|---|---|
| **Outcome** | ✅ Complete | ✅ Complete | ❌ Agent never written |
| Algorithm | Parameter-shared Double DQN | Shared multi-agent I-DQN | — |
| Simulator | Eclipse SUMO 1.27 (`libsumo`) | Own pure-Python micro-sim | SUMO — smoke test only |
| Network | 4×4, 16 signals | 4×4, 16 signals | 4×4 buildable; only a 1-signal net was ever run |
| Model | 19,844 params | 5,890 params | none |
| Training | 3,063 episodes · 1.70 h | 339 episodes · 0.23 h | never started |
| Evaluation | 160/160 episodes, 0 failed | 160/160 episodes, 0 failed | 0 episodes |
| Report + viz | ✅ | ✅ | ❌ |

Fable and Grok both delivered the full pipeline: a trained policy, a 160-episode evaluation across
8 scenarios × 20 seeds, a technical report, and a final visualization. Neither had a single failed
episode.

GPT wrote 219 lines across three files, proved SUMO worked on a one-intersection toy network, and
stopped. **It has no model, no evaluation and no results, so it appears in no performance table in
this report.** Section 2.3 covers what it did and did not build, because "it didn't finish" is not
the same as "it did nothing," and the difference is worth being precise about.

---

## 2. What each model built

### 2.1 Fable — Double DQN on real SUMO

Fable took the most conservative, most standard route available and executed it carefully.

It ran **Eclipse SUMO 1.27 through `libsumo`** (the in-process API, ~3× faster than TraCI's socket
layer), on a proper 4×4 network generated with `netconvert`. Each intersection has **4 phases** —
NS-through, NS-left, EW-through, EW-left — with a 3-second yellow inserted on every phase change and
a 5-second minimum green. That is a realistic control problem: the agent cannot thrash the lights.

**Architecture.** One MLP, `21 → 128 → 128 → 4`, **19,844 parameters**, shared across all 16
intersections. Each intersection feeds its own observation through the same network and gets its own
action back. The observation is 21 numbers: queue and occupancy for each of 8 incoming lanes, a
4-way one-hot of the current phase, and time-in-phase.

Parameter sharing is the key decision. Sixteen independent networks would mean 16× the parameters
and 1/16th the data each; one shared network sees every intersection's experience. It trades away
the ability to specialize per-junction — reasonable on a symmetric grid.

**Reward** (`src/env.py:264`), per intersection:

```
pressure = halted_incoming − halted_outgoing
r = −(0.05 × pressure  +  0.02 × halted_incoming)
```

This is a **max-pressure** reward with a waiting penalty bolted on. "Pressure" is the imbalance
between vehicles piling up on the way in versus the way out; minimizing it is a well-established
traffic-control objective that provably stabilizes queues under mild assumptions. The second term
adds a direct penalty for cars sitting still, so the agent cannot game pressure by starving an
approach.

**Training.** Double DQN, γ=0.95, lr=5e-4, 100k replay buffer, batch 128, target sync every 500
gradient steps, ε annealed 1.0 → 0.05 over 40k decisions. It ran 3,063 episodes in 1.70 hours.

Its learning curve is the interesting part:

| Training quartile | Mean episode reward |
|---|---:|
| Episodes 1–765 | −7.85 |
| Episodes 766–1,531 | −6.10 |
| Episodes 1,532–2,297 | −5.93 |
| Episodes 2,298–3,063 | **−5.95** |

Reward went from about **−27 in the first ten episodes to −5.9**, so it learned — but it was
essentially **done by the halfway mark**. The last two quartiles are statistically
indistinguishable (−5.93 vs −5.95). Fable derived its 3,063-episode budget from a benchmark that
projected a 3-hour compute allowance, and then spent roughly **half of it after convergence**. The
budget was honoured; it just wasn't needed.

### 2.2 Grok — I-DQN on a simulator it had to write itself

Grok hit a wall early: **CityFlow would not install on the host.** Rather than fall back to SUMO, it
wrote **its own pure-Python microscopic simulator** and trained against that.

This is the single most consequential decision in the whole comparison, and it cuts both ways. It is
genuinely resourceful — the run completed, the code is clean, and the simulator does model queues,
per-lane waiting, spawn processes, phase timing and vehicle progression. But it also means **Grok's
numbers live in a universe it built for itself.** Its micro-sim has far looser car-following than
SUMO's, no lane-change dynamics, and 300 m lanes capped at 40 vehicles. A policy that looks good
there has not been shown to work anywhere else.

**Architecture.** MLP `24 → 64 → 64 → 2`, **5,890 parameters**, shared across the 16 intersections
(independent DQN with a shared network — each agent acts on its own local observation). Only **2
phases**: north–south or east–west. No protected lefts.

That is a materially easier control problem than Fable's. Two actions instead of four, and a third
of the parameters.

**Reward** (`src/env/traffic_env.py:259`), per intersection:

```
r = −1.0 × (pressure / 10)
    + 0.5 × throughput_delta
    − 0.1 × (total_waiting_time / 50)
    − 2.0 × gridlock_flag
    + 0.2 × (previous_pressure − pressure) / 10     ← bonus for reducing pressure
r = clip(r, −20, +20)
```

Also max-pressure at heart, but richer: it explicitly rewards throughput, penalizes gridlock, and
adds a shaping term for *improving* pressure rather than merely having low pressure. More knobs
than Fable's two-term reward, and correspondingly more ways to be mis-tuned.

**Training.** γ=0.95, lr=5e-4, 50k buffer, batch 64, target sync every 200 steps, ε 1.0 → 0.05 over
30k steps. 339 episodes, 122,040 environment steps, 0.23 hours.

Its learning curve is much flatter than Fable's:

| Training quartile | Mean episode return |
|---|---:|
| Episodes 1–84 | −51.98 |
| Episodes 85–169 | −46.29 |
| Episodes 170–253 | −45.28 |
| Episodes 254–339 | **−45.21** |

From −51.98 to −45.21 is a **13% improvement**, and it too plateaus at the halfway point. Compare
Fable, which cut its reward by roughly 4× from a much worse start. (The two reward scales are
different and cannot be compared to each other — only each curve's *shape* is meaningful.)

A 13% gain over 339 episodes raises a fair question that this repo **cannot answer**: how much of
Grok's excellent-looking waiting time is the policy, and how much is a simulator permissive enough
that almost any controller does well? Without a random or fixed-time baseline in that same micro-sim
— and Grok never ran one — there is no way to tell. See §6.

### 2.3 GPT — the scaffolding, and then nothing

GPT's entire implementation is **219 lines across three files.**

**What it genuinely built, and what works:**

- `src/network.py` (67 lines) — really does build the full 4×4 grid via `netconvert`: 16 signalized
  junctions, 16 boundary nodes, bidirectional edges, and a route generator producing valid multi-hop
  trips with per-scenario spawn rates for all 8 scenarios. This is correct, working code.
- `src/simulator.py` (69 lines) — a TraCI environment wrapper. It validates the 16 signal IDs,
  discovers the green phases, returns a 12-dim observation per intersection, and computes a reward
  of `−(mean queue)/50`. Its `step()` raises `ValueError` unless handed **exactly 16 actions**.
- `scripts/smoke_sumo.py` — the one runnable entry point. It passes: SUMO 1.27.0, 1 traffic light,
  1 vehicle inserted and arrived, 26 steps, clean exit.

**What was never written:**

- **No `torch` import anywhere in the source.** It is pinned in `requirements.txt` and never used.
- No Q-network, no replay buffer, no training loop, no checkpoint.
- **Nothing produces the 16 actions `simulator.py` demands.** The environment sits waiting for a
  controller that does not exist.
- No evaluation, no metrics, no visualization. Zero episodes run.
- The files its own README instructs you to run — `run_pipeline.py`, `validate_artifacts.py`,
  `config/config.yaml`, `REPORT.md`, `tests/` — **none of them exist on disk.**

So GPT built the stage and never wrote the actor. The honest summary is that it got the
*environment* right and never started the *agent* — which is the part the brief was actually about.

Because it has no policy, the video labelled `gpt_rollout.mp4` in this repo is **not a GPT result.**
It is GPT's own network running under SUMO's default fixed-time signal program — ordinary timers
that sense nothing. It is included as an un-learned reference point and is labelled
`NO LEARNED CONTROLLER` on every frame.

---

## 3. Design, side by side

| | **Fable** | **Grok** |
|---|---|---|
| Simulator | SUMO 1.27, real car-following | Hand-written Python micro-sim |
| Episode | 1,800 sim-seconds | 360 steps (1 s each) |
| Decision interval | every 5 s (~360 decisions) | every 1 s (360 decisions) |
| Phases (actions) | **4** (incl. protected lefts) | **2** (NS / EW) |
| Observation | 21-dim | 24-dim |
| Network | 21 → 128 → 128 → 4 | 24 → 64 → 64 → 2 |
| Parameters | **19,844** | **5,890** |
| Algorithm | Double DQN | Independent DQN (shared net) |
| Replay buffer | 100,000 | 50,000 |
| Batch | 128 | 64 |
| Target sync | 500 grad steps | 200 steps |
| ε schedule | 1.0 → 0.05 over 40k decisions | 1.0 → 0.05 over 30k steps |
| Reward | pressure + waiting penalty | pressure + throughput + waiting + gridlock + shaping |
| Training | 3,063 eps · ~4.59M sim steps · 1.70 h | 339 eps · 122k env steps · 0.23 h |

Both landed on the same core idea — **parameter-shared, max-pressure-style DQN** — independently.
That convergence is itself a meaningful signal about what the brief invites. They differ most in
*rigour of the environment* (Fable) versus *richness of the reward* (Grok).

---

## 4. Results

### 4.1 First, the thing you cannot do

**You cannot compare Fable's and Grok's raw numbers.** They ran on different simulators.

> Grok's mean waiting time is **7.68 s**. Fable's is **47.68 s**.
>
> This is **not a 6× win.** Grok's episodes are 5× shorter, so queues have less time to build. Its
> car-following is far looser, so they build more slowly. And the two report throughput in different
> units entirely — **veh/h** for Fable, **veh/step** for Grok. These are different quantities on
> different axes. Ranking them would be meaningless.

Two things do survive the difference in simulators, because in both cases the simulator cancels out:
**self-relative degradation** (§4.2) and **resource cost** (§4.4).

### 4.2 The fair comparison: degradation from each run's own baseline

Each scenario's mean waiting time ÷ *that same run's* normal-traffic waiting time. Sorted by
severity. This is the only performance comparison in this report that is safe to read across runs.

| Scenario | Fable | Grok | Both agree? |
|---|---:|---:|---|
| **Partial signal failure** | **4.84×** | **4.48×** | ✅ worst for both |
| **Missing sensors** | **4.25×** | **3.13×** | ✅ 2nd worst for both |
| Noisy sensors | 3.39× | 1.27× | ⚠️ big disagreement |
| Sudden surge | 1.60× | 1.17× | ✅ mild |
| High demand | 1.53× | 1.25× | ✅ mild |
| Road closure | 1.23× | 0.99× | ✅ nearly free |
| Uneven directional | 0.95× | 1.06× | ✅ nearly free |
| *normal (the baseline)* | *20.30 s* | *4.29 s* | — |

**The headline finding: two different algorithms, on two different simulators, independently agree
on which scenarios are hard.** Sensor and signal *failures* dominate; *demand* changes are cheap.
Both policies handle more traffic gracefully and neither handles broken infrastructure well. That
agreement is a stronger result than either run's absolute numbers, because it is the one conclusion
that does not depend on whose simulator you trust.

The one real disagreement is **noisy sensors**, where Fable degrades 3.39× and Grok only 1.27×.
Both inject the *same* noise (σ=0.15), so this is a genuine difference in robustness, not a
severity artifact — see §5.

### 4.3 Absolute numbers (valid *within* a column only)

Read down a column, never across. Mean over 20 seeds per scenario.

**Fable** — SUMO, 1800 s episodes:

| Scenario | Avg wait (s) | Avg travel (s) | Avg queue | Trips | Throughput (veh/h) |
|---|---:|---:|---:|---:|---:|
| Normal | 20.30 | 125.4 | 0.060 | 683 | 1,366.9 |
| Uneven directional | 19.26 | 124.0 | 0.056 | 673 | 1,346.5 |
| Road closure | 25.00 | 130.5 | 0.073 | 675 | 1,349.7 |
| High demand | 31.05 | 141.1 | 0.165 | 1,221 | 2,442.3 |
| Sudden surge | 32.56 | 142.4 | 0.134 | 946 | 1,892.2 |
| Noisy sensors | 68.78 | 178.9 | 0.200 | 670 | 1,339.2 |
| Missing sensors | 86.28 | 176.8 | 0.250 | 642 | 1,283.3 |
| Partial signal failure | 98.22 | 161.8 | 0.290 | 629 | 1,258.0 |
| **Overall (160 eps)** | **47.68** | **147.6** | **0.154** | **767** | **1,534.8** |

**Grok** — own micro-sim, 360-step episodes:

| Scenario | Avg wait (s) | Avg travel (s) | Avg queue | Trips | Throughput (veh/step) |
|---|---:|---:|---:|---:|---:|
| Normal | 4.29 | 85.4 | 0.158 | 330 | 0.92 |
| Road closure | 4.23 | 86.3 | 0.160 | 329 | 0.92 |
| Uneven directional | 4.54 | 87.9 | 0.190 | 375 | 1.04 |
| Sudden surge | 5.00 | 89.7 | 0.253 | 469 | 1.30 |
| High demand | 5.38 | 89.5 | 0.446 | 736 | 2.05 |
| Noisy sensors | 5.45 | 87.5 | 0.195 | 327 | 0.91 |
| Missing sensors | 13.41 | 93.9 | 0.431 | 305 | 0.85 |
| Partial signal failure | 19.18 | **72.3** ⚠️ | 0.541 | 280 | 0.78 |
| **Overall (160 eps)** | **7.68** | **86.6** | **0.297** | **394** | **1.09** |

⚠️ **That 72.3 s is not an improvement.** Grok's travel time under signal failure comes in *below*
its own normal baseline (85.4 s) — while waiting time rises 4.5× and completed trips fall 15%
(330 → 280). This is **completion bias**: inside a fixed 360-step episode, the long trips that
*would* have dragged the average up simply never finish, so they never get counted. It is an
artifact of the metric, not a win, and it is why this figure appears in no chart.

Neither run recorded a single **gridlock event** — 0 events, 0 duration, across all 320 episodes.
See §6 before reading that as robustness.

### 4.4 Resource cost

Both measured on the same CPU-only node, so these are comparable **in kind** — with one caveat each.

| Metric | Fable | Grok | Read it with care |
|---|---:|---:|---|
| Mean inference latency | 0.385 ms | 0.012 ms | **Different definitions.** Fable batches all 16 intersections into one forward pass; Grok times a single agent. Grok's net is also 3.4× smaller. |
| p95 latency | 0.479 ms | 0.016 ms ⚠️ | Grok's benchmark records **no latency at all**, so this is the p95 of 160 per-episode *means* — a systematically narrower statistic than Fable's per-inference p95. Not like-for-like. |
| Peak memory | **388 MB** | **720 MB** | Directly comparable. Grok's smaller model carries the larger footprint. |
| Simulation speed | 574 steps/s | 113 steps/s | Measures the two **simulators**, not the policies. SUMO in-process beats hand-written Python by ~5×. |
| Training wall clock | 1.70 h | 0.23 h | Fable spent ~38× more sim steps (4.59M vs 122k) for ~7× the wall clock. |
| CPU utilisation | 132% | 100% | Fable uses its 2 torch threads; Grok is effectively single-threaded. |

Both are far inside any real-time budget. A traffic signal decides every 1–5 seconds; both models
decide in well under a millisecond. **Inference speed is a non-issue for this problem** — which is
worth saying plainly, because it means the 33× latency gap between them is not a reason to prefer
either.

---

## 5. What the numbers actually tell us

**Both models are far more fragile to broken infrastructure than to heavy traffic.** This is the
clearest result in the repo and both runs found it independently. Doubling the traffic costs
Fable 53% more waiting; breaking a quarter of its signals costs it **384% more.** These agents
learned to manage *flow*, not to survive *failure* — and the reason is plain in their training code:
**neither one ever trained on a failure scenario.**

Fable's training sampler (`src/scenarios.py:106`) varies only demand scale (0.7–1.5×), directional
bias and surge windows — it never sets `sensor_dropout_p`, sensor noise or `tls_failure`. Grok's
339 training episodes drew only from `normal`, `uneven`, `surge`, `variable` and `random_routes`.
Both models met every sensor and signal failure **for the first time at evaluation**. Their poor
showing there measures zero-shot transfer, not trained robustness — and it means the obvious next
move for either run is simply to put failures in the training mix.

(Fable is also mildly out-of-distribution on demand: it trained at 0.7–1.5× and was evaluated at
1.8×, which likely accounts for some of its 1.53× high-demand degradation.)

**Fable's failures are catastrophic; Grok's are merely bad.** This is invisible in the means and
obvious in the spread:

| Scenario | Fable mean ± std | Grok mean ± std |
|---|---|---|
| Partial signal failure | 98.22 ± **61.75** s | 19.18 ± 2.55 s |
| Missing sensors | 86.28 ± **25.68** s | 13.41 ± 1.90 s |
| Normal | 20.30 ± 0.90 s | 4.29 ± 0.32 s |

Fable's standard deviation under signal failure is **63% of its mean.** Some seeds it copes; on
others it comes apart entirely. Grok degrades smoothly and predictably (std 13% of mean). For a
system that controls real infrastructure, *predictably mediocre* under failure is often worth more
than *usually fine, occasionally catastrophic* — and a mean-only comparison hides this completely.

**"Fable degrades harder" is real, but partly confounded — and not in the direction you'd guess.**
The scenarios share names but not severities, and the mismatch does not consistently favour either
run:

| Scenario | Fable's shock | Grok's shock | Whose is harsher? |
|---|---|---|---|
| Partial signal failure | **4 of 16** signals fail | 2 of 16 signals fail | Fable's, 2× |
| High demand | 1.8× base demand | **2.25×** base demand | **Grok's** |
| Missing sensors | 25% sensor dropout | **35%** obs dropout | **Grok's** |
| Noisy sensors | σ = 0.15 | σ = 0.15 | identical |

So Fable's worse signal-failure number is partly just a harsher test. But on **high demand and
missing sensors, Grok faces the bigger shock and still degrades less** — that gap is real, not an
artifact. And on **noisy sensors the tests are identical** (σ=0.15), and Fable degrades 3.39× against
Grok's 1.27×. That is the cleanest evidence in the repo that Fable's policy is genuinely more
sensitive to corrupted input — plausibly because a 4-phase policy leans harder on its observations
than a 2-phase one, which can only ever choose between NS and EW.

**Fable overspent its compute budget by roughly half.** It converged by episode ~1,500 and trained
to 3,063. The extra 1,500 episodes and ~0.85 hours bought a reward change of 0.02. The budget was
derived honestly from a benchmark; it simply wasn't checked against the learning curve.

**Grok's policy barely improved, and we cannot tell how much it matters.** A 13% return improvement
over 339 episodes is a weak learning signal. It is entirely possible that Grok's micro-sim is
permissive enough that a naive controller would post similar waiting times — and because Grok never
evaluated a fixed-time or random baseline in its own simulator, **there is no way to find out from
what's in this repo.** This is not an accusation; it is a gap. It is also the single cheapest
experiment anyone could run to make Grok's result meaningful.

---

## 6. Caveats — the part that matters most

1. **Different simulators.** SUMO's car-following against a hand-rolled Python micro-sim. No raw
   metric from one is on the same axis as the other. This is not a technicality; it invalidates
   every head-to-head absolute comparison.

2. **Different episode lengths and decision intervals.** 1,800 s at 5 s decisions versus 360 steps at
   1 s. Waiting and travel times are both bounded by the episode they were measured in.

3. **Different action spaces.** Fable chooses among 4 phases, Grok among 2. Grok is solving a
   strictly easier control problem, and its network is a third of the size because of it.

4. **Scenario names match; severities do not.** See the table in §5. The degradation ratios measure
   brittleness to *each run's own version* of a stress, not to an identical shock.

5. **Grok's travel time under signal failure is completion bias, not an improvement.** Detailed in
   §4.3. It should never be reported as a win.

6. **Grok's p95 latency is not like-for-like with Fable's.** Its benchmark records no latency, so the
   figure is a p95 of per-episode means — a systematically narrower statistic.

7. **Neither run has a non-RL baseline.** No fixed-time or actuated controller was evaluated by
   either. **So neither can claim to beat conventional traffic signals** — only to beat its own
   degraded self. Both runs admit this in their own reports (Fable §10.6, Grok §11). It is the
   biggest single gap in the whole comparison.

8. **Gridlock never triggered — in either run, in any of the 320 episodes.** Zero events, zero
   duration. That is a metric that went **unexercised**, not robustness that was demonstrated. Fable
   says so explicitly in its own report (§10.7).

9. **Neither model trained on failure scenarios.** Both trained on demand variations only and met
   sensor and signal failures for the first time at evaluation. Their poor showing there is a
   measure of zero-shot transfer, not of trained robustness.

10. **GPT contributes no performance data.** It is in no chart and no results table. Where a number
    would go, the honest answer is that it was never built.

---

## 7. Appendix: scenario definitions

The eight stress scenarios, as each run actually parameterizes them in code. Where these differ, the
degradation ratios in §4.2 are not measuring quite the same thing.

| Scenario | Fable (`config.yaml`) | Grok (`traffic_env.py`) |
|---|---|---|
| **Normal** | base demand 0.45 veh/s | base spawn 0.08 |
| **High demand** | demand × **1.8** | spawn 0.08 → 0.18 (**2.25×**) |
| **Sudden surge** | **3×** demand, t=600–900 s (17% of episode) | **3×** spawn, t=120–180 (17% of episode) |
| **Uneven directional** | W/E × 3.0, N/S × 0.6 | directional bias **2.5** |
| **Road closure** | edges `B1C1`/`C1B1` closed at t=600 | 2 edges closed (one link, both ways) |
| **Noisy sensors** | Gaussian **σ = 0.15** on readings | Gaussian **σ = 0.15** on obs |
| **Missing sensors** | **25%** lane-sensor dropout | **35%** of obs vector masked |
| **Partial signal failure** | **4 of 16** signals revert to fixed-time at t=300 | **2 of 16** signals stuck on NS |

Evaluation protocol is the same shape in both: **8 scenarios × 20 seeds = 160 episodes**, fixed seed
bases for reproducibility, greedy policy from a frozen checkpoint. Both completed all 160 with zero
failures.

---

## Reproducing

Every number above traces to a committed artifact:

| Source | What it holds |
|---|---|
| `Fable/results/aggregate.json` | Fable per-scenario and overall metrics |
| `Fable/results/benchmark.json` | Fable latency / throughput benchmark |
| `Fable/logs/train.jsonl` | Fable's 3,063-episode training log |
| `Grok/artifacts/metrics_aggregate.json` | Grok per-scenario and overall metrics |
| `Grok/artifacts/metrics_per_seed.csv` | Grok's 160 individual episodes |
| `Grok/logs/train.jsonl` | Grok's 339-episode training log |
| `GPT/artifacts/smoke/verification.json` | The full extent of what GPT executed |

The videos replay each frozen checkpoint greedily on the seeds recorded in that run's evaluation.
The replay is deterministic: Grok's normal-traffic replay completes **311 trips**, matching the 311
in its own per-seed CSV. See the [README](README.md#reproducing) for commands.
