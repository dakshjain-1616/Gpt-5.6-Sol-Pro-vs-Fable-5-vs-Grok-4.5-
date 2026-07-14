# Three Agents, One Brief — Comparison Report

Fable, Grok and GPT were each given the same task, the same machine, and no help:

> Train a reinforcement-learning controller for a 4×4 grid of 16 traffic signals. Evaluate it across
> eight stress scenarios. Show that it holds up.

This report covers what each one actually built, what the numbers say, and — importantly — which
numbers you are not allowed to compare. Every figure is read from the runs' own artifacts, and every
claim about behaviour was checked against the source rather than against the runs' own reports.
**That distinction turned out to matter:** §8 lists eleven places where a run's README or REPORT
describes something its code does not do.

**Contents**

1. [What happened](#1-what-happened)
2. [What each model built](#2-what-each-model-built)
3. [Design, side by side](#3-design-side-by-side)
4. [Results](#4-results)
5. [What the numbers actually tell us](#5-what-the-numbers-actually-tell-us)
6. [Caveats](#6-caveats)
7. [Appendix A: scenario definitions](#7-appendix-a-scenario-definitions)
8. [Appendix B: claims the code does not support](#8-appendix-b-claims-the-code-does-not-support)

---

## 1. What happened

| | **Fable** | **Grok** | **GPT** |
|---|---|---|---|
| **Outcome** | ✅ Complete | ✅ Complete | ❌ Agent never written |
| Algorithm | Parameter-shared **Double** DQN | Shared I-DQN (**vanilla**, not Double) | — |
| Simulator | Eclipse SUMO 1.27 (`libsumo`) | Own 583-line Python micro-sim | SUMO — 26-step smoke test only |
| Model | 19,844 params · 4 phases | 5,890 params · 2 phases | none |
| Training | 3,063 episodes · 1.70 h | 339 episodes · 0.23 h | never started |
| Evaluation | 160/160 episodes, 0 failed | 160/160 episodes, 0 failed | 0 episodes |
| Report + viz | ✅ | ✅ | ❌ |

Fable and Grok both delivered the full pipeline. Neither had a failed evaluation episode. GPT wrote
219 lines, proved SUMO ran on a one-intersection toy network, and stopped — **it appears in no
performance table in this report.**

But "both completed" hides the interesting part. The honest one-line summary of this comparison is:

> **Fable built a rigorous system and then undermined it with a sloppy checkpoint-selection rule.
> Grok built a system that finished, but the evidence suggests its policy barely learned to use its
> own sensors. GPT never started.**

Sections 5 and 8 are where that gets substantiated.

---

## 2. What each model built

### 2.1 Fable — Double DQN on real SUMO

Fable took the most conservative route and mostly executed it well. It rejected CityFlow **up front
and on stated reliability grounds** (`plans/plan.md:7`: *"requires building C++ from source … a
higher-risk path on this node"*) and ran **Eclipse SUMO 1.27 via `libsumo`**, the in-process API.
That single call is why Fable has a real microsimulator and Grok does not.

Each intersection has **4 phases** (NS-through, NS-left, EW-through, EW-left), a 3-second yellow on
every change, and a 5-second minimum green. A practical consequence its own report never mentions:
the yellow is inserted *inside* the 5-second decision interval, so **on any phase change the new
green only gets 2 of its 5 seconds.**

**Architecture.** One MLP, `21 → 128 → 128 → 4`, **19,844 parameters**, shared across all 16
intersections — every intersection's experience trains the same weights. The 21-dim observation is
**purely local**: 8 lane queues, 8 lane occupancies, a 4-way phase one-hot, and time-in-phase. There
is no neighbour information at all.

**Reward** (`src/env.py:264`, with `pressure_coef=0.05`, `waiting_coef=0.02`):

```python
pressure = halted_incoming − halted_outgoing
r = −(0.05 × pressure + 0.02 × halted_incoming)
   = −0.07 × halted_incoming + 0.05 × halted_outgoing
```

**This is genuine max-pressure** — the incoming-*minus*-outgoing form from the PressLight/MPLight
literature. It credits an intersection for cars queued on its *exit* links, which discourages it
from shovelling traffic into an already-jammed neighbour. Since Fable's observation carries no
neighbour features, **this reward is the only mechanism by which its 16 agents coordinate at all.**

**Training.** Double DQN (genuinely — `agent.py:118` argmaxes with the online net and evaluates with
the target), γ=0.95, lr=5e-4, 100k buffer, batch 128, target sync every 500 grad steps, ε 1.0 → 0.05
over 40k decisions (floor at episode 133).

**The learning curve — and why the compute budget was mostly wasted.** Deduplicated from
`logs/train.jsonl` (see §8 — the raw log contains 479 duplicate rows from concurrent processes):

| Milestone | Episode |
|---|---|
| First 10 episodes | mean reward **−25.5** |
| Rolling-100 mean crosses −10 | ep 140 |
| Rolling-100 mean crosses −7 | ep 299 |
| Rolling-100 mean crosses −6 | ep 715 |
| **Everything after** | **flat: ep 300–3062 mean −6.01, last-100 −5.93** |

Fable converged by roughly **episode 300–700** and then trained for another **~2,400 episodes that
bought nothing.** The 3,063-episode budget came from a steps/sec benchmark, not from any convergence
criterion. About **80% of a 1.7-hour run was spent not improving.**

**🚩 And the checkpoint it evaluated was chosen by a confounded rule.** Training demand is randomized
(`demand_scale ~ Uniform(0.7, 1.5)`), and `train.py:97` selects `model_best.pt` on **raw episode
reward with no adjustment for how hard the episode was.** Reward correlates with demand at
**r = −0.70**, and every one of the top-6 episodes drew demand from the bottom of the range:

| Selected as "best" | Reward | Demand scale (range: 0.70–1.50) |
|---|---:|---:|
| **ep 1129 → `model_best.pt`** | **−1.653** | **0.705** |
| ep 1320 | −1.890 | 0.731 |
| ep 762 | −1.892 | 0.707 |
| ep 958 | −2.009 | 0.736 |

So `model_best.pt` is not the best policy — it is **whichever weights happened to be live when the
easiest traffic came up.** Since the curve is flat after ~ep 700 this probably costs little in
practice, but it means the evaluated checkpoint (and the one in the video) was selected essentially
at random among converged checkpoints. Fable's REPORT §8 presents "episode 1129, reward −1.653" as a
result without noting any of this.

### 2.2 Grok — a simulator it had to write itself, and a policy that may not be looking

Grok hit a wall: **CityFlow would not install.** It had pre-planned the fallback (`plans/plan.md:30`)
and wrote **its own 583-line pure-Python micro-simulator.** The run finished because of that
foresight, and the code is clean. But it means **Grok's numbers live in a universe it built for
itself.**

**What the micro-sim does model:** gap-based car-following (`gap < 2 → stop`, `gap < 15 → slow`),
stop-line logic, downstream capacity checks (so spillback exists), multi-hop random routes, min-green
5 / yellow 3 / max-green 60, and an explicit gridlock detector.

**What it does not model, versus SUMO:**
- **No acceleration or deceleration limits** — `veh.speed = v_des` assigns speed instantaneously
  (`python_micro_sim.py:427`). Calling this "vehicle-level car-following" (its REPORT §2) is
  generous.
- **One aggregated lane per approach → no lane changing at all.**
- **No turning-movement conflict resolution.** A vehicle arriving from N on an NS-green may turn east
  straight across the EW movement with nothing stopping it. No protected lefts, no conflict matrix.
- **Intersections have zero internal geometry** — crossing is a teleport to the next lane in one step.
  No junction traversal time, no blocking-the-box.
- Only **2 phases** (NS/EW), an explicit CPU trade-off.

**Architecture.** MLP `24 → 64 → 64 → 2`, **5,890 parameters**, shared. Its 24-dim observation is
richer than Fable's in one important way: it includes **4 neighbour-pressure features**, making Grok
the only run with an explicit coordination signal in its *observation*.

**Reward** (`src/env/traffic_env.py:259`):

```python
r = −1.0 × (pressure / 10)
    + 0.5 × throughput_delta          # global: network trips completed, shared by all agents
    − 0.1 × (own waiting time / 50)
    − 2.0 × gridlock_flag
    + 0.2 × (previous_pressure − pressure) / 10    # potential-based shaping
r = clip(r, −20, +20)
```

**🚩 But Grok's "pressure" is not max-pressure.** `python_micro_sim.py:243` defines it as:

```python
def intersection_pressure(self, node):
    """Max-pressure style: sum of incoming queues."""
    return float(sum(self.lane_queue_length(lid) for lid in self._incoming_lanes(node)))
```

There is **no outgoing/downstream term.** It is a plain queue-length penalty wearing a max-pressure
label. Despite the naming throughout Grok's README and REPORT, **it is Fable that implements real
max-pressure and Grok that does not.** (Grok's REPORT §3 does honestly transcribe the formula, so
this is a framing stretch rather than a lie — but the two runs' reward *names* invite exactly the
wrong conclusion.) Grok compensates with the **global throughput term**, which is its only genuine
coordination signal in the reward.

Also worth stating plainly: Grok uses **vanilla DQN**, not Double DQN (`idqn.py:152` takes a plain
`max` over the target net). It never claims otherwise — but it is the key algorithmic difference
between the two completed runs.

**The learning curve — and the finding that reframes everything.**

| Episode block | Return | ε |
|---|---:|---:|
| 1–30 | −54.30 | 0.83 |
| 30–60 | −51.27 | 0.49 |
| 60–90 | −49.00 | 0.16 |
| **90–120** | **−47.30** | **0.05 ← floor** |
| 150–180 | −41.93 | 0.05 |
| 240–270 | −45.79 | 0.05 |
| 300–330 | −46.82 | 0.05 |

**Every measurable gain lands in episodes 1–90 — exactly the window in which ε anneals from 1.0 to
0.05.** After the exploration floor is reached at episode ~84, a scenario-normalized comparison of
the remaining 256 episodes shows a difference of **0.15 return units on a base of ≈−45**, and
**0.024 s of waiting time.** That is nothing.

In other words: the visible improvement in Grok's curve is largely **"the agent stopped acting
randomly,"** not **"the agent learned."** The last ~250 episodes — roughly 10 of its 14 training
minutes — bought no measurable policy gain. §5 shows why this matters far more than it first appears.

### 2.3 GPT — the scaffolding, and then nothing

GPT's entire implementation is **219 lines across three files.**

**What it genuinely built, and what works:**
- `src/network.py` — really does build the full 4×4 grid via `netconvert`: 16 signalized junctions,
  boundary nodes, valid multi-hop routes, per-scenario spawn rates. It even emits a SHA-256
  `geometry_hash` with every metrics dict — **a reproducibility discipline neither peer has.**
- `src/simulator.py` — a working TraCI environment. Returns a `(16, 12)` observation *plus a
  `(16, 2)` action mask* for signals disabled under failure. Its `step()` demands exactly 16 actions.
- `scripts/smoke_sumo.py` — the one runnable entry point. It passes: 1 traffic light, 1 vehicle, 26
  steps, clean exit.

**What was never written:** no `torch` import anywhere in the source (it is pinned in
`requirements.txt` and never imported), no Q-network, no replay buffer, no training loop, no
checkpoint, no evaluation, no visualization. **Nothing produces the 16 actions `simulator.py`
demands** — the environment sits waiting for a controller that does not exist.

Its reward is also unfinished in a way worth naming, because it reveals where GPT stopped
(`simulator.py:62`):

```python
reward = -(self.queue_sum / max(1, self.step_count)) / 50.
```

`queue_sum` is **cumulative across the whole episode**, so this is the negative *running mean* queue
since episode start. It is **non-Markovian and asymptotically frozen** — 400 steps in, one decision
barely moves a 400-sample running average. And it is **a single global scalar**, despite the
carefully-built per-intersection observations and action masks. The obs/action interface was
designed for 16 agents; the reward was not. **The reward is the piece it hadn't finished.**

So GPT built the stage and never wrote the actor. Its README instructs you to run
`run_pipeline.py`, `validate_artifacts.py`, `config/config.yaml` and read `REPORT.md` — **none of
which exist.**

The video labelled `gpt_rollout.mp4` is therefore **not a GPT result.** It is GPT's own network under
SUMO's default fixed-time program, included as an un-learned reference point and labelled
`NO LEARNED CONTROLLER` on every frame.

---

## 3. Design, side by side

| | **Fable** | **Grok** |
|---|---|---|
| Simulator | SUMO 1.27, real car-following | Own Python micro-sim, no accel/decel limits |
| Episode | 1,800 sim-seconds | 360 steps |
| Phases (actions) | **4** (incl. protected lefts) | **2** (NS / EW) |
| Observation | 21-dim, **purely local** | 24-dim, **includes 4 neighbour pressures** |
| Network | 21 → 128 → 128 → 4 | 24 → 64 → 64 → 2 |
| Parameters | **19,844** | **5,890** |
| Algorithm | **Double** DQN | **Vanilla** DQN |
| Buffer / batch | 100,000 / 128 | 50,000 / 64 |
| Target sync | 500 grad steps | 200 grad steps |
| ε → 0.05 floor | episode 133 | episode 84 |
| Reward | **true max-pressure** (in − out) + waiting | queue penalty + **global throughput** + gridlock + shaping |
| Coordination | via **reward** (outgoing term) | via **observation** (neighbour pressure) + global throughput |
| Checkpoint used | `model_best.pt` — **confounded selection** | `idqn_shared.pt` — simply the final weights |
| Training | 3,063 eps · ~4.59M sim steps · 1.70 h | 339 eps · 122k steps · 0.23 h |

The two runs solved coordination in **opposite** ways — Fable put it in the reward and kept the
observation local; Grok put it in the observation and kept the reward local (plus a global throughput
bonus). Neither used a comms channel or a centralized critic.

---

## 4. Results

### 4.1 First, the thing you cannot do

**Fable's and Grok's raw numbers are not on a common axis.** Different simulators, different episode
lengths, and — importantly — **different metric definitions:**

| | Fable | Grok |
|---|---|---|
| "avg wait" | halted vehicle-seconds ÷ **all departed vehicles** | mean over steps of the mean wait of **vehicles currently in the network** |
| "throughput" | **veh/hour** | **veh/step** |
| Episode | 1,800 sim-s | 360 steps |

> Grok's mean waiting time is **7.68 s**; Fable's is **47.68 s**. This is **not a 6× win.** They are
> different quantities, over different denominators, in different simulators, across episodes 5×
> different in length.

Two things survive, because the simulator cancels out of both: **self-relative degradation** (§4.2)
and **resource cost** (§4.4).

### 4.2 The fair comparison: degradation from each run's own baseline

Each scenario's mean waiting time ÷ *that same run's* normal-traffic waiting time.

| Scenario | Fable | Grok |
|---|---:|---:|
| **Partial signal failure** | **4.84×** | **4.48×** |
| **Missing sensors** | **4.25×** | **3.13×** |
| Noisy sensors | 3.39× | 1.27× |
| Sudden surge | 1.60× | 1.17× |
| High demand | 1.53× | 1.25× |
| Road closure | 1.23× | 0.99× |
| Uneven directional | 0.95× | 1.06× |
| *normal (baseline)* | *20.30 s* | *4.29 s* |

**Both runs independently agree on the ranking: infrastructure and sensor *failures* dominate;
*demand* changes are cheap.** That agreement is the most robust conclusion available here, because
it is the one that does not depend on whose simulator you trust.

**But do not read the Fable-vs-Grok gap as "Grok is more robust."** §5 argues the opposite.

### 4.3 Absolute numbers (valid *within* a column only)

Read down a column, never across. Mean over 20 seeds.

**Fable** — SUMO, 1800 s episodes:

| Scenario | Avg wait (s) | Avg travel (s) | Avg queue | Trips | Throughput (veh/h) |
|---|---:|---:|---:|---:|---:|
| Normal | 20.30 ± 0.90 | 125.4 | 0.060 | 683 | 1,366.9 |
| Uneven directional | 19.26 ± 0.90 | 124.0 | 0.056 | 673 | 1,346.5 |
| Road closure | 25.00 ± 2.43 | 130.5 | 0.073 | 675 | 1,349.7 |
| High demand | 31.05 ± 1.34 | 141.1 | 0.165 | 1,221 | 2,442.3 |
| Sudden surge | 32.56 ± 3.38 | 142.4 | 0.134 | 946 | 1,892.2 |
| Noisy sensors | 68.78 ± 3.16 | 178.9 | 0.200 | 670 | 1,339.2 |
| Missing sensors | 86.28 ± **25.68** | 176.8 | 0.250 | 642 | 1,283.3 |
| Partial signal failure | 98.22 ± **61.75** | 161.8 | 0.290 | 629 | 1,258.0 |
| **Overall (160 eps)** | **47.68** | **147.6** | **0.154** | **767** | **1,534.8** |

**Grok** — own micro-sim, 360-step episodes:

| Scenario | Avg wait (s) | Avg travel (s) | Avg queue | Trips | Throughput (veh/step) |
|---|---:|---:|---:|---:|---:|
| Normal | 4.29 ± 0.32 | 85.4 | 0.158 | 330 | 0.92 |
| Road closure | 4.23 ± 0.27 | 86.3 | 0.160 | 329 | 0.92 |
| Uneven directional | 4.54 ± 0.34 | 87.9 | 0.190 | 375 | 1.04 |
| Sudden surge | 5.00 ± 0.23 | 89.7 | 0.253 | 469 | 1.30 |
| High demand | 5.38 ± 0.25 | 89.5 | 0.446 | 736 | 2.05 |
| Noisy sensors | 5.45 ± 0.40 | 87.5 | 0.195 | 327 | 0.91 |
| Missing sensors | 13.41 ± 1.90 | 93.9 | 0.431 | 305 | 0.85 |
| Partial signal failure | 19.18 ± 2.55 | **72.3** ⚠️ | 0.541 | 280 | 0.78 |
| **Overall (160 eps)** | **7.68** | **86.6** | **0.297** | **394** | **1.09** |

⚠️ **That 72.3 s is not an improvement.** It falls *below* Grok's own normal baseline (85.4 s) while
waiting rises 4.5× and completed trips fall 15% (330 → 280). This is **completion bias**: inside a
fixed 360-step episode, the long trips that would have dragged the average up simply never finish.
An artifact of the metric, not a win — which is why it appears in no chart.

Neither run recorded a single **gridlock event** across all 320 episodes. See §6.

### 4.4 Resource cost

| Metric | Fable | Grok | Read it with care |
|---|---:|---:|---|
| Mean inference latency | 0.385 ms | 0.012 ms | **Different definitions.** Fable batches all 16 intersections into one forward pass; Grok times a single agent. Grok's net is 3.4× smaller. |
| p95 latency | 0.479 ms | 0.016 ms ⚠️ | Grok's benchmark records **no latency at all**; this is the p95 of 160 per-episode *means* — a systematically narrower statistic. Not like-for-like. |
| Peak memory | **388 MB** | **720 MB** | Directly comparable. Grok's smaller model carries the larger footprint. |
| Simulation speed | 574 steps/s | 113 steps/s | Measures the **simulators**, not the policies. |
| Training wall clock | 1.70 h | 0.23 h | Fable used ~38× more sim steps (4.59M vs 122k). |

Both are far inside any real-time budget — a signal decides every 1–5 seconds, and both decide in
well under a millisecond. **Inference cost is a non-issue for this problem**, which is worth saying
plainly: the 33× latency gap is not a reason to prefer either.

---

## 5. What the numbers actually tell us

### 5.1 Both models are far more fragile to broken infrastructure than to heavy traffic

Doubling Fable's traffic costs it 53% more waiting; breaking a quarter of its signals costs it
**384% more.** Both runs found this independently, and the reason is plain in the training code:
**neither one ever trained on a failure scenario.**

Fable's training sampler (`src/scenarios.py:106`) varies only demand scale (0.7–1.5×), directional
bias and surge windows — it never sets sensor noise, dropout, or signal failure. Grok's 339 episodes
drew only from `normal`, `uneven`, `surge`, `variable`, `random_routes`. **Both met every failure for
the first time at evaluation.** Their poor showing there measures *zero-shot transfer*, not trained
robustness — and it makes the obvious next move for either run simply putting failures in the
training mix.

### 5.2 Fable's failures are catastrophic; Grok's are merely bad

This is invisible in the means and obvious in the spread:

| Scenario | Fable | Grok |
|---|---|---|
| Partial signal failure | 98.22 ± **61.75** s (std = **63% of mean**) | 19.18 ± 2.55 s (13%) |
| Missing sensors | 86.28 ± **25.68** s | 13.41 ± 1.90 s |
| Normal | 20.30 ± 0.90 s | 4.29 ± 0.32 s |

On some seeds Fable copes; on others it comes apart entirely, depending on *which* four intersections
fail. Fable's own report admits this. For infrastructure control, **predictably mediocre often beats
usually-fine-occasionally-catastrophic** — and a comparison of means alone would hide it completely.

### 5.3 🚩 Grok's apparent robustness is very likely a policy that isn't looking at its sensors

This is the most important finding in the comparison, and it inverts the naive read of §4.2.

Grok looks dramatically more robust to sensor corruption than Fable — 1.27× against 3.39× under
noisy sensors. But look at what the noise actually does to each observation.

**Grok's traffic features are normalized to tiny magnitudes.** From `traffic_env.py:198-240`:
queue `/40`, count `/40`, pressure `/160`. Grok's own measured queues are 0.16–0.54 vehicles, so
those features carry values of roughly **0.004–0.014**. The noisy-sensors scenario adds
**N(0, 0.15)** — noise that is **10–30× larger than the signal** — and applies it to **all 24
dimensions**, including the phase one-hot and the neighbour pressures.

Grok's observation vector under `noisy_sensors` is, for practical purposes, **destroyed.** Its
waiting time rises 27%.

A controller whose sensors are replaced with noise and which barely notices **was not using those
sensors much to begin with.** And that is exactly what its training curve independently says: all of
its measurable improvement came from ε annealing (§2.2), with no detectable policy gain across the
last 250 episodes.

**Three independent lines of evidence converge on the same conclusion:**

1. Its return improved only 13%, and all of it during exploration decay.
2. A scenario-normalized test over post-floor episodes finds **no measurable improvement.**
3. Destroying its observations costs it almost nothing.

The most parsimonious explanation is that **Grok's policy is close to observation-independent** —
something near a fixed or degenerate controller that its permissive micro-sim rewards perfectly well.
Fable, by contrast, *is* using its observations — which is precisely why corrupting them hurts it so
much. **Fable's higher sensitivity to noise is evidence that it learned something.**

**This is a strongly-supported inference, not a proof**, and the decisive experiment was never run:
neither Grok nor anyone else evaluated a **fixed-time or random baseline inside Grok's own
micro-sim.** That single missing experiment — cheap, maybe an hour's work — is what separates
"Grok's policy is robust" from "Grok's policy is barely a policy." **It is the highest-value thing
anyone could add to this repo.**

### 5.4 The scenario severities differ, and not in a way that rescues the comparison

The names match; the code does not. Crucially, **the differences do not consistently favour either
run**, so they cannot be waved away as a uniform bias:

| Scenario | Fable's shock | Grok's shock | Harsher |
|---|---|---|---|
| Partial signal failure | 4 of 16 signals **revert to fixed-time** (still cycling), injected at **t=300** | 2 of 16 signals **frozen on NS forever** — EW never gets a green — from **t=0** | **Qualitatively different.** Fable's hits more nodes but they keep cycling; Grok's permanently starves one axis at fewer nodes. Not rankable. |
| Missing sensors | 25% of lane sensors zeroed; **phase features survive** | 35% of *all* obs dims zeroed — **can blank the phase one-hot itself** | **Grok's** |
| Noisy sensors | σ=0.15 on queue + occupancy only | σ=0.15 on **all 24 dims** | **Grok's** |
| High demand | 1.8× base | 2.25× base | **Grok's** |
| Road closure | closed **mid-episode at t=600**, with **50% of the fleet rerouting** | lanes **never built** — present from t=0, **no rerouting at all** | **Fable's, by far** |

Two of these deserve calling out:

- **Fable is the only run that injects perturbations mid-episode.** Its closure lands at t=600 and its
  signal failure at t=300, so Fable is tested on **adaptation to a change**. Grok's are baked in at
  t=0, so it is tested on **steady-state operation of an already-degraded network** — a materially
  easier problem, and one its policy has 360 steps to settle into.
- **Grok's road_closure is not a disruption at all.** The closed lanes are never constructed, and
  routes are generated on the already-reduced graph. Its REPORT claims *"throughput remains near
  normal via rerouting"* — there is no rerouting code. Throughput stays normal because nothing was
  ever disrupted. Its 0.99× score is not resilience; it is a different network.

### 5.5 Fable overspent its compute budget by roughly 80%

Converged by episode ~300–700; trained to 3,063. The extra ~2,400 episodes changed the rolling mean
by about 0.1 reward units. The budget was derived honestly from a throughput benchmark — it was
simply never checked against the learning curve. (It was also measured single-process, while the
actual run had **two training processes contending for the same 4 cores** — see §8.)

---

## 6. Caveats

1. **Different simulators.** SUMO's car-following against a hand-rolled Python micro-sim with no
   accel/decel limits, no lane changes and no turning conflicts. No raw metric from one is on the
   same axis as the other.
2. **Different metric definitions.** "Avg wait" uses different denominators in each run; throughput is
   veh/h vs veh/step. See §4.1.
3. **Different action spaces.** 4 phases vs 2. Grok solves a strictly easier control problem.
4. **Scenario names match; severities and even *kinds* do not.** See §5.4.
5. **Grok's travel time under signal failure is completion bias**, not an improvement (§4.3).
6. **Grok's p95 latency is not like-for-like** with Fable's (§4.4).
7. **Neither model ever trained on a failure scenario** (§5.1). Failure results measure zero-shot
   transfer.
8. **Neither run has a non-RL baseline.** No fixed-time or actuated controller was evaluated by
   either. **Neither can claim to beat conventional traffic signals** — only to beat its own degraded
   self. This is the biggest single gap in the comparison, and for Grok specifically it is what makes
   §5.3 unresolvable.
9. **Gridlock never triggered** — 0 events in all 320 episodes, in either run. That is a metric that
   went **unexercised**, not robustness demonstrated.
10. **GPT contributes no performance data.** It is in no chart and no results table.
11. **Fable's evaluated checkpoint was selected by a confounded rule** (§2.1). Its flat post-700 curve
    means this likely costs little, but the selection was effectively arbitrary.

---

## 7. Appendix A: scenario definitions

Baseline demand differs ~3× between the runs *before* any scenario multiplier, and episodes differ 5×
in length. This table is why §4.2's ratios compare "brittleness to each run's own version of a
stress," not to an identical shock.

| | **Fable** (1,800 s eval; base 0.45 veh/s) | **Grok** (360 steps; base ≈1.28 veh/s) |
|---|---|---|
| **normal** | `demand_scale: 1.0` | spawn 0.08/entry-lane/step |
| **high_demand** | **1.8×** | **2.25×** |
| **sudden_surge** | **3.0×** for t ∈ [600, 900) — **17% of episode** | **3.0×** for t ∈ [120, 180) — **17% of episode** |
| **uneven_directional** | bias W/E ×3.0, N/S ×0.6 → **5:1 ratio**, ~83% of trips enter E/W | 🚩 **not directional at all** — the only effect is a **uniform ×1.15 demand bump** (see §8) |
| **road_closure** | 2 edges disallowed **at t=600, mid-episode**; **50% of fleet reroutes** | 2 lanes **never constructed**; present from t=0; **no rerouting** |
| **noisy_sensors** | N(0, 0.15) on **queue + occupancy only**; phase features clean | N(0, 0.15) on **all 24 dims**, incl. phase one-hot and neighbour pressures |
| **missing_sensors** | **25%** of lane sensors zeroed; phase features survive | **35%** of all obs dims zeroed; **can blank the phase one-hot** |
| **partial_*_failure** | **4 of 16** signals revert to a fixed-time program (**still cycling**) at **t=300** | **2 of 16** signals **frozen on NS green for the whole episode** — EW never runs |

Evaluation protocol is the same shape in both: **8 scenarios × 20 seeds = 160 episodes**, seeded,
greedy policy from a frozen checkpoint. Both completed all 160 with zero failures.

---

## 8. Appendix B: claims the code does not support

Checked against source, not against the runs' own reports. These are the reason §5 reads the way it
does.

### Grok

1. **🚩 `uneven` is not uneven.** `traffic_env.py:125` sets `uneven_bias = 2.5`, but its only consumer
   (`traffic_env.py:158`) multiplies the *uniform* spawn rate by **1.15** — and the code comment admits
   it (`# (approx: boost rate)`). The simulator's `_spawn` applies one identical rate to every entry
   lane with **no directional weighting anywhere.** Grok's "uneven directional demand" scenario is a
   **15% uniform demand increase.** It is listed as an asymmetric-demand stress test in the README and
   REPORT — **and it is used as a *training* scenario at p=0.20.**
2. **🚩 `road_closure` has no rerouting.** REPORT §9: *"throughput remains near normal via rerouting in
   the micro-sim."* There is no rerouting code. Closed lanes are never built; routes are generated on
   the reduced graph at spawn. Vehicles never re-plan.
3. **🚩 "Max-pressure" reward has no pressure term** — it is a sum of incoming queues only (§2.2).
4. REPORT §2's "vehicle-level car-following" has **no accel/decel limits** — speed is assigned
   instantaneously.
5. REPORT §4 lists a 50,000 replay buffer; at 16 transitions/step over 122k steps the buffer only ever
   holds the **last ~8.7 episodes**. Not wrong, but the effective memory horizon is far shorter than
   the number implies.

### Fable

6. **🚩 REPORT §4: "The reward is always negative because there is always some waiting traffic."**
   False as written. The reward expands to `−0.07·inc + 0.05·out` and is never clipped. Whenever an
   intersection's outgoing lanes hold more than 1.4× the halted vehicles of its incoming lanes,
   **r is positive.** The claim is then reused in §10 as a "known failure case."
7. **🚩 Two training processes ran concurrently and both wrote to the log.** `logs/train.jsonl` holds
   **three** `train_start` events; one resumed from episode 130 and ran to 606 while the main run was
   still going. Result: **479 duplicate episode rows**, and ~20 minutes of two SUMO+torch processes
   contending for 4 cores. Nothing in REPORT §7/§11 mentions this — and the 574.3 steps/s benchmark
   that *sized the training budget* was measured single-process.
8. **🚩 `model_best.pt` selection is confounded by demand difficulty** (r = −0.70; all top-6 episodes
   drew near-minimum demand). Presented in REPORT §8 as a result without qualification. See §2.1.
9. REPORT §6 lists "100,000 transitions | ~1.6M from 16 intersections." The buffer is **full by
   episode 20** and thereafter holds only the last ~21 episodes. 1.6M is transitions *generated*, not
   stored.
10. REPORT §6 explains ε decay as "40,000 decisions ≈ 133 episodes (16 intersections × 300
    decisions)". The **result is right but the arithmetic is nonsense** — 16 × 300 = 4,800. The counter
    ticks once per env decision, so 40,000 / 300 = 133.

### GPT

11. **🚩 The README documents an entire pipeline that does not exist** — `scripts/run_pipeline.py`,
    `scripts/validate_artifacts.py`, `config/config.yaml`, `REPORT.md`, `tests/`. None of these files
    are on disk. `setup.sh` only creates a venv and runs the smoke test.

### Verified good

- Fable's "16 tests, all passing" is **true** (`16 passed`). Grok's suite passes too (`10 passed`).
- Both runs' evaluation harnesses genuinely ran 160/160 episodes with zero failures.
- GPT's `network.py` and `simulator.py` genuinely work; its smoke test genuinely passes.

---

## Reproducing

| Source | What it holds |
|---|---|
| `Fable/results/aggregate.json` · `benchmark.json` | Fable metrics and latency benchmark |
| `Fable/logs/train.jsonl` | Fable's training log (**contains 479 duplicate rows — dedupe by episode**) |
| `Grok/artifacts/metrics_aggregate.json` · `metrics_per_seed.csv` | Grok metrics, per-episode |
| `Grok/logs/train.jsonl` | Grok's 339-episode training log |
| `GPT/artifacts/smoke/verification.json` | The full extent of what GPT executed |

See the [README](README.md#reproducing) for commands.
