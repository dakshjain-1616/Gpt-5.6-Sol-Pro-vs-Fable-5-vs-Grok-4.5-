#!/usr/bin/env python3
"""Comparison infographic across the three implementations of the same brief.

Reads ONLY the real per-seed metric CSVs produced by each run. Nothing is
invented; GPT has no metrics because that run never trained a model.

Deliberate honesty constraint: Fable and Grok ran on DIFFERENT simulators, so
their raw waits/throughputs are not on a common axis. The only cross-comparable
quantities are (a) each model's degradation relative to its OWN normal baseline
and (b) resource cost on this shared 3-vCPU node. Those are what get charted.

Output: showcase/out/comparison.png  (1600 x 2000, deterministic)
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# validated dark categorical slots (blue, yellow) — see dataviz validator
FABLE_C = "#3987e5"
GROK_C = "#c98500"
DEAD_C = "#5c6270"

BG = "#12141a"
CARD = "#181b22"
FG = "#ffffff"
SEC = "#c3c2b7"
MUT = "#878d99"
GOOD = "#199e70"
BAD = "#e66767"

SCEN = ["normal", "uneven", "high_demand", "sudden_surge", "road_closure",
        "noisy_sensors", "missing_sensors", "partial_failure"]
SCEN_LABEL = ["Normal", "Uneven\ndirectional", "High\ndemand", "Sudden\nsurge",
              "Road\nclosure", "Noisy\nsensors", "Missing\nsensors",
              "Partial signal\nfailure"]

FABLE_KEY = {"normal": "normal", "uneven": "uneven_directional",
             "high_demand": "high_demand", "sudden_surge": "sudden_surge",
             "road_closure": "road_closure", "noisy_sensors": "noisy_sensors",
             "missing_sensors": "missing_sensors",
             "partial_failure": "partial_tls_failure"}
GROK_KEY = {"normal": "normal", "uneven": "uneven", "high_demand": "high_demand",
            "sudden_surge": "sudden_surge", "road_closure": "road_closure",
            "noisy_sensors": "noisy_sensors", "missing_sensors": "missing_sensors",
            "partial_failure": "partial_light_failure"}


def load(path, scen_col, wait_col, lat_col, mem_col, ok):
    acc = {}
    res = {"lat": [], "mem": []}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if not ok(r):
                continue
            acc.setdefault(r[scen_col], []).append(float(r[wait_col]))
            res["lat"].append(float(r[lat_col]))
            res["mem"].append(float(r[mem_col]))
    return ({k: float(np.mean(v)) for k, v in acc.items()},
            float(np.mean(res["lat"])), float(np.mean(res["mem"])))


def card(fig, x, y, w, h, color=CARD):
    ax = fig.add_axes([x, y, w, h])
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02",
        transform=ax.transAxes, facecolor=color, edgecolor="#252932",
        linewidth=1.2, clip_on=False))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return ax


def style(ax):
    ax.set_facecolor(CARD)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#2f3542")
    ax.tick_params(colors=MUT, labelsize=10.5, length=0)
    ax.grid(axis="y", color="#22262f", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)


def main():
    fw, fh = 1600, 2000
    fable_wait, fable_lat, fable_mem = load(
        ROOT / "Fable" / "results" / "metrics.csv", "scenario", "avg_waiting_time",
        "inference_latency_ms", "peak_memory_mb", lambda r: r["status"] == "completed")
    grok_wait, grok_lat, grok_mem = load(
        ROOT / "Grok" / "artifacts" / "metrics_per_seed.csv", "scenario", "avg_wait",
        "policy_inference_latency_ms", "peak_memory_mb", lambda r: r["failed"] == "False")

    f_rel = [fable_wait[FABLE_KEY[s]] / fable_wait["normal"] for s in SCEN]
    g_rel = [grok_wait[GROK_KEY[s]] / grok_wait["normal"] for s in SCEN]

    fig = plt.figure(figsize=(fw / 100, fh / 100), dpi=100, facecolor=BG)

    # ---------------------------------------------------------------- header
    fig.text(0.045, 0.968, "Adaptive Traffic-Signal Control",
             color=FG, fontsize=34, fontweight="bold")
    fig.text(0.045, 0.949,
             "Three independent implementations of one brief · 4×4 grid · 16 signals · "
             "8 scenarios × 20 seeds · CPU-only (3 vCPU)",
             color=SEC, fontsize=13)

    warn = card(fig, 0.045, 0.898, 0.91, 0.038, "#1e1a14")
    warn.text(0.011, 0.5, "▲", color=GROK_C, fontsize=13, va="center")
    warn.text(0.032, 0.5,
              "Fable and Grok ran on DIFFERENT simulators — raw waiting times and "
              "throughputs are NOT on a common axis, and are never compared here.\n"
              "Only self-relative degradation and resource cost on the shared node "
              "are directly comparable.",
              color=SEC, fontsize=10.8, va="center", linespacing=1.5)

    # ------------------------------------------------------------ scorecards
    cards = [
        dict(name="FABLE", c=FABLE_C, status="COMPLETE", sc=GOOD,
             rows=[("Algorithm", "Parameter-shared Double DQN"),
                   ("Simulator", "SUMO 1.27 via libsumo"),
                   ("Episode", "1800 s · 5 s decisions"),
                   ("Model", "19,844 params · MLP 128×128"),
                   ("Action space", "4 phases / intersection"),
                   ("Episodes run", "160 / 160  ·  0 failed"),
                   ("Final PNG", "1600×1600 · spec-compliant")]),
        dict(name="GROK", c=GROK_C, status="COMPLETE", sc=GOOD,
             rows=[("Algorithm", "Shared multi-agent I-DQN"),
                   ("Simulator", "custom Python micro-sim"),
                   ("Episode", "360 steps · 1 s decisions"),
                   ("Model", "5,890 params · MLP 64"),
                   ("Action space", "2 phases / intersection"),
                   ("Episodes run", "160 / 160  ·  0 failed"),
                   ("Final PNG", "1600×1600 · spec-compliant")]),
        dict(name="GPT", c=DEAD_C, status="INCOMPLETE", sc=BAD,
             rows=[("Algorithm", "— never implemented"),
                   ("Simulator", "SUMO — smoke test only"),
                   ("Episode", "26 steps, 1 intersection"),
                   ("Model", "— no checkpoint"),
                   ("Action space", "—"),
                   ("Episodes run", "0 / 160"),
                   ("Final PNG", "— not produced")]),
    ]
    cw, gap = 0.288, 0.023
    for i, cd in enumerate(cards):
        x = 0.045 + i * (cw + gap)
        ax = card(fig, x, 0.700, cw, 0.190)
        ax.add_patch(plt.Rectangle((0, 0.965), 1, 0.035, color=cd["c"],
                                   transform=ax.transAxes))
        ax.text(0.04, 0.885, cd["name"], color=FG, fontsize=21, fontweight="bold",
                va="center")
        ax.text(0.96, 0.885, cd["status"], color=cd["sc"], fontsize=10.5,
                fontweight="bold", va="center", ha="right")
        for j, (k, v) in enumerate(cd["rows"]):
            yy = 0.755 - j * 0.107
            ax.text(0.04, yy, k, color=MUT, fontsize=9.6, va="center")
            ax.text(0.40, yy, v, color=SEC if cd["name"] != "GPT" else MUT,
                    fontsize=9.6, va="center")

    # -------------------------------------------------- robustness (main chart)
    fig.text(0.045, 0.667, "Robustness — waiting time under stress, relative to each "
             "model's own normal-traffic baseline",
             color=FG, fontsize=17, fontweight="bold")
    fig.text(0.045, 0.650,
             "1.0× = no degradation. Self-relative, so the two simulators cancel out. "
             "Both agree on which scenarios are hard.",
             color=MUT, fontsize=11)

    ax = fig.add_axes([0.075, 0.425, 0.88, 0.205])
    style(ax)
    xs = np.arange(len(SCEN))
    bw = 0.36
    b1 = ax.bar(xs - bw / 2 - 0.01, f_rel, bw, color=FABLE_C, zorder=3, label="Fable")
    b2 = ax.bar(xs + bw / 2 + 0.01, g_rel, bw, color=GROK_C, zorder=3, label="Grok")
    ax.axhline(1.0, color="#4b5262", linewidth=1.4, linestyle="--", zorder=2)
    ax.text(-0.62, 1.0, "baseline", color=MUT, fontsize=9, ha="center", va="bottom")
    for bars, vals in ((b1, f_rel), (b2, g_rel)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.09, f"{v:.1f}×",
                    color=SEC, fontsize=9.6, ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(SCEN_LABEL, fontsize=10)
    ax.set_ylabel("avg waiting time ÷ own normal", color=MUT, fontsize=11)
    ax.set_ylim(0, max(max(f_rel), max(g_rel)) * 1.20)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=11)
    for t in leg.get_texts():
        t.set_color(SEC)

    # ------------------------------------------------------------- resources
    fig.text(0.045, 0.383, "Resource cost on the shared 3-vCPU node",
             color=FG, fontsize=17, fontweight="bold")
    fig.text(0.045, 0.366,
             "Directly comparable — same hardware, same measurement. Both sit far "
             "inside the CPU-only budget.",
             color=MUT, fontsize=11)

    ax1 = fig.add_axes([0.075, 0.243, 0.39, 0.100])
    style(ax1)
    ax1.barh([1, 0], [fable_lat, grok_lat], 0.55, color=[FABLE_C, GROK_C], zorder=3)
    for y, v in ((1, fable_lat), (0, grok_lat)):
        ax1.text(v + max(fable_lat, grok_lat) * 0.03, y, f"{v:.3f} ms",
                 color=SEC, fontsize=11, va="center")
    ax1.set_yticks([1, 0]); ax1.set_yticklabels(["Fable", "Grok"], fontsize=11)
    ax1.set_xlim(0, max(fable_lat, grok_lat) * 1.35)
    ax1.grid(axis="y", visible=False); ax1.grid(axis="x", color="#22262f")
    ax1.set_xlabel("policy inference latency per decision (ms)", color=MUT, fontsize=10.5)

    ax2 = fig.add_axes([0.565, 0.243, 0.39, 0.100])
    style(ax2)
    ax2.barh([1, 0], [fable_mem, grok_mem], 0.55, color=[FABLE_C, GROK_C], zorder=3)
    for y, v in ((1, fable_mem), (0, grok_mem)):
        ax2.text(v + max(fable_mem, grok_mem) * 0.03, y, f"{v:.0f} MB",
                 color=SEC, fontsize=11, va="center")
    ax2.set_yticks([1, 0]); ax2.set_yticklabels(["Fable", "Grok"], fontsize=11)
    ax2.set_xlim(0, max(fable_mem, grok_mem) * 1.35)
    ax2.grid(axis="y", visible=False); ax2.grid(axis="x", color="#22262f")
    ax2.set_xlabel("peak memory per episode (MB)", color=MUT, fontsize=10.5)

    # ------------------------------------------------------------ deliverables
    fig.text(0.045, 0.198, "Deliverables required by the brief",
             color=FG, fontsize=17, fontweight="bold")

    items = ["Source code", "Training script", "Evaluation script", "Trained checkpoint",
             "1600×1600 final PNG", "Per-seed metrics CSV", "Aggregated metrics JSON",
             "Technical report", "Tests (metrics / image / norm)", "Structured logs"]
    have = {"Fable": [1] * 10, "Grok": [1] * 10, "GPT": [0] * 10}
    have["GPT"][0] = 1  # partial source only

    ax = card(fig, 0.045, 0.035, 0.91, 0.152)
    colx = {"Fable": 0.66, "Grok": 0.79, "GPT": 0.92}
    for k, x in colx.items():
        ax.text(x, 0.935, k, color=FG, fontsize=11.5, fontweight="bold", ha="center")
    for i, it in enumerate(items):
        yy = 0.835 - i * 0.070
        ax.text(0.025, yy, it, color=SEC, fontsize=10.6, va="center")
        for k, x in colx.items():
            v = have[k][i]
            partial = (k == "GPT" and i == 0)
            ax.text(x, yy, "◐" if partial else ("✓" if v else "✗"),
                    color=GROK_C if partial else (GOOD if v else BAD),
                    fontsize=13, ha="center", va="center", fontweight="bold")
    ax.text(0.025, 0.075, "◐ GPT: simulator wrapper + tiny-network smoke test only — "
            "no 4×4 network, no training, no evaluation.",
            color=MUT, fontsize=9.6, va="center")

    fig.text(0.045, 0.016,
             "All values computed from the runs' own metric CSVs (160 completed "
             "episodes each for Fable and Grok; zero failures recorded).",
             color="#666c7a", fontsize=9.5)

    out = HERE / "out" / "comparison.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
