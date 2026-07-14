#!/usr/bin/env python3
"""Render a rollout trace into an MP4 showing the trained policy driving traffic.

One video per model, walking through all 8 evaluation scenarios back to back.
Visual language is identical across models so the two videos can be read the
same way:

    road colour   -> live queue on that road (auto-scaled per model)
    vehicle dot   -> a vehicle; red = stopped, pale = moving at free speed
    node colour   -> active green phase (teal = N/S green, amber = E/W green)
    node ring     -> red X ring when that signal has failed
    HUD           -> live state + the metrics this scenario actually scored
                     in the original 20-seed evaluation

Usage:  <venv>/bin/python render_video.py --model fable|grok
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

BG = "#0d0f13"
FG = "#e8eaee"
DIM = "#7b8290"
NS_GREEN = "#2fd4a7"   # node showing a north/south green
EW_GREEN = "#f5a623"   # node showing an east/west green
FAIL_RED = "#ff4d5a"

# road congestion ramp: quiet slate -> amber -> hot red
ROAD_CMAP = LinearSegmentedColormap.from_list(
    "road", ["#232833", "#2b6b8a", "#4fa3c7", "#e9c46a", "#f4772e", "#e03131"])
# vehicle ramp: stopped (red) -> free-flowing (pale blue)
VEH_CMAP = LinearSegmentedColormap.from_list(
    "veh", ["#ff4d5a", "#f5a623", "#9fd6e8", "#dbe7ee"])

MODELS = {
    "fable": {
        "title": "FABLE",
        "algo": "Parameter-shared Double DQN",
        "sim": "SUMO (libsumo) · 1800 s episode · 5 s decisions",
        "metrics_csv": ROOT / "Fable" / "results" / "metrics.csv",
        "cols": {"wait": "avg_waiting_time", "queue": "avg_queue_length",
                 "thru": "throughput", "scenario": "scenario"},
        "thru_unit": "veh/h",
        "order": ["normal", "uneven_directional", "high_demand", "sudden_surge",
                  "road_closure", "noisy_sensors", "missing_sensors",
                  "partial_tls_failure"],
        "n_phases": 4,
    },
    "grok": {
        "title": "GROK",
        "algo": "Shared multi-agent I-DQN (max-pressure reward)",
        "sim": "custom Python micro-sim · 360 steps · 1 s decisions",
        "metrics_csv": ROOT / "Grok" / "artifacts" / "metrics_per_seed.csv",
        "cols": {"wait": "avg_wait", "queue": "avg_queue",
                 "thru": "throughput", "scenario": "scenario"},
        "thru_unit": "veh/step",
        "order": ["normal", "uneven", "high_demand", "sudden_surge",
                  "road_closure", "noisy_sensors", "missing_sensors",
                  "partial_light_failure"],
        "n_phases": 2,
    },
    # GPT never implemented a controller. This is its network under SUMO's
    # default static signal program — a fixed-time BASELINE, not a trained model.
    "gpt": {
        "title": "GPT",
        "algo": "NO LEARNED CONTROLLER — fixed-time signals",
        "sim": "SUMO (traci) · GPT's own 4×4 network · static program",
        "metrics_csv": None,
        "cols": None,
        "thru_unit": "",
        "order": ["normal", "uneven_directional", "high_demand", "sudden_surge",
                  "road_closure", "noisy_sensors", "missing_sensors",
                  "partial_signal_failure"],
        "n_phases": 2,
    },
}

# Shown in place of evaluation metrics for a model that was never trained.
NO_MODEL_NOTE = {
    "noisy_sensors": "fixed-time control reads\nno sensors — this scenario\nis identical to normal",
    "missing_sensors": "fixed-time control reads\nno sensors — this scenario\nis identical to normal",
}

PRETTY = {
    "normal": "NORMAL TRAFFIC",
    "uneven": "UNEVEN DIRECTIONAL",
    "uneven_directional": "UNEVEN DIRECTIONAL",
    "high_demand": "HIGH DEMAND",
    "sudden_surge": "SUDDEN SURGE",
    "road_closure": "ROAD CLOSURE",
    "noisy_sensors": "NOISY SENSORS",
    "missing_sensors": "MISSING SENSORS",
    "partial_tls_failure": "PARTIAL SIGNAL FAILURE",
    "partial_light_failure": "PARTIAL SIGNAL FAILURE",
    "partial_signal_failure": "PARTIAL SIGNAL FAILURE",
}


def scenario_means(cfg):
    """Mean metrics per scenario across all 20 evaluated seeds (from the real CSV)."""
    if cfg["metrics_csv"] is None:
        return {}          # model was never trained/evaluated — nothing to show
    c = cfg["cols"]
    acc = {}
    with open(cfg["metrics_csv"]) as fh:
        for row in csv.DictReader(fh):
            if row.get("status", "completed") != "completed" or row.get("failed") == "True":
                continue
            acc.setdefault(row[c["scenario"]], []).append(
                (float(row[c["wait"]]), float(row[c["queue"]]), float(row[c["thru"]])))
    return {k: tuple(np.mean(v, axis=0)) for k, v in acc.items()}


# --------------------------------------------------------------------- geometry
def load_segments(model: str, traces: Path):
    """Return (segments Nx2x2, seg_key list, node_xy Kx2, bounds)."""
    if model in ("fable", "gpt"):
        g = json.loads((traces / f"{model}_geom.json").read_text())
        segs, keys = [], []
        for e in g["edges"]:
            shape = np.array(e["shape"], dtype=float)
            for a, b in zip(shape[:-1], shape[1:]):
                segs.append([a, b])
                keys.append(e["id"])
        # signal order must match the order used when the trace was recorded
        tls_ids = g.get("tls_order") or [f"{c}{r}" for c in "ABCD" for r in range(4)]
        nodes = np.array([g["tls"][t] for t in tls_ids], dtype=float)
        idx = {eid: i for i, eid in enumerate(g["edge_ids"])}
        seg_qidx = np.array([idx[k] for k in keys])
        return np.array(segs), seg_qidx, nodes, tls_ids, g["bounds"]

    g = json.loads((traces / "grok_geom.json").read_text())
    segs, seg_qidx = [], []
    lane_ids = g["lane_ids"]
    idx = {lid: i for i, lid in enumerate(lane_ids)}
    for lane in g["lanes"]:
        s = np.array(lane["start"], float)
        e = np.array(lane["end"], float)
        d = e - s
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        u = d / n
        perp = np.array([-u[1], u[0]])          # offset so the two directions separate
        off = perp * -14.0                       # right-hand side of travel
        segs.append([s + off, e + off])
        seg_qidx.append(idx[lane["id"]])
    nodes = np.array([g["nodes"][k] for k in
                      [f"{r},{c}" for r in range(4) for c in range(4)]], dtype=float)
    node_order = [f"{r},{c}" for r in range(4) for c in range(4)]
    return np.array(segs), np.array(seg_qidx), nodes, node_order, g["bounds"]


# ------------------------------------------------------------------------ render
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--fps", type=int, default=40)
    ap.add_argument("--traces", default=str(HERE / "traces"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = MODELS[args.model]
    traces = Path(args.traces)
    out = Path(args.out or HERE / "out" / f"{args.model}_rollout.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    segs, seg_qidx, nodes, node_order, bounds = load_segments(args.model, traces)
    means = scenario_means(cfg)

    data = {sc: np.load(traces / f"{args.model}_{sc}.npz", allow_pickle=True)
            for sc in cfg["order"]}
    qkey = "lqueue" if args.model == "grok" else "equeue"
    trained = cfg["metrics_csv"] is not None

    # Auto-scale the congestion ramp to this model's own data (98th pct across all
    # scenarios), so differences are visible rather than crushed by a fixed 0-40 range.
    allq = np.concatenate([data[sc][qkey].ravel() for sc in cfg["order"]])
    qmax = float(np.percentile(allq[allq > 0], 98)) if np.any(allq > 0) else 1.0
    qmax = max(qmax, 1.0)
    qnorm = Normalize(0.0, qmax)
    vnorm = Normalize(0.0, 14.0)   # vehicle speed m/s

    # ---- figure: 1920x1080, fixed view, no rescaling between frames
    W, H = 1920, 1080
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.20, 0.03, 0.78, 0.94])
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    pad = 0.06 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)

    roads = LineCollection(segs, linewidths=3.0, cmap=ROAD_CMAP, norm=qnorm, zorder=2)
    roads.set_array(np.zeros(len(segs)))
    ax.add_collection(roads)

    veh = ax.scatter([], [], s=17, c=[], cmap=VEH_CMAP, norm=vnorm,
                     linewidths=0, zorder=4, alpha=0.95)
    sig = ax.scatter(nodes[:, 0], nodes[:, 1], s=190, c=[NS_GREEN] * len(nodes),
                     edgecolors=BG, linewidths=1.8, zorder=5)
    failring = ax.scatter(nodes[:, 0], nodes[:, 1], s=460, facecolors="none",
                          edgecolors=[(0, 0, 0, 0)] * len(nodes), linewidths=2.6, zorder=6)

    # ---- static left panel
    fig.text(0.025, 0.945, cfg["title"], color=FG, fontsize=34, fontweight="bold",
             family="DejaVu Sans")
    fig.text(0.025, 0.915, cfg["algo"],
             color=DIM if trained else FAIL_RED,
             fontsize=12.5, fontweight="normal" if trained else "bold")
    fig.text(0.025, 0.893, cfg["sim"], color=DIM, fontsize=10.5)
    fig.text(0.025, 0.855, "─" * 34, color="#2a2f3a", fontsize=9)

    t_scn = fig.text(0.025, 0.79, "", color=FG, fontsize=19, fontweight="bold")
    t_seed = fig.text(0.025, 0.762, "", color=DIM, fontsize=11)
    t_live = fig.text(0.025, 0.60, "", color=FG, fontsize=13, family="DejaVu Sans Mono",
                      linespacing=1.9, va="top")
    t_eval = fig.text(0.025, 0.355, "", color=DIM, fontsize=11.5,
                      family="DejaVu Sans Mono", linespacing=1.9, va="top")

    legend = (
        "road colour   live queue\n"
        "vehicle dot   red = stopped\n"
        "node teal     N/S green\n"
        "node amber    E/W green\n"
        "red ring      signal failed"
    )
    fig.text(0.025, 0.163, legend, color=DIM, fontsize=10,
             family="DejaVu Sans Mono", linespacing=1.85, va="top")
    fig.text(0.025, 0.028,
             f"congestion scale auto-fit to this model  (0 – {qmax:.0f} halted veh/road)",
             color="#5a6070", fontsize=8.5)

    prog = fig.add_axes([0.025, 0.838, 0.155, 0.007])
    prog.set_facecolor("#232833")
    prog.set_xticks([]); prog.set_yticks([])
    for s in prog.spines.values():
        s.set_visible(False)
    prog.set_xlim(0, 1); prog.set_ylim(0, 1)
    bar = prog.barh([0.5], [0.0], height=1.0, color=NS_GREEN)[0]

    total = sum(len(data[sc][qkey]) for sc in cfg["order"])
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)

    done = 0
    for sc in cfg["order"]:
        d = data[sc]
        q, ph, fl = d[qkey], d["phase"], d["failed"]
        vall, voff = d["veh"], d["veh_off"]
        nv = d["nveh"]
        seed = int(d["seed"])
        n = len(q)

        t_scn.set_text(PRETTY.get(sc, sc.upper()))
        if trained:
            w, qm, th = means.get(sc, (float("nan"),) * 3)
            t_seed.set_text(f"seed {seed}  ·  frozen checkpoint, no fine-tuning")
            t_eval.set_text(
                "SCORED IN EVALUATION\n"
                "(mean of 20 seeds)\n\n"
                f"avg wait    {w:7.1f} s\n"
                f"avg queue   {qm:7.2f} veh\n"
                f"throughput  {th:7.1f} {cfg['thru_unit']}")
            t_eval.set_color(DIM)
        else:
            t_seed.set_text(f"seed {seed}  ·  no model was ever trained")
            note = NO_MODEL_NOTE.get(
                sc, "GPT produced no controller,\nno checkpoint and no\nevaluation. Signals here\nrun on a fixed timer and\nadapt to nothing.")
            t_eval.set_text("NOT A TRAINED POLICY\n\n" + note)
            t_eval.set_color(FAIL_RED)

        for k in range(n):
            roads.set_array(q[k][seg_qidx])

            pts = vall[voff[k]:voff[k + 1]]
            if len(pts):
                veh.set_offsets(pts[:, :2])
                veh.set_array(pts[:, 2])
            else:
                veh.set_offsets(np.zeros((0, 2)))
                veh.set_array(np.zeros(0))

            phk = ph[k]
            # phase < half of the phase set => N/S axis is green
            ns = phk < (cfg["n_phases"] // 2)
            sig.set_color([NS_GREEN if v else EW_GREEN for v in ns])
            failring.set_edgecolors(
                [FAIL_RED if v else (0, 0, 0, 0) for v in fl[k]])

            qlive = float(q[k].mean())
            nfail = int(fl[k].sum())
            t_live.set_text(
                f"LIVE\n\n"
                f"step        {k + 1:4d} / {n}\n"
                f"vehicles    {int(nv[k]):4d}\n"
                f"mean queue  {qlive:6.2f} veh/road\n"
                f"max queue   {float(q[k].max()):6.0f} veh\n"
                f"signals out {nfail:4d} / 16")

            done += 1
            bar.set_width(done / total)

            fig.canvas.draw()
            ff.stdin.write(fig.canvas.buffer_rgba())

        print(f"  rendered {sc:<22} {n} frames", flush=True)

    ff.stdin.close()
    rc = ff.wait()
    plt.close(fig)
    if rc != 0:
        sys.exit(f"ffmpeg failed rc={rc}")
    mb = out.stat().st_size / 1e6
    print(f"{args.model}: {out}  ({total} frames, {total/args.fps:.0f}s, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
