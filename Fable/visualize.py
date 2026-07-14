#!/usr/bin/env python3
"""Render the single final visualization PNG (exactly 1600x1600 px).

Deterministic: same inputs -> byte-identical image. Reads only files under
results/ plus the fixed network geometry.

Visual encoding (all clipping is VISUAL ONLY; raw metric files untouched):
    road colour      = avg congestion (occupancy %), fixed 0-100% scale
    road thickness   = avg queue per road, fixed 0-40 vehicles scale
    marker size      = avg intersection wait, fixed 0-180 s scale
    arrows           = dominant flow direction per road pair
    red glow zones   = bottleneck/gridlock (congestion > 60%)
    text block       = algorithm + avg wait + avg queue + throughput + gridlock %
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sumolib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from src.utils import load_config


def clip01(v: float, lo: float, hi: float) -> float:
    """Normalize v to [0,1] with visual clipping to the fixed scale [lo, hi]."""
    if hi <= lo:
        raise ValueError("bad scale")
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--spatial", default=None)
    ap.add_argument("--aggregate", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    vz = cfg["visualization"]
    results_dir = Path(cfg["paths"]["results_dir"])
    spatial_path = Path(args.spatial or results_dir / "spatial.json")
    agg_path = Path(args.aggregate or results_dir / "aggregate.json")
    out_path = Path(args.output) if args.output else Path(
        cfg["paths"]["results_dir"]) / Path(vz["output"]).name

    with open(spatial_path) as f:
        spatial = json.load(f)
    with open(agg_path) as f:
        agg = json.load(f)

    net = sumolib.net.readNet(cfg["paths"]["net_file"])
    edges = spatial["edges"]
    occ = dict(zip(edges, spatial["edge_avg_occupancy"]))
    queue = dict(zip(edges, spatial["edge_avg_queue"]))
    veh = dict(zip(edges, spatial["edge_avg_vehicles"]))
    tls_wait = spatial["tls_avg_wait"]

    c_lo, c_hi = vz["congestion_scale"]
    q_lo, q_hi = vz["queue_scale"]
    w_lo, w_hi = vz["wait_scale"]
    size_px, dpi = int(vz["size_px"]), int(vz["dpi"])
    bg = vz["background"]

    cmap = LinearSegmentedColormap.from_list(
        "cong", ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])

    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(bg)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------- geometry
    # Deduplicate bidirectional pairs: draw each road once per direction with
    # a small perpendicular offset so both directions are visible.
    drawn_pairs = set()
    inner_nodes = {f"{c}{r}": net.getNode(f"{c}{r}").getCoord()
                   for c in "ABCD" for r in range(4)}
    xs = [c[0] for c in inner_nodes.values()]
    ys = [c[1] for c in inner_nodes.values()]
    pad = 180.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    def edge_endpoints(eid):
        e = net.getEdge(eid)
        return np.array(e.getFromNode().getCoord()), np.array(e.getToNode().getCoord())

    # bottleneck zones first (under roads)
    for eid in sorted(edges):
        cong_pct = 100.0 * occ[eid] / 100.0 if occ[eid] > 1.0 else 100.0 * occ[eid]
        # libsumo occupancy is already in percent [0,100]
        cong_pct = float(occ[eid])
        if cong_pct > 60.0:
            p0, p1 = edge_endpoints(eid)
            mid = (p0 + p1) / 2
            ax.add_patch(plt.Circle(mid, 70, color="#ff2222",
                                    alpha=0.18, zorder=1))

    for eid in sorted(edges):
        p0, p1 = edge_endpoints(eid)
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 1:
            continue
        u = d / L
        perp = np.array([-u[1], u[0]])
        off = perp * 7.0                      # separate the two directions
        a, b = p0 + off, p1 + off
        cong = clip01(float(occ[eid]), c_lo, c_hi)
        width = 1.0 + 7.0 * clip01(float(queue[eid]), q_lo, q_hi)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=cmap(cong),
                linewidth=width, solid_capstyle="round", zorder=2)
        drawn_pairs.add(eid)

    # dominant-flow arrows: for each bidirectional pair draw one arrow in the
    # direction with more average vehicles
    seen = set()
    for eid in sorted(edges):
        e = net.getEdge(eid)
        fid, tid_ = e.getFromNode().getID(), e.getToNode().getID()
        key = tuple(sorted((fid, tid_)))
        if key in seen:
            continue
        seen.add(key)
        rev = None
        for cand in edges:
            ce = net.getEdge(cand)
            if ce.getFromNode().getID() == tid_ and ce.getToNode().getID() == fid:
                rev = cand
                break
        dom = eid if rev is None or veh[eid] >= veh[rev] else rev
        p0, p1 = edge_endpoints(dom)
        mid = p0 + (p1 - p0) * 0.5
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        ax.annotate("", xy=mid + d * 26, xytext=mid - d * 26,
                    arrowprops=dict(arrowstyle="-|>", color="#c8cdd8",
                                    lw=1.4, alpha=0.85), zorder=4)

    # intersection markers sized by avg wait
    for tid, coord in sorted(inner_nodes.items()):
        wait = float(tls_wait.get(tid, 0.0))
        s = 120 + 1400 * clip01(wait, w_lo, w_hi)
        ax.scatter([coord[0]], [coord[1]], s=s, color="#dfe6f2",
                   edgecolors="#7f8c9b", linewidths=1.5, zorder=5)
        ax.text(coord[0], coord[1] - 34, tid, color="#9aa3b2",
                fontsize=8, ha="center", zorder=6)

    # ---------------------------------------------------------------- text
    ov = agg["overall"]
    ep_time = cfg["evaluation"]["episode_length"]
    gl_mean = ov["gridlock_duration"]["mean"] or 0.0
    gridlock_pct = float(np.clip(100.0 * gl_mean / ep_time, 0.0, 100.0))
    txt = (
        f"Algorithm: {agg.get('algorithm', 'Parameter-shared Double DQN')}\n"
        f"Avg waiting time: {ov['avg_waiting_time']['mean']:.1f} s\n"
        f"Avg queue length: {ov['avg_queue_length']['mean']:.2f} veh\n"
        f"Throughput: {ov['throughput']['mean']:.0f} veh/h\n"
        f"Gridlock: {gridlock_pct:.1f} % of episode"
    )
    ax.text(0.03, 0.985, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=15, color="#e8ecf3", family="monospace",
            bbox=dict(facecolor="#20242c", edgecolor="#3a404d",
                      boxstyle="round,pad=0.6", alpha=0.95), zorder=10)

    # colour legend (congestion scale)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=c_lo, vmax=c_hi))
    cax = fig.add_axes([0.70, 0.045, 0.25, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("avg congestion (%)", color="#c8cdd8", fontsize=10)
    cb.ax.tick_params(colors="#c8cdd8", labelsize=9)
    cb.outline.set_edgecolor("#3a404d")

    legend = [
        Line2D([0], [0], color=cmap(0.15), lw=6, label="road: colour=congestion, width=queue (0-40 veh)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#dfe6f2",
               markeredgecolor="#7f8c9b", markersize=11,
               label="intersection: size=avg wait (0-180 s)"),
        Line2D([0], [0], marker=">", color="#c8cdd8", lw=1.2, label="dominant flow"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff2222",
               alpha=0.4, markersize=11, label="bottleneck/gridlock zone"),
    ]
    leg = ax.legend(handles=legend, loc="lower left", fontsize=10,
                    facecolor="#20242c", edgecolor="#3a404d", framealpha=0.95)
    for t in leg.get_texts():
        t.set_color("#e8ecf3")

    fig.savefig(out_path, dpi=dpi, facecolor=bg,
                metadata={"Software": None})   # strip mutable metadata
    plt.close(fig)

    from PIL import Image
    img = Image.open(out_path)
    assert img.size == (size_px, size_px), f"bad size {img.size}"
    print(f"Wrote {out_path} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
