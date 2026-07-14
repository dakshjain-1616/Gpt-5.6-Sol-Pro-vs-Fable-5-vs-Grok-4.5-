"""
Deterministic 1600x1600 traffic-flow / congestion map visualization.

Visual encodings (fixed scales, clip visuals only):
- Road color: average congestion 0-100%
- Road thickness: average queue 0-40 vehicles
- Intersection marker size: average wait 0-180s
- Arrows: dominant flow direction
- Labels: algorithm, avg wait, avg queue, throughput, gridlock %
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch

from src.metrics.aggregate import clip_visual, normalize_visual


def _cmap():
    # Dark-friendly congestion: deep blue -> cyan -> yellow -> red
    colors = ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ffc857"]
    return LinearSegmentedColormap.from_list("cong", colors, N=256)


def build_network_geometry(grid_size: int = 4, spacing: float = 1.0):
    """Return node positions and edge list for a GxG grid."""
    G = grid_size
    nodes = {}
    for r in range(G):
        for c in range(G):
            # top-down: row 0 at top
            x = c * spacing
            y = (G - 1 - r) * spacing
            nodes[(r, c)] = (x, y)
    edges = []  # (from, to, direction, lane_key_candidates)
    for r in range(G):
        for c in range(G):
            if c + 1 < G:
                edges.append(((r, c), (r, c + 1), "E", f"L_{r}_{c}_{r}_{c+1}_E"))
                edges.append(((r, c + 1), (r, c), "W", f"L_{r}_{c+1}_{r}_{c}_W"))
            if r + 1 < G:
                edges.append(((r, c), (r + 1, c), "S", f"L_{r}_{c}_{r+1}_{c}_S"))
                edges.append(((r + 1, c), (r, c), "N", f"L_{r+1}_{c}_{r}_{c}_N"))
    return nodes, edges


def render_traffic_map(
    viz_agg: Dict[str, Any],
    metrics_labels: Dict[str, Any],
    out_path: str | Path,
    grid_size: int = 4,
    width: int = 1600,
    height: int = 1600,
    congestion_scale: Tuple[float, float] = (0.0, 100.0),
    queue_scale: Tuple[float, float] = (0.0, 40.0),
    wait_scale: Tuple[float, float] = (0.0, 180.0),
    gridlock_scale: Tuple[float, float] = (0.0, 100.0),
    algorithm: str = "Shared-IDQN",
) -> Path:
    """
    Render a single deterministic PNG.

    viz_agg keys: lane_queue, lane_congestion, lane_flow, node_wait (dicts)
    metrics_labels: avg_wait, avg_queue, throughput, gridlock_pct
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic matplotlib state
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d0d0d",
            "axes.facecolor": "#121212",
            "savefig.facecolor": "#0d0d0d",
            "text.color": "#e0e0e0",
            "axes.labelcolor": "#e0e0e0",
            "xtick.color": "#888888",
            "ytick.color": "#888888",
            "font.size": 11,
        }
    )

    nodes, edges = build_network_geometry(grid_size, spacing=1.0)
    lane_q = viz_agg.get("lane_queue", {})
    lane_c = viz_agg.get("lane_congestion", {})
    lane_f = viz_agg.get("lane_flow", {})
    node_w = viz_agg.get("node_wait", {})

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
    ax.set_aspect("equal")
    ax.set_facecolor("#121212")

    cmap = _cmap()
    norm = Normalize(vmin=congestion_scale[0], vmax=congestion_scale[1])

    # Draw edges (offset parallel for bidirectional)
    offset = 0.04
    for fr, to, direction, lid in edges:
        x0, y0 = nodes[fr]
        x1, y1 = nodes[to]
        dx, dy = x1 - x0, y1 - y0
        length = max(np.hypot(dx, dy), 1e-6)
        ux, uy = dx / length, dy / length
        # perpendicular offset
        px, py = -uy * offset, ux * offset
        # shorten a bit so nodes visible
        sx0, sy0 = x0 + ux * 0.12 + px, y0 + uy * 0.12 + py
        sx1, sy1 = x1 - ux * 0.12 + px, y1 - uy * 0.12 + py

        cong = float(lane_c.get(lid, 0.0))
        queue = float(lane_q.get(lid, 0.0))
        flow = float(lane_f.get(lid, 0.0))
        cong_c = clip_visual(cong, *congestion_scale)
        queue_c = clip_visual(queue, *queue_scale)
        # thickness: 1.5 .. 14
        qn = normalize_visual(queue_c, *queue_scale)
        lw = 1.5 + qn * 12.5
        color = cmap(norm(cong_c))

        ax.plot([sx0, sx1], [sy0, sy1], color=color, linewidth=lw, solid_capstyle="round", zorder=2)

        # Dominant flow arrow if flow significant
        if flow > 0.01:
            mx, my = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            arr_len = 0.12 + 0.1 * min(flow / 5.0, 1.0)
            ax.annotate(
                "",
                xy=(mx + ux * arr_len, my + uy * arr_len),
                xytext=(mx - ux * arr_len * 0.3, my - uy * arr_len * 0.3),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#cccccc",
                    lw=0.8,
                    mutation_scale=10,
                ),
                zorder=3,
            )

    # Intersections
    gl_steps = float(viz_agg.get("gridlock_steps", 0))
    gl_pct_raw = float(metrics_labels.get("gridlock_pct", 0.0))
    for node, (x, y) in nodes.items():
        # node_wait keys may be tuple or str
        w = 0.0
        if node in node_w:
            w = float(node_w[node])
        else:
            # try string key
            w = float(node_w.get(str(node), 0.0))
        wn = normalize_visual(clip_visual(w, *wait_scale), *wait_scale)
        radius = 0.04 + wn * 0.12
        # gridlock hotspots: high wait + high local congestion
        face = "#4fc3f7"
        if w > wait_scale[1] * 0.6:
            face = "#ff7043"
        if gl_pct_raw > 20 and w > wait_scale[1] * 0.4:
            face = "#ff1744"
        circ = Circle((x, y), radius=radius, facecolor=face, edgecolor="#eeeeee", linewidth=0.8, zorder=5)
        ax.add_patch(circ)
        ax.text(x, y, f"{node[0]},{node[1]}", ha="center", va="center", fontsize=7, color="#111111", zorder=6)

    # Bounds centered
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    pad = 0.55
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.axis("off")

    # Title / labels
    avg_wait = float(metrics_labels.get("avg_wait", 0.0))
    avg_queue = float(metrics_labels.get("avg_queue", 0.0))
    thr = float(metrics_labels.get("throughput", 0.0))
    gl_pct = float(metrics_labels.get("gridlock_pct", 0.0))
    # Clip label display only
    avg_wait_d = clip_visual(avg_wait, *wait_scale)
    avg_queue_d = clip_visual(avg_queue, *queue_scale)
    gl_pct_d = clip_visual(gl_pct, *gridlock_scale)

    title = (
        f"{algorithm}  |  avg wait={avg_wait_d:.1f}s  avg queue={avg_queue_d:.2f}  "
        f"throughput={thr:.3f}/step  gridlock={gl_pct_d:.1f}%"
    )
    fig.suptitle(title, fontsize=13, color="#f5f5f5", y=0.965)

    # Colorbar for congestion
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.20, 0.035, 0.60, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Avg congestion (%)  |  thickness∝queue  |  node size∝wait  |  arrows=flow", fontsize=9)
    cb.ax.tick_params(labelsize=8, colors="#cccccc")
    cb.outline.set_edgecolor("#555555")

    # Fixed pixel size
    fig.set_size_inches(width / 100.0, height / 100.0)
    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Ensure exact dimensions via Pillow
    from PIL import Image

    im = Image.open(out_path)
    if im.size != (width, height):
        im = im.resize((width, height), Image.Resampling.LANCZOS)
        im.save(out_path)
    return out_path


def merge_viz_aggregates(aggs: list) -> Dict[str, Any]:
    """Average multiple episode viz aggregates for the final map."""
    if not aggs:
        return {
            "lane_queue": {},
            "lane_congestion": {},
            "lane_flow": {},
            "node_wait": {},
            "gridlock_steps": 0,
            "gridlock_events": 0,
            "completed_trips": 0,
        }
    lane_q: Dict[str, float] = {}
    lane_c: Dict[str, float] = {}
    lane_f: Dict[str, float] = {}
    node_w: Dict[Any, float] = {}
    counts_l: Dict[str, int] = {}
    counts_n: Dict[Any, int] = {}
    gl_steps = 0
    gl_events = 0
    completed = 0
    for a in aggs:
        for k, v in a.get("lane_queue", {}).items():
            lane_q[k] = lane_q.get(k, 0.0) + float(v)
            counts_l[k] = counts_l.get(k, 0) + 1
        for k, v in a.get("lane_congestion", {}).items():
            lane_c[k] = lane_c.get(k, 0.0) + float(v)
        for k, v in a.get("lane_flow", {}).items():
            lane_f[k] = lane_f.get(k, 0.0) + float(v)
        for k, v in a.get("node_wait", {}).items():
            node_w[k] = node_w.get(k, 0.0) + float(v)
            counts_n[k] = counts_n.get(k, 0) + 1
        gl_steps += int(a.get("gridlock_steps", 0))
        gl_events += int(a.get("gridlock_events", 0))
        completed += int(a.get("completed_trips", 0))
    n = max(len(aggs), 1)
    for k in list(lane_q.keys()):
        c = max(counts_l.get(k, 1), 1)
        lane_q[k] /= c
        lane_c[k] = lane_c.get(k, 0.0) / c
        lane_f[k] = lane_f.get(k, 0.0) / c
    for k in list(node_w.keys()):
        node_w[k] /= max(counts_n.get(k, 1), 1)
    return {
        "lane_queue": lane_q,
        "lane_congestion": lane_c,
        "lane_flow": lane_f,
        "node_wait": node_w,
        "gridlock_steps": gl_steps / n,
        "gridlock_events": gl_events / n,
        "completed_trips": completed / n,
    }
