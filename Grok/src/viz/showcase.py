"""
Polished multi-panel showcase dashboard for Shared I-DQN traffic control.

This is a presentation visual (not the fixed-spec final_traffic_map.png).
Uses real evaluation aggregates: metrics_aggregate.json + eval_viz_aggregate.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from PIL import Image

from src.viz.map_plot import build_network_geometry


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
BG = "#0b0f19"
PANEL = "#121826"
PANEL_EDGE = "#1e293b"
TEXT = "#e8eef7"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
ACCENT2 = "#a78bfa"
GOOD = "#34d399"
WARN = "#fbbf24"
BAD = "#f87171"
GRID = "#1f2937"


def _cong_cmap():
    colors = ["#0f172a", "#1e3a5f", "#0ea5e9", "#fbbf24", "#ef4444"]
    return LinearSegmentedColormap.from_list("showcase_cong", colors, N=256)


def _scenario_order() -> List[str]:
    return [
        "normal",
        "high_demand",
        "sudden_surge",
        "uneven",
        "road_closure",
        "noisy_sensors",
        "missing_sensors",
        "partial_light_failure",
    ]


def _pretty_scenario(name: str) -> str:
    return {
        "normal": "Normal",
        "high_demand": "High demand",
        "sudden_surge": "Sudden surge",
        "uneven": "Uneven flow",
        "road_closure": "Road closure",
        "noisy_sensors": "Noisy sensors",
        "missing_sensors": "Missing sensors",
        "partial_light_failure": "Light failure",
    }.get(name, name.replace("_", " ").title())


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_node_wait(raw: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    for k, v in raw.items():
        if isinstance(k, str) and "," in k:
            a, b = k.split(",")
            out[(int(a), int(b))] = float(v)
        elif isinstance(k, tuple):
            out[k] = float(v)
        else:
            try:
                # e.g. "(0, 1)"
                s = str(k).strip("() ")
                a, b = s.split(",")
                out[(int(a), int(b))] = float(v)
            except Exception:
                continue
    return out


def _metric_mean(block: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(block[key]["mean"])
    except Exception:
        return default


def _draw_kpi_card(ax, title: str, value: str, subtitle: str, color: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(1.2)

    ax.text(0.06, 0.72, title.upper(), fontsize=9, color=MUTED, fontweight="600", va="center")
    ax.text(0.06, 0.40, value, fontsize=22, color=color, fontweight="700", va="center")
    ax.text(0.06, 0.14, subtitle, fontsize=8.5, color=MUTED, va="center")
    # accent bar
    ax.add_patch(Rectangle((0, 0), 0.012, 1, transform=ax.transAxes, color=color, clip_on=False))


def _draw_network(ax, viz: Dict[str, Any], grid_size: int = 4):
    ax.set_facecolor(PANEL)
    ax.set_aspect("equal")
    ax.axis("off")

    nodes, edges = build_network_geometry(grid_size, spacing=1.0)
    lane_q = viz.get("lane_queue", {})
    lane_c = viz.get("lane_congestion", {})
    lane_f = viz.get("lane_flow", {})
    node_w = _parse_node_wait(viz.get("node_wait", {}))

    cmap = _cong_cmap()
    # Use a soft display scale so low real queues still look informative
    cong_scale = (0.0, max(5.0, max((float(v) for v in lane_c.values()), default=1.0) * 1.2))
    queue_scale = (0.0, max(1.0, max((float(v) for v in lane_q.values()), default=1.0) * 1.3))
    wait_scale = (0.0, max(1.0, max((float(v) for v in node_w.values()), default=1.0) * 1.3))
    norm = Normalize(vmin=cong_scale[0], vmax=cong_scale[1])

    offset = 0.045
    for fr, to, direction, lid in edges:
        x0, y0 = nodes[fr]
        x1, y1 = nodes[to]
        dx, dy = x1 - x0, y1 - y0
        length = max(np.hypot(dx, dy), 1e-6)
        ux, uy = dx / length, dy / length
        px, py = -uy * offset, ux * offset
        sx0, sy0 = x0 + ux * 0.14 + px, y0 + uy * 0.14 + py
        sx1, sy1 = x1 - ux * 0.14 + px, y1 - uy * 0.14 + py

        cong = float(lane_c.get(lid, 0.0))
        queue = float(lane_q.get(lid, 0.0))
        flow = float(lane_f.get(lid, 0.0))
        qn = min(max((queue - queue_scale[0]) / (queue_scale[1] - queue_scale[0] + 1e-9), 0.0), 1.0)
        lw = 2.0 + qn * 10.0
        color = cmap(norm(min(max(cong, cong_scale[0]), cong_scale[1])))
        ax.plot([sx0, sx1], [sy0, sy1], color=color, linewidth=lw, solid_capstyle="round", zorder=2, alpha=0.95)

        if flow > 0.01:
            mx, my = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            arr_len = 0.10 + 0.08 * min(flow / 5.0, 1.0)
            ax.annotate(
                "",
                xy=(mx + ux * arr_len, my + uy * arr_len),
                xytext=(mx - ux * arr_len * 0.25, my - uy * arr_len * 0.25),
                arrowprops=dict(arrowstyle="-|>", color="#cbd5e1", lw=0.9, mutation_scale=11),
                zorder=3,
            )

    for node, (x, y) in nodes.items():
        w = float(node_w.get(node, 0.0))
        wn = min(max((w - wait_scale[0]) / (wait_scale[1] - wait_scale[0] + 1e-9), 0.0), 1.0)
        radius = 0.055 + wn * 0.10
        if wn > 0.75:
            face = BAD
        elif wn > 0.45:
            face = WARN
        else:
            face = ACCENT
        circ = Circle((x, y), radius=radius, facecolor=face, edgecolor="#f8fafc", linewidth=1.0, zorder=5, alpha=0.95)
        ax.add_patch(circ)
        ax.text(x, y, f"{node[0]}{node[1]}", ha="center", va="center", fontsize=7.5, color="#0b1220", fontweight="700", zorder=6)

    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    pad = 0.45
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_title("Network congestion map (all eval scenarios)", color=TEXT, fontsize=12, pad=8, fontweight="600")

    # mini legend
    ax.text(
        0.02,
        0.02,
        "color = congestion   ·   thickness = queue   ·   node size = wait",
        transform=ax.transAxes,
        fontsize=8,
        color=MUTED,
        va="bottom",
    )


def _draw_scenario_bars(ax, metrics: Dict[str, Any]):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
    ax.tick_params(colors=MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)

    order = [s for s in _scenario_order() if s in metrics.get("per_scenario", {})]
    waits = [_metric_mean(metrics["per_scenario"][s], "avg_wait") for s in order]
    labels = [_pretty_scenario(s) for s in order]
    colors = []
    for w in waits:
        if w >= 15:
            colors.append(BAD)
        elif w >= 8:
            colors.append(WARN)
        else:
            colors.append(GOOD)

    y = np.arange(len(order))
    bars = ax.barh(y, waits, color=colors, edgecolor="#0f172a", height=0.68, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average waiting time (s)", fontsize=9)
    ax.set_title("Wait by evaluation scenario", fontsize=12, fontweight="600", pad=8)
    ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    xmax = max(waits + [1.0]) * 1.18
    ax.set_xlim(0, xmax)
    for bar, val in zip(bars, waits):
        ax.text(
            val + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}s",
            va="center",
            ha="left",
            fontsize=8.5,
            color=TEXT,
            fontweight="600",
        )


def _draw_throughput_queue(ax, metrics: Dict[str, Any]):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)

    order = [s for s in _scenario_order() if s in metrics.get("per_scenario", {})]
    thr = [_metric_mean(metrics["per_scenario"][s], "throughput") for s in order]
    queue = [_metric_mean(metrics["per_scenario"][s], "avg_queue") for s in order]
    x = np.arange(len(order))
    w = 0.38

    ax.bar(x - w / 2, thr, width=w, color=ACCENT, label="Throughput (trips/step)", edgecolor="#0f172a", zorder=3)
    ax.bar(x + w / 2, queue, width=w, color=ACCENT2, label="Avg queue (veh)", edgecolor="#0f172a", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_scenario(s) for s in order], rotation=28, ha="right", fontsize=8)
    ax.set_title("Throughput vs queue by scenario", fontsize=12, fontweight="600", pad=8)
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", frameon=True, fontsize=8)
    leg.get_frame().set_facecolor(PANEL)
    leg.get_frame().set_edgecolor(PANEL_EDGE)
    for t in leg.get_texts():
        t.set_color(TEXT)


def _draw_radar(ax, metrics: Dict[str, Any]):
    """Normalized multi-metric radar for overall system health (higher = better)."""
    ax.set_facecolor(PANEL)
    overall = metrics.get("overall", {})

    # Convert metrics so higher is better
    wait = _metric_mean(overall, "avg_wait", 10.0)
    queue = _metric_mean(overall, "avg_queue", 1.0)
    thr = _metric_mean(overall, "throughput", 1.0)
    trips = _metric_mean(overall, "completed_trips", 300.0)
    lat = _metric_mean(overall, "policy_inference_latency_ms", 1.0)
    gl = _metric_mean(overall, "gridlock_events", 0.0)

    # Soft normalization for display
    scores = {
        "Low wait": float(np.clip(1.0 - wait / 25.0, 0.05, 1.0)),
        "Low queue": float(np.clip(1.0 - queue / 2.0, 0.05, 1.0)),
        "Throughput": float(np.clip(thr / 2.0, 0.05, 1.0)),
        "Trips": float(np.clip(trips / 600.0, 0.05, 1.0)),
        "Fast infer": float(np.clip(1.0 - lat / 1.0, 0.05, 1.0)),
        "No gridlock": float(np.clip(1.0 - gl / 5.0, 0.05, 1.0)),
    }
    labels = list(scores.keys())
    values = list(scores.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, color=MUTED)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], color=MUTED)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(PANEL_EDGE)
    ax.plot(angles, values, color=ACCENT, linewidth=2.0)
    ax.fill(angles, values, color=ACCENT, alpha=0.25)
    ax.set_title("System health (normalized)", fontsize=12, fontweight="600", color=TEXT, pad=14)


def _draw_footer_notes(ax, metrics: Dict[str, Any]):
    ax.set_facecolor(BG)
    ax.axis("off")
    n = int(metrics.get("n_episodes", 0))
    nf = int(metrics.get("n_failed", 0))
    text = (
        f"Shared Multi-Agent Independent DQN  ·  4×4 grid · 16 intersections  ·  "
        f"{n} eval episodes ({nf} failed)  ·  pure-Python micro-sim  ·  CPU-only"
    )
    ax.text(0.5, 0.55, text, ha="center", va="center", fontsize=9, color=MUTED)
    ax.text(
        0.5,
        0.15,
        "Showcase dashboard (presentation)  ·  Official fixed-spec map remains artifacts/final_traffic_map.png",
        ha="center",
        va="center",
        fontsize=8,
        color="#64748b",
    )


def render_showcase(
    metrics_path: str | Path,
    viz_path: str | Path,
    out_path: str | Path,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    metrics_path = Path(metrics_path)
    viz_path = Path(viz_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = _load_json(metrics_path)
    viz = _load_json(viz_path) if viz_path.exists() else {
        "lane_queue": {},
        "lane_congestion": {},
        "lane_flow": {},
        "node_wait": {},
    }

    overall = metrics.get("overall", {})
    avg_wait = _metric_mean(overall, "avg_wait")
    avg_queue = _metric_mean(overall, "avg_queue")
    thr = _metric_mean(overall, "throughput")
    trips = _metric_mean(overall, "completed_trips")
    gl_events = _metric_mean(overall, "gridlock_events")
    lat = _metric_mean(overall, "policy_inference_latency_ms")

    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "text.color": TEXT,
            "axes.edgecolor": PANEL_EDGE,
            "font.size": 10,
        }
    )

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    gs = GridSpec(
        4,
        12,
        figure=fig,
        height_ratios=[0.14, 0.12, 0.62, 0.12],
        hspace=0.35,
        wspace=0.35,
        left=0.04,
        right=0.97,
        top=0.93,
        bottom=0.05,
    )

    # Header
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_facecolor(BG)
    ax_h.axis("off")
    ax_h.text(
        0.0,
        0.62,
        "Adaptive Traffic-Signal Control",
        fontsize=22,
        fontweight="800",
        color=TEXT,
        va="center",
    )
    ax_h.text(
        0.0,
        0.12,
        "Shared Multi-Agent Independent DQN  ·  Max-pressure reward  ·  Evaluation showcase",
        fontsize=12,
        color=ACCENT,
        va="center",
    )

    # KPI cards
    kpis = [
        ("Avg wait", f"{avg_wait:.1f}s", "mean over 160 episodes", GOOD if avg_wait < 10 else WARN),
        ("Avg queue", f"{avg_queue:.2f}", "vehicles per approach", GOOD if avg_queue < 1 else WARN),
        ("Throughput", f"{thr:.2f}", "completed trips / step", ACCENT),
        ("Completed trips", f"{trips:.0f}", "mean per episode", ACCENT2),
        ("Gridlock events", f"{gl_events:.1f}", "mean per episode", GOOD if gl_events < 0.1 else BAD),
        ("Inference", f"{lat*1000:.1f}µs", "policy latency / agent", ACCENT),
    ]
    for i, (title, value, sub, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[1, i * 2 : i * 2 + 2])
        _draw_kpi_card(ax, title, value, sub, color)

    # Main panels
    ax_map = fig.add_subplot(gs[2, 0:5])
    _draw_network(ax_map, viz)

    ax_bars = fig.add_subplot(gs[2, 5:9])
    _draw_scenario_bars(ax_bars, metrics)

    # right column: radar + thr/queue stacked via nested gridspec
    gs_right = gs[2, 9:12].subgridspec(2, 1, hspace=0.35)
    ax_radar = fig.add_subplot(gs_right[0], projection="polar")
    _draw_radar(ax_radar, metrics)

    ax_tq = fig.add_subplot(gs_right[1])
    # compact thr/queue for top scenarios only to avoid clutter
    _draw_throughput_queue(ax_tq, metrics)

    # Footer
    ax_f = fig.add_subplot(gs[3, :])
    _draw_footer_notes(ax_f, metrics)

    fig.savefig(out_path, dpi=100, facecolor=BG)
    plt.close(fig)

    # Exact pixel dimensions
    im = Image.open(out_path)
    if im.size != (width, height):
        im = im.resize((width, height), Image.Resampling.LANCZOS)
        im.save(out_path)
    return out_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    render_showcase(
        root / "artifacts" / "metrics_aggregate.json",
        root / "artifacts" / "eval_viz_aggregate.json",
        root / "artifacts" / "showcase_dashboard.png",
    )
    print("Wrote", root / "artifacts" / "showcase_dashboard.png")
