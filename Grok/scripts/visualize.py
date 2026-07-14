#!/usr/bin/env python3
"""Generate deterministic 1600x1600 final_traffic_map.png from eval aggregates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import ensure_dirs, load_config, resolve_path
from src.viz.map_plot import render_traffic_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--viz-json", default=None, help="eval_viz_aggregate.json")
    p.add_argument("--metrics-json", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    art = resolve_path(cfg["paths"]["artifact_dir"])
    viz_path = Path(args.viz_json) if args.viz_json else art / "eval_viz_aggregate.json"
    metrics_path = Path(args.metrics_json) if args.metrics_json else resolve_path(cfg["paths"]["metrics_json"])
    out = Path(args.out) if args.out else resolve_path(cfg["paths"]["final_png"])

    if not viz_path.exists():
        # Fallback empty geometry so PNG still generates
        viz = {
            "lane_queue": {},
            "lane_congestion": {},
            "lane_flow": {},
            "node_wait": {},
            "gridlock_steps": 0,
        }
        print(f"WARNING: missing {viz_path}, using empty viz")
    else:
        with open(viz_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # parse node_wait keys "r,c" -> tuple
        node_wait = {}
        for k, v in raw.get("node_wait", {}).items():
            if isinstance(k, str) and "," in k:
                a, b = k.split(",")
                node_wait[(int(a), int(b))] = float(v)
            else:
                node_wait[k] = float(v)
        viz = {
            "lane_queue": {k: float(v) for k, v in raw.get("lane_queue", {}).items()},
            "lane_congestion": {k: float(v) for k, v in raw.get("lane_congestion", {}).items()},
            "lane_flow": {k: float(v) for k, v in raw.get("lane_flow", {}).items()},
            "node_wait": node_wait,
            "gridlock_steps": raw.get("gridlock_steps", 0),
            "gridlock_events": raw.get("gridlock_events", 0),
            "completed_trips": raw.get("completed_trips", 0),
        }

    labels = {
        "avg_wait": 0.0,
        "avg_queue": 0.0,
        "throughput": 0.0,
        "gridlock_pct": 0.0,
    }
    algorithm = "Shared-IDQN"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        labels["avg_wait"] = float(m.get("label_avg_wait", 0.0))
        labels["avg_queue"] = float(m.get("label_avg_queue", 0.0))
        labels["throughput"] = float(m.get("label_throughput", 0.0))
        labels["gridlock_pct"] = float(m.get("label_gridlock_pct", 0.0))
        algorithm = str(m.get("algorithm", algorithm))

    vcfg = cfg.get("visualization", {})
    render_traffic_map(
        viz_agg=viz,
        metrics_labels=labels,
        out_path=out,
        grid_size=int(cfg.get("network", {}).get("grid_size", 4)),
        width=int(vcfg.get("width", 1600)),
        height=int(vcfg.get("height", 1600)),
        congestion_scale=tuple(vcfg.get("congestion_scale", [0.0, 100.0])),
        queue_scale=tuple(vcfg.get("queue_scale", [0.0, 40.0])),
        wait_scale=tuple(vcfg.get("wait_scale", [0.0, 180.0])),
        gridlock_scale=tuple(vcfg.get("gridlock_scale", [0.0, 100.0])),
        algorithm=algorithm,
    )
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
