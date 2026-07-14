#!/usr/bin/env python3
"""Generate a polished multi-panel showcase dashboard from evaluation results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.viz.showcase import render_showcase


def parse_args():
    p = argparse.ArgumentParser(description="Render presentation showcase dashboard")
    p.add_argument("--metrics-json", default=str(ROOT / "artifacts" / "metrics_aggregate.json"))
    p.add_argument("--viz-json", default=str(ROOT / "artifacts" / "eval_viz_aggregate.json"))
    p.add_argument("--out", default=str(ROOT / "artifacts" / "showcase_dashboard.png"))
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    return p.parse_args()


def main():
    args = parse_args()
    out = render_showcase(
        metrics_path=args.metrics_json,
        viz_path=args.viz_json,
        out_path=args.out,
        width=args.width,
        height=args.height,
    )
    print(f"Showcase dashboard written to: {out}")
    print(f"Open with:  code {out}")
    print(f"Or:         xdg-open {out}")


if __name__ == "__main__":
    main()
