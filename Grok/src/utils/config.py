"""Config loading utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML config; default to configs/default.yaml under project root."""
    if path is None:
        path = PROJECT_ROOT / "configs" / "default.yaml"
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    return cfg


def resolve_path(rel: str | Path) -> Path:
    """Resolve path relative to project root."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create artifact/checkpoint/log directories from config."""
    paths = cfg.get("paths", {})
    for key in ("checkpoint_dir", "artifact_dir", "log_dir"):
        if key in paths:
            resolve_path(paths[key]).mkdir(parents=True, exist_ok=True)
