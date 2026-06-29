"""Configuration helpers for LFCS2M inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return data


def merge_cli_overrides(config: Dict[str, Any], *, input_dir: Optional[str] = None, output_dir: Optional[str] = None,
                        checkpoint: Optional[str] = None, steps: Optional[int] = None) -> Dict[str, Any]:
    config = dict(config)
    config.setdefault("io", {})
    config.setdefault("sampling", {})
    if input_dir is not None:
        config["io"]["input_dir"] = input_dir
    if output_dir is not None:
        config["io"]["output_dir"] = output_dir
    if checkpoint is not None:
        config["io"]["checkpoint"] = checkpoint
    if steps is not None:
        config["sampling"]["steps"] = int(steps)
    return config
