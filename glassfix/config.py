"""Load config.yaml and apply command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(Path(path) if path else DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def with_overrides(cfg: dict[str, Any], **dotted: Any) -> dict[str, Any]:
    """Copy of cfg with dotted keys set, e.g. with_overrides(cfg, **{"openai.quality": "high"}). None = skip."""
    out = copy.deepcopy(cfg)
    for key, value in dotted.items():
        if value is None:
            continue
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out
