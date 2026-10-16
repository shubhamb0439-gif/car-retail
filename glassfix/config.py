"""Load config.yaml and apply command-line overrides."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def with_overrides(cfg: dict[str, Any], **dotted: Any) -> dict[str, Any]:
    """Return a copy of cfg with dotted keys set, e.g. with_overrides(cfg, **{"style.name": "tint"})."""
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


def style_fingerprint(cfg: dict[str, Any]) -> str:
    """Short hash of everything that determines the look of the new glass.

    Identical fingerprints across a car's photos prove they used identical style parameters.
    """
    relevant = {"style": cfg["style"], "layers": cfg["layers"], "cracks": cfg["cracks"]}
    blob = json.dumps(relevant, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]
