"""YAML config loading with dotted-key overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, then apply ``key.sub=value`` overrides (values parsed as YAML)."""
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        set_path(cfg, key, yaml.safe_load(raw))
    return cfg


def set_path(cfg: dict, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for p in parents:
        node = node[p]
    if leaf not in node:
        raise KeyError(f"unknown config key {dotted!r}")
    node[leaf] = value


def make_rng(seed: int) -> np.random.Generator:
    """The only place a Generator is created; pass the result explicitly everywhere."""
    return np.random.default_rng(seed)
