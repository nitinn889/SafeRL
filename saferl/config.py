"""Loads YAML config files (e.g. configs/default.yaml) into plain dicts.

A config may name a parent with a top-level `extends:` key, as a path relative
to the child file. The parent loads first and the child's values override it
key by key, recursively. A variant like space3d.yaml then states only what it
changes, instead of carrying a second copy of every tuned value that could
silently drift from the original.
"""
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None) -> dict:
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    return _merge(load_config(path.parent / parent), cfg)
