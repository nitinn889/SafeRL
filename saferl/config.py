"""Loads YAML config files (e.g. configs/default.yaml) into plain dicts."""
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"


def load_config(path=None) -> dict:
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)
