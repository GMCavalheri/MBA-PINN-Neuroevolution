"""Load the YAML configuration of a problem merged over configs/base.yaml."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(problem: str) -> dict:
    with open(CONFIG_DIR / "base.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(CONFIG_DIR / f"{problem}.yaml", encoding="utf-8") as f:
        specific = yaml.safe_load(f)
    return _merge(base, specific)


def config_hash(cfg: dict) -> str:
    """Short hash identifying the exact configuration used in a run."""
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
