"""Public config loading and serialization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ._parsing import config_from_mapping, require_mapping
from .schema import NDAEConfig
from .validation import validate_config


def load_config(path: str | Path) -> NDAEConfig:
    """Load and validate an NDAE config from a YAML file."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config = require_mapping(payload, "config")
    config = config_from_mapping(raw_config)
    validate_config(config, base_dir=Path.cwd())
    return config


def to_dict(config: NDAEConfig) -> dict[str, Any]:
    """Convert a config dataclass tree back to plain dictionaries."""
    return asdict(config)
