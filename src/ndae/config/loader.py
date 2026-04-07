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


_TOP_LEVEL_SECTIONS = {"experiment", "data", "model", "train"}


def load_config(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
    validate_dataset: bool = True,
) -> NDAEConfig:
    """Load and validate an NDAE config from a YAML file."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_payload = require_mapping(payload, "config")
    if _TOP_LEVEL_SECTIONS.issubset(raw_payload):
        raw_config = raw_payload
    else:
        raw_config = require_mapping(raw_payload["config"], "config")
    config = config_from_mapping(raw_config)
    validate_config(
        config,
        base_dir=Path(base_dir) if base_dir is not None else Path.cwd(),
        validate_dataset=validate_dataset,
    )
    return config


def to_dict(config: NDAEConfig) -> dict[str, Any]:
    """Convert a config dataclass tree back to plain dictionaries."""
    return asdict(config)
