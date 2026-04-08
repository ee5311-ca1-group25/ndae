"""Public config loading and serialization helpers."""

from __future__ import annotations

from collections.abc import Mapping
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
    return {
        "experiment": {
            "name": config.experiment.name,
            "output_root": config.experiment.output_root,
            "seed": config.experiment.seed,
        },
        "data": {
            "root": config.data.root,
            "exemplar": config.data.exemplar,
            "image_size": config.data.image_size,
            "crop_size": config.data.crop_size,
            "n_frames": config.data.n_frames,
            "t_S": config.data.t_S,
            "t_E": config.data.t_E,
        },
        "model": {
            "dim": config.model.dim,
            "solver": config.model.solver,
        },
        "rendering": {
            "renderer_type": config.rendering.renderer_type,
            "n_brdf_channels": config.rendering.n_brdf_channels,
            "n_normal_channels": config.rendering.n_normal_channels,
            "n_aug_channels": config.rendering.n_aug_channels,
            "camera_fov": config.rendering.camera_fov,
            "camera_distance": config.rendering.camera_distance,
            "light_intensity": config.rendering.light_intensity,
            "light_xy_position": config.rendering.light_xy_position,
            "height_scale": config.rendering.height_scale,
            "gamma": config.rendering.gamma,
        },
        "train": {
            "runtime": {
                "batch_size": config.train.runtime.batch_size,
                "lr": config.train.runtime.lr,
                "dry_run": config.train.runtime.dry_run,
                "n_iter": config.train.runtime.n_iter,
                "log_every": config.train.runtime.log_every,
                "checkpoint_every": config.train.runtime.checkpoint_every,
                "resume_from": config.train.runtime.resume_from,
            },
            "stage": {
                "n_init_iter": config.train.stage.n_init_iter,
                "refresh_rate_init": config.train.stage.refresh_rate_init,
                "refresh_rate_local": config.train.stage.refresh_rate_local,
            },
            "loss": {
                "loss_type": config.train.loss.loss_type,
                "n_loss_crops": config.train.loss.n_loss_crops,
                "overflow_weight": config.train.loss.overflow_weight,
                "init_height_weight": config.train.loss.init_height_weight,
            },
            "scheduler": {
                "eval_every": config.train.scheduler.eval_every,
                "scheduler_factor": config.train.scheduler.scheduler_factor,
                "scheduler_patience_evals": config.train.scheduler.scheduler_patience_evals,
                "scheduler_min_lr": config.train.scheduler.scheduler_min_lr,
            },
        },
    }
