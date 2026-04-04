"""Validation helpers for NDAE configuration."""

from __future__ import annotations

from .errors import ConfigError
from .schema import NDAEConfig


def validate_config(config: NDAEConfig) -> None:
    """Validate structural and semantic constraints for an NDAE config."""
    ensure_non_empty_string(config.experiment.name, "experiment.name")
    ensure_non_empty_string(config.experiment.output_root, "experiment.output_root")
    ensure_int(config.experiment.seed, "experiment.seed")

    ensure_non_empty_string(config.data.root, "data.root")
    ensure_positive_int(config.data.image_size, "data.image_size")
    ensure_positive_int(config.data.crop_size, "data.crop_size")
    ensure_positive_int(config.data.n_frames, "data.n_frames")
    if config.data.crop_size > config.data.image_size:
        raise ConfigError("data.crop_size must be less than or equal to data.image_size")

    ensure_positive_int(config.model.dim, "model.dim")
    ensure_non_negative_int(config.model.n_aug_channels, "model.n_aug_channels")
    ensure_non_empty_string(config.model.solver, "model.solver")

    ensure_positive_int(config.train.batch_size, "train.batch_size")
    ensure_positive_float(config.train.lr, "train.lr")
    ensure_bool(config.train.dry_run, "train.dry_run")


def ensure_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string")


def ensure_int(value: int, field_name: str) -> None:
    if type(value) is not int:
        raise ConfigError(f"{field_name} must be an integer")


def ensure_positive_int(value: int, field_name: str) -> None:
    ensure_int(value, field_name)
    if value <= 0:
        raise ConfigError(f"{field_name} must be greater than 0")


def ensure_non_negative_int(value: int, field_name: str) -> None:
    ensure_int(value, field_name)
    if value < 0:
        raise ConfigError(f"{field_name} must be greater than or equal to 0")


def ensure_positive_float(value: float, field_name: str) -> None:
    if type(value) not in {int, float}:
        raise ConfigError(f"{field_name} must be a float")
    if float(value) <= 0.0:
        raise ConfigError(f"{field_name} must be greater than 0")


def ensure_bool(value: bool, field_name: str) -> None:
    if type(value) is not bool:
        raise ConfigError(f"{field_name} must be a boolean")
