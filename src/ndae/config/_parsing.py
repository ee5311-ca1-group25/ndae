"""Internal helpers for reading config payloads into dataclasses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ConfigError
from .schema import DataConfig, ExperimentConfig, ModelConfig, NDAEConfig, TrainConfig


def config_from_mapping(payload: Mapping[str, Any]) -> NDAEConfig:
    expect_keys(payload, "config", {"experiment", "data", "model", "train"})

    experiment = build_experiment_config(require_mapping(payload["experiment"], "experiment"))
    data = build_data_config(require_mapping(payload["data"], "data"))
    model = build_model_config(require_mapping(payload["model"], "model"))
    train = build_train_config(require_mapping(payload["train"], "train"))

    return NDAEConfig(
        experiment=experiment,
        data=data,
        model=model,
        train=train,
    )


def build_experiment_config(payload: Mapping[str, Any]) -> ExperimentConfig:
    expect_keys(payload, "experiment", {"name", "output_root", "seed"})
    return ExperimentConfig(
        name=read_str(payload, "name", "experiment"),
        output_root=read_str(payload, "output_root", "experiment"),
        seed=read_int(payload, "seed", "experiment"),
    )


def build_data_config(payload: Mapping[str, Any]) -> DataConfig:
    expect_keys(payload, "data", {"root", "exemplar", "image_size", "crop_size", "n_frames"})
    return DataConfig(
        root=read_str(payload, "root", "data"),
        exemplar=read_str(payload, "exemplar", "data"),
        image_size=read_int(payload, "image_size", "data"),
        crop_size=read_int(payload, "crop_size", "data"),
        n_frames=read_int(payload, "n_frames", "data"),
    )


def build_model_config(payload: Mapping[str, Any]) -> ModelConfig:
    expect_keys(payload, "model", {"dim", "n_aug_channels", "solver"})
    return ModelConfig(
        dim=read_int(payload, "dim", "model"),
        n_aug_channels=read_int(payload, "n_aug_channels", "model"),
        solver=read_str(payload, "solver", "model"),
    )


def build_train_config(payload: Mapping[str, Any]) -> TrainConfig:
    expect_keys(payload, "train", {"batch_size", "lr", "dry_run"})
    return TrainConfig(
        batch_size=read_int(payload, "batch_size", "train"),
        lr=read_float(payload, "lr", "train"),
        dry_run=read_bool(payload, "dry_run", "train"),
    )


def expect_keys(payload: Mapping[str, Any], section: str, expected: set[str]) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    problems: list[str] = []
    if missing:
        problems.append(f"missing keys: {', '.join(missing)}")
    if extra:
        problems.append(f"unknown keys: {', '.join(extra)}")
    if problems:
        raise ConfigError(f"Invalid {section} section: {'; '.join(problems)}")


def read_str(payload: Mapping[str, Any], key: str, section: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ConfigError(f"{section}.{key} must be a string")
    return value


def read_int(payload: Mapping[str, Any], key: str, section: str) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ConfigError(f"{section}.{key} must be an integer")
    return value


def read_float(payload: Mapping[str, Any], key: str, section: str) -> float:
    value = payload[key]
    if type(value) not in {int, float}:
        raise ConfigError(f"{section}.{key} must be a float")
    return float(value)


def read_bool(payload: Mapping[str, Any], key: str, section: str) -> bool:
    value = payload[key]
    if type(value) is not bool:
        raise ConfigError(f"{section}.{key} must be a boolean")
    return value


def require_mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section} must be a mapping")
    return value
