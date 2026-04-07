"""Internal helpers for reading config payloads into dataclasses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ndae.rendering import RENDERER_REGISTRY, select_renderer

from .errors import ConfigError
from .schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    NDAEConfig,
    RenderingConfig,
    TrainConfig,
)


def config_from_mapping(payload: Mapping[str, Any]) -> NDAEConfig:
    expect_keys(
        payload,
        "config",
        {"experiment", "data", "model", "train"},
        optional={"rendering"},
    )

    experiment = build_experiment_config(require_mapping(payload["experiment"], "experiment"))
    data = build_data_config(require_mapping(payload["data"], "data"))
    model = build_model_config(require_mapping(payload["model"], "model"))
    rendering_payload = (
        require_mapping(payload["rendering"], "rendering") if "rendering" in payload else {}
    )
    rendering = build_rendering_config(rendering_payload)
    train = build_train_config(require_mapping(payload["train"], "train"))

    return NDAEConfig(
        experiment=experiment,
        data=data,
        model=model,
        rendering=rendering,
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
    expect_keys(
        payload,
        "data",
        {"root", "exemplar", "image_size", "crop_size", "n_frames"},
        optional={"t_I", "t_S", "t_E"},
    )
    return DataConfig(
        root=read_str(payload, "root", "data"),
        exemplar=read_str(payload, "exemplar", "data"),
        image_size=read_int(payload, "image_size", "data"),
        crop_size=read_int(payload, "crop_size", "data"),
        n_frames=read_int(payload, "n_frames", "data"),
        t_I=read_optional_float(payload, "t_I", "data", default=-2.0),
        t_S=read_optional_float(payload, "t_S", "data", default=0.0),
        t_E=read_optional_float(payload, "t_E", "data", default=10.0),
    )


def build_model_config(payload: Mapping[str, Any]) -> ModelConfig:
    expect_keys(payload, "model", {"dim", "solver"})
    return ModelConfig(
        dim=read_int(payload, "dim", "model"),
        solver=read_str(payload, "solver", "model"),
    )


def build_rendering_config(payload: Mapping[str, Any]) -> RenderingConfig:
    defaults = RenderingConfig()
    expect_keys(
        payload,
        "rendering",
        set(),
        optional={
            "renderer_type",
            "n_brdf_channels",
            "n_normal_channels",
            "n_aug_channels",
            "camera_fov",
            "camera_distance",
            "light_intensity",
            "light_xy_position",
            "height_scale",
            "gamma",
        },
    )

    renderer_type = read_optional_str(
        payload,
        "renderer_type",
        "rendering",
        default=defaults.renderer_type,
    )
    if renderer_type not in RENDERER_REGISTRY:
        supported = ", ".join(RENDERER_REGISTRY)
        raise ConfigError(f"rendering.renderer_type must be one of: {supported}")
    renderer_spec = select_renderer(renderer_type)

    return RenderingConfig(
        renderer_type=renderer_type,
        n_brdf_channels=read_optional_int(
            payload,
            "n_brdf_channels",
            "rendering",
            default=renderer_spec.n_brdf_channels,
        ),
        n_normal_channels=read_optional_int(
            payload,
            "n_normal_channels",
            "rendering",
            default=defaults.n_normal_channels,
        ),
        n_aug_channels=read_optional_int(
            payload,
            "n_aug_channels",
            "rendering",
            default=defaults.n_aug_channels,
        ),
        camera_fov=read_optional_float(
            payload,
            "camera_fov",
            "rendering",
            default=defaults.camera_fov,
        ),
        camera_distance=read_optional_float(
            payload,
            "camera_distance",
            "rendering",
            default=defaults.camera_distance,
        ),
        light_intensity=read_optional_float(
            payload,
            "light_intensity",
            "rendering",
            default=defaults.light_intensity,
        ),
        light_xy_position=read_optional_float_pair(
            payload,
            "light_xy_position",
            "rendering",
            default=defaults.light_xy_position,
        ),
        height_scale=read_optional_float(
            payload,
            "height_scale",
            "rendering",
            default=defaults.height_scale,
        ),
        gamma=read_optional_float(
            payload,
            "gamma",
            "rendering",
            default=defaults.gamma,
        ),
    )


def build_train_config(payload: Mapping[str, Any]) -> TrainConfig:
    expect_keys(
        payload,
        "train",
        {
            "batch_size",
            "lr",
            "dry_run",
            "n_iter",
            "n_init_iter",
            "log_every",
            "checkpoint_every",
            "sample_every",
            "sample_size",
        },
        optional={"resume_from"},
    )
    return TrainConfig(
        batch_size=read_int(payload, "batch_size", "train"),
        lr=read_float(payload, "lr", "train"),
        dry_run=read_bool(payload, "dry_run", "train"),
        n_iter=read_int(payload, "n_iter", "train"),
        n_init_iter=read_int(payload, "n_init_iter", "train"),
        log_every=read_int(payload, "log_every", "train"),
        checkpoint_every=read_int(payload, "checkpoint_every", "train"),
        sample_every=read_int(payload, "sample_every", "train"),
        sample_size=read_int(payload, "sample_size", "train"),
        resume_from=read_optional_str(
            payload,
            "resume_from",
            "train",
            default=None,
        ),
    )


def expect_keys(
    payload: Mapping[str, Any],
    section: str,
    required: set[str],
    *,
    optional: set[str] | None = None,
) -> None:
    known_keys = required | (optional or set())
    actual = set(payload)
    missing = sorted(required - actual)
    extra = sorted(actual - known_keys)
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


def read_optional_float(
    payload: Mapping[str, Any],
    key: str,
    section: str,
    *,
    default: float,
) -> float:
    if key not in payload:
        return default
    return read_float(payload, key, section)


def read_optional_str(
    payload: Mapping[str, Any],
    key: str,
    section: str,
    *,
    default: str | None,
) -> str | None:
    if key not in payload:
        return default
    if payload[key] is None:
        return None
    return read_str(payload, key, section)


def read_optional_int(
    payload: Mapping[str, Any],
    key: str,
    section: str,
    *,
    default: int,
) -> int:
    if key not in payload:
        return default
    return read_int(payload, key, section)


def read_bool(payload: Mapping[str, Any], key: str, section: str) -> bool:
    value = payload[key]
    if type(value) is not bool:
        raise ConfigError(f"{section}.{key} must be a boolean")
    return value


def read_optional_float_pair(
    payload: Mapping[str, Any],
    key: str,
    section: str,
    *,
    default: tuple[float, float],
) -> tuple[float, float]:
    if key not in payload:
        return default

    value = payload[key]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{section}.{key} must be a sequence of two floats")

    x, y = value
    if type(x) not in {int, float} or type(y) not in {int, float}:
        raise ConfigError(f"{section}.{key} must be a sequence of two floats")
    return (float(x), float(y))


def require_mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section} must be a mapping")
    return value
