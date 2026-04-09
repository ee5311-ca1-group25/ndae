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
    TrainLossConfig,
    TrainRuntimeConfig,
    TrainSchedulerConfig,
    TrainStageConfig,
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
        optional={"t_S", "t_E"},
    )
    return DataConfig(
        root=read_str(payload, "root", "data"),
        exemplar=read_str(payload, "exemplar", "data"),
        image_size=read_int(payload, "image_size", "data"),
        crop_size=read_int(payload, "crop_size", "data"),
        n_frames=read_int(payload, "n_frames", "data"),
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
        {"runtime", "stage", "loss", "scheduler"},
    )
    runtime = build_train_runtime_config(
        require_mapping(payload["runtime"], "train.runtime")
    )
    stage = build_train_stage_config(
        require_mapping(payload["stage"], "train.stage")
    )
    loss = build_train_loss_config(
        require_mapping(payload["loss"], "train.loss")
    )
    scheduler = build_train_scheduler_config(
        require_mapping(payload["scheduler"], "train.scheduler")
    )
    return TrainConfig(
        runtime=runtime,
        stage=stage,
        loss=loss,
        scheduler=scheduler,
    )


def build_train_runtime_config(payload: Mapping[str, Any]) -> TrainRuntimeConfig:
    defaults = TrainRuntimeConfig(
        batch_size=1,
        lr=5e-4,
        dry_run=True,
        n_iter=1,
        log_every=1,
    )
    expect_keys(
        payload,
        "train.runtime",
        {
            "batch_size",
            "lr",
            "dry_run",
            "n_iter",
            "log_every",
        },
        optional={
            "checkpoint_every",
            "resume_from",
        },
    )
    return TrainRuntimeConfig(
        batch_size=read_int(payload, "batch_size", "train.runtime"),
        lr=read_float(payload, "lr", "train.runtime"),
        dry_run=read_bool(payload, "dry_run", "train.runtime"),
        n_iter=read_int(payload, "n_iter", "train.runtime"),
        log_every=read_int(payload, "log_every", "train.runtime"),
        checkpoint_every=read_optional_int(
            payload,
            "checkpoint_every",
            "train.runtime",
            default=defaults.checkpoint_every,
        ),
        resume_from=read_optional_str(
            payload,
            "resume_from",
            "train.runtime",
            default=None,
        ),
    )


def build_train_stage_config(payload: Mapping[str, Any]) -> TrainStageConfig:
    defaults = TrainStageConfig(n_init_iter=0)
    expect_keys(
        payload,
        "train.stage",
        {"n_init_iter"},
        optional={"refresh_rate_init", "refresh_rate_local"},
    )
    return TrainStageConfig(
        n_init_iter=read_int(payload, "n_init_iter", "train.stage"),
        refresh_rate_init=read_optional_int(
            payload,
            "refresh_rate_init",
            "train.stage",
            default=defaults.refresh_rate_init,
        ),
        refresh_rate_local=read_optional_int(
            payload,
            "refresh_rate_local",
            "train.stage",
            default=defaults.refresh_rate_local,
        ),
    )


def build_train_loss_config(payload: Mapping[str, Any]) -> TrainLossConfig:
    defaults = TrainLossConfig()
    expect_keys(
        payload,
        "train.loss",
        set(),
        optional={
            "loss_type",
            "n_loss_crops",
            "overflow_weight",
            "init_height_weight",
        },
    )
    loss_type = read_optional_str(
        payload,
        "loss_type",
        "train.loss",
        default=defaults.loss_type,
    ).upper()
    return TrainLossConfig(
        loss_type=loss_type,
        n_loss_crops=read_optional_int(
            payload,
            "n_loss_crops",
            "train.loss",
            default=defaults.n_loss_crops,
        ),
        overflow_weight=read_optional_float(
            payload,
            "overflow_weight",
            "train.loss",
            default=defaults.overflow_weight,
        ),
        init_height_weight=read_optional_float(
            payload,
            "init_height_weight",
            "train.loss",
            default=defaults.init_height_weight,
        ),
    )


def build_train_scheduler_config(payload: Mapping[str, Any]) -> TrainSchedulerConfig:
    defaults = TrainSchedulerConfig()
    expect_keys(
        payload,
        "train.scheduler",
        set(),
        optional={
            "eval_every",
            "scheduler_factor",
            "scheduler_patience_evals",
            "scheduler_min_lr",
        },
    )
    return TrainSchedulerConfig(
        eval_every=read_optional_int(
            payload,
            "eval_every",
            "train.scheduler",
            default=defaults.eval_every,
        ),
        scheduler_factor=read_optional_float(
            payload,
            "scheduler_factor",
            "train.scheduler",
            default=defaults.scheduler_factor,
        ),
        scheduler_patience_evals=read_optional_int(
            payload,
            "scheduler_patience_evals",
            "train.scheduler",
            default=defaults.scheduler_patience_evals,
        ),
        scheduler_min_lr=read_optional_float(
            payload,
            "scheduler_min_lr",
            "train.scheduler",
            default=defaults.scheduler_min_lr,
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
