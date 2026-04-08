"""Configuration schema and loading helpers for NDAE."""

from .errors import ConfigError
from .loader import load_config, to_dict
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
from .validation import validate_config

__all__ = [
    "ConfigError",
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "NDAEConfig",
    "RenderingConfig",
    "TrainConfig",
    "TrainRuntimeConfig",
    "TrainStageConfig",
    "TrainLossConfig",
    "TrainSchedulerConfig",
    "load_config",
    "to_dict",
    "validate_config",
]
