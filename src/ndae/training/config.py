"""Trainer-side config dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..data import Timeline


@dataclass(slots=True)
class TrainerRuntimeConfig:
    """Runtime controls for one trainer instance."""

    batch_size: int
    workspace: Path
    n_iter: int
    log_every: int
    gamma: float = 2.2
    generator: torch.Generator | None = None
    device: torch.device | None = None


@dataclass(slots=True)
class TrainerStageConfig:
    """Stage controls mirrored from the repo config schema."""

    n_init_iter: int
    refresh_rate_init: int = 2
    refresh_rate_local: int = 6


@dataclass(slots=True)
class TrainerLossConfig:
    """Loss controls mirrored from the repo config schema."""

    loss_type: str = "SW"
    n_loss_crops: int = 32
    overflow_weight: float = 100.0
    init_height_weight: float = 1.0


@dataclass(slots=True)
class TrainerSchedulerConfig:
    """Scheduler/eval controls mirrored from the repo config schema."""

    eval_every: int = 500
    scheduler_factor: float = 0.5
    scheduler_patience_evals: int = 5
    scheduler_min_lr: float = 1e-4


@dataclass(slots=True)
class TrainerConfig:
    """Config and data inputs for one trainer instance."""

    exemplar_frames: torch.Tensor
    timeline: Timeline
    crop_size: int
    runtime: TrainerRuntimeConfig
    stage: TrainerStageConfig
    loss: TrainerLossConfig
    scheduler: TrainerSchedulerConfig


__all__ = [
    "TrainerConfig",
    "TrainerLossConfig",
    "TrainerRuntimeConfig",
    "TrainerSchedulerConfig",
    "TrainerStageConfig",
]
