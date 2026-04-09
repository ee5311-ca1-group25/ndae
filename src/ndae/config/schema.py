"""Dataclass schemas for NDAE configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    output_root: str
    seed: int


@dataclass(slots=True)
class DataConfig:
    root: str
    exemplar: str
    image_size: int
    crop_size: int
    n_frames: int
    t_S: float = 0.0
    t_E: float = 10.0

    @property
    def t_I(self) -> float:
        duration = self.t_E - self.t_S
        return self.t_S - 0.2 * duration


@dataclass(slots=True)
class ModelConfig:
    dim: int
    solver: str


@dataclass(slots=True)
class RenderingConfig:
    renderer_type: str = "diffuse_cook_torrance"
    n_brdf_channels: int = 8
    n_normal_channels: int = 1
    n_aug_channels: int = 9
    camera_fov: float = 50.0
    camera_distance: float = 1.0
    light_intensity: float = 0.0
    light_xy_position: tuple[float, float] = (0.0, 0.0)
    height_scale: float = 1.0
    gamma: float = 2.2

    @property
    def total_channels(self) -> int:
        return self.n_brdf_channels + self.n_normal_channels + self.n_aug_channels


@dataclass(slots=True)
class TrainRuntimeConfig:
    batch_size: int
    lr: float
    dry_run: bool
    n_iter: int
    log_every: int
    checkpoint_every: int = 1
    resume_from: str | None = None


@dataclass(slots=True)
class TrainStageConfig:
    n_init_iter: int
    refresh_rate_init: int = 2
    refresh_rate_local: int = 6


@dataclass(slots=True)
class TrainLossConfig:
    loss_type: str = "SW"
    n_loss_crops: int = 32
    overflow_weight: float = 100.0
    init_height_weight: float = 1.0


@dataclass(slots=True)
class TrainSchedulerConfig:
    eval_every: int = 500
    scheduler_factor: float = 0.5
    scheduler_patience_evals: int = 5
    scheduler_min_lr: float = 1e-4


@dataclass(slots=True)
class TrainConfig:
    runtime: TrainRuntimeConfig
    stage: TrainStageConfig
    loss: TrainLossConfig
    scheduler: TrainSchedulerConfig


@dataclass(slots=True)
class NDAEConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    rendering: RenderingConfig
    train: TrainConfig
