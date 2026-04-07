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
    t_I: float = -2.0
    t_S: float = 0.0
    t_E: float = 10.0


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
class TrainConfig:
    batch_size: int
    lr: float
    dry_run: bool
    n_iter: int
    n_init_iter: int
    log_every: int
    checkpoint_every: int
    sample_every: int
    sample_size: int
    resume_from: str | None = None


@dataclass(slots=True)
class NDAEConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    rendering: RenderingConfig
    train: TrainConfig
