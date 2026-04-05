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
    n_aug_channels: int
    solver: str


@dataclass(slots=True)
class TrainConfig:
    batch_size: int
    lr: float
    dry_run: bool


@dataclass(slots=True)
class NDAEConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
