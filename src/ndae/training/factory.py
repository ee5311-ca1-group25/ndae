"""Factory helpers for assembling Lecture 8 training runtime."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ndae.config import NDAEConfig
from ndae.data import ExemplarDataset, Timeline
from ndae.losses import VGG19Features

from .schedule import RefreshSchedule, StageConfig
from .system import build_svbrdf_system
from .trainer import Trainer, TrainerComponents, TrainerConfig


def build_trainer(
    config: NDAEConfig,
    workspace: Path,
    *,
    dataset_base_dir: Path | None = None,
    vgg_features: nn.Module | None = None,
    generator: torch.Generator | None = None,
) -> Trainer:
    """Build a Trainer from repo-level config and runtime defaults."""
    generator = generator or torch.Generator().manual_seed(config.experiment.seed)
    system = build_svbrdf_system(config)
    dataset = ExemplarDataset.from_config(
        config.data,
        base_dir=dataset_base_dir or Path.cwd(),
    )
    timeline = Timeline.from_config(config.data)
    stage_config = StageConfig(
        t_init=config.data.t_I,
        t_start=config.data.t_S,
        t_end=config.data.t_E,
    )
    schedule = RefreshSchedule(stage_config, generator=generator)
    vgg_features = vgg_features or VGG19Features()

    def optimizer_factory() -> torch.optim.Optimizer:
        return torch.optim.Adam(system.trajectory_model.parameters(), lr=config.train.lr)

    return Trainer(
        components=TrainerComponents(
            system=system,
            optimizer_factory=optimizer_factory,
            schedule=schedule,
            stage_config=stage_config,
            vgg_features=vgg_features,
        ),
        config=TrainerConfig(
            exemplar_frames=dataset.frames,
            timeline=timeline,
            crop_size=config.data.crop_size,
            batch_size=config.train.batch_size,
            workspace=workspace,
            n_iter=config.train.n_iter,
            n_init_iter=config.train.n_init_iter,
            log_every=config.train.log_every,
            generator=generator,
        ),
    )


__all__ = ["build_trainer"]
