"""Lecture 8 training CLI."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import torch

from ndae.config import NDAEConfig, load_config
from ndae.data import ExemplarDataset, Timeline
from ndae.losses import VGG19Features
from ndae.training import (
    RefreshSchedule,
    StageConfig,
    Trainer,
    load_resume_checkpoint,
    save_checkpoint,
)
from ndae.utils import create_workspace, format_run_summary, save_resolved_config

from ._svbrdf_system import build_svbrdf_system


def build_argparser() -> argparse.ArgumentParser:
    """Build the Lecture 8 train CLI parser."""
    parser = argparse.ArgumentParser(
        description="NDAE Lecture 8 train entry point.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for experiment.output_root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run mode regardless of config value.",
    )
    return parser


def run_train_cli(argv: Sequence[str] | None = None) -> int:
    """Run the Lecture 8 train entry point."""
    parser = build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    config = apply_overrides(
        config,
        output_root=args.output_root,
        force_dry_run=args.dry_run,
    )

    workspace = create_workspace(config)
    save_resolved_config(config, workspace)
    print(format_run_summary(config, workspace))

    if config.train.dry_run:
        print("Dry run completed.")
        return 0

    trainer = build_trainer(config, workspace)
    if config.train.resume_from is not None:
        load_resume_checkpoint(Path(config.train.resume_from), trainer)
        print(f"Resumed from checkpoint: {Path(config.train.resume_from).resolve()}")
    trainer.run(make_checkpoint_callback(config, workspace))
    print("Training completed.")
    return 0


def apply_overrides(
    config: NDAEConfig,
    *,
    output_root: str | None,
    force_dry_run: bool,
) -> NDAEConfig:
    """Apply CLI overrides to the config tree."""
    experiment = config.experiment
    train = config.train

    if output_root is not None:
        experiment = replace(experiment, output_root=output_root)
    if force_dry_run:
        train = replace(train, dry_run=True)

    if experiment is config.experiment and train is config.train:
        return config

    return replace(config, experiment=experiment, train=train)


def build_trainer(config: NDAEConfig, workspace: Path) -> Trainer:
    """Assemble the minimal Lecture 8 trainer from repo-level components."""
    generator = torch.Generator().manual_seed(config.experiment.seed)
    system = build_svbrdf_system(config)
    dataset = ExemplarDataset.from_config(config.data, base_dir=Path.cwd())
    timeline = Timeline.from_config(config.data)
    stage_config = StageConfig(
        t_init=config.data.t_I,
        t_start=config.data.t_S,
        t_end=config.data.t_E,
    )
    schedule = RefreshSchedule(stage_config, generator=generator)
    vgg_features = VGG19Features()

    def optimizer_factory() -> torch.optim.Optimizer:
        return torch.optim.Adam(system.trajectory_model.parameters(), lr=config.train.lr)

    return Trainer(
        trajectory_model=system.trajectory_model,
        optimizer_factory=optimizer_factory,
        schedule=schedule,
        stage_config=stage_config,
        solver_config=system.solver_config,
        exemplar_frames=dataset.frames,
        timeline=timeline,
        crop_size=config.data.crop_size,
        batch_size=config.train.batch_size,
        workspace=workspace,
        camera=system.camera,
        flash_light=system.flash_light,
        renderer_pp=system.renderer_pp,
        unpack_fn=system.unpack_fn,
        vgg_features=vgg_features,
        n_iter=config.train.n_iter,
        n_init_iter=config.train.n_init_iter,
        log_every=config.train.log_every,
        total_channels=system.total_channels,
        n_brdf_channels=system.n_brdf_channels,
        n_normal_channels=system.n_normal_channels,
        height_scale=system.height_scale,
        gamma=system.gamma,
        generator=generator,
    )


def make_checkpoint_callback(
    config: NDAEConfig,
    workspace: Path,
) -> Callable[[Trainer], None]:
    def callback(trainer: Trainer) -> None:
        if trainer.state.global_step % config.train.checkpoint_every != 0:
            return
        if trainer.state.cycle_step != 0:
            return
        save_checkpoint(workspace, trainer)

    return callback
