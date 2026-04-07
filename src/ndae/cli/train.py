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
from ndae.models import NDAEUNet, ODEFunction, TrajectoryModel
from ndae.rendering import (
    Camera,
    FlashLight,
    diffuse_cook_torrance,
    diffuse_iso_cook_torrance,
    unpack_brdf_diffuse_cook_torrance,
    unpack_brdf_diffuse_iso_cook_torrance,
)
from ndae.training import RefreshSchedule, SolverConfig, StageConfig, Trainer
from ndae.utils import create_workspace, format_run_summary, save_resolved_config


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
    trainer.run()
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
    trajectory_model = build_trajectory_model(config)
    renderer_pp, unpack_fn = resolve_renderer_runtime(config)
    dataset = ExemplarDataset.from_config(config.data, base_dir=Path.cwd())
    timeline = Timeline.from_config(config.data)
    stage_config = StageConfig(
        t_init=config.data.t_I,
        t_start=config.data.t_S,
        t_end=config.data.t_E,
    )
    schedule = RefreshSchedule(stage_config, generator=generator)
    solver_config = SolverConfig(method=resolve_solver_method(config.model.solver))
    camera = Camera(
        fov=config.rendering.camera_fov,
        distance=config.rendering.camera_distance,
    )
    flash_light = FlashLight(
        intensity=config.rendering.light_intensity,
        xy_position=config.rendering.light_xy_position,
    )
    vgg_features = VGG19Features()

    def optimizer_factory() -> torch.optim.Optimizer:
        return torch.optim.Adam(trajectory_model.parameters(), lr=config.train.lr)

    return Trainer(
        trajectory_model=trajectory_model,
        optimizer_factory=optimizer_factory,
        schedule=schedule,
        stage_config=stage_config,
        solver_config=solver_config,
        exemplar_frames=dataset.frames,
        timeline=timeline,
        crop_size=config.data.crop_size,
        batch_size=config.train.batch_size,
        workspace=workspace,
        camera=camera,
        flash_light=flash_light,
        renderer_pp=renderer_pp,
        unpack_fn=unpack_fn,
        vgg_features=vgg_features,
        n_iter=config.train.n_iter,
        n_init_iter=config.train.n_init_iter,
        log_every=config.train.log_every,
        total_channels=config.rendering.total_channels,
        n_brdf_channels=config.rendering.n_brdf_channels,
        n_normal_channels=config.rendering.n_normal_channels,
        height_scale=config.rendering.height_scale,
        gamma=config.rendering.gamma,
        generator=generator,
    )


def build_trajectory_model(config: NDAEConfig) -> TrajectoryModel:
    total_channels = config.rendering.total_channels
    vector_field = NDAEUNet(
        in_dim=total_channels,
        out_dim=total_channels,
        dim=config.model.dim,
        dim_mults=(1, 2),
        use_attn=False,
    )
    return TrajectoryModel(ODEFunction(vector_field))


def resolve_solver_method(solver: str) -> str:
    if solver == "heun":
        return "adaptive_heun"
    if solver == "euler":
        return "euler"
    raise ValueError(
        "Unsupported model.solver for Lecture 8 trainer: "
        f"{solver!r}. Expected 'heun' or 'euler'."
    )


def resolve_renderer_runtime(
    config: NDAEConfig,
) -> tuple[Callable[..., torch.Tensor], Callable[..., tuple[torch.Tensor, ...]]]:
    renderer_type = config.rendering.renderer_type
    if renderer_type == "diffuse_cook_torrance":
        return diffuse_cook_torrance, unpack_brdf_diffuse_cook_torrance
    if renderer_type == "diffuse_iso_cook_torrance":
        return diffuse_iso_cook_torrance, unpack_brdf_diffuse_iso_cook_torrance
    raise ValueError(
        "Lecture 8 non-dry-run training only supports renderer_type "
        "'diffuse_cook_torrance' and 'diffuse_iso_cook_torrance', got "
        f"{renderer_type!r}."
    )
