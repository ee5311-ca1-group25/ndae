"""Workspace helpers for NDAE train and debug runs."""

from __future__ import annotations

from pathlib import Path

import yaml

from ndae.config import NDAEConfig, to_dict


def create_workspace(config: NDAEConfig) -> Path:
    """Create the experiment workspace directory and return its path."""
    workspace = Path(config.experiment.output_root) / config.experiment.name
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def save_resolved_config(config: NDAEConfig, workspace: Path) -> Path:
    """Persist the resolved config under the workspace directory."""
    output_path = workspace / "config.resolved.yaml"
    output_path.write_text(
        yaml.safe_dump(to_dict(config), sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def format_run_summary(config: NDAEConfig, workspace: Path) -> str:
    """Return a concise run summary for the current config."""
    return "\n".join(
        [
            "NDAE Train Run Summary",
            f"experiment.name: {config.experiment.name}",
            f"experiment.seed: {config.experiment.seed}",
            f"data.root: {config.data.root}",
            f"data.exemplar: {config.data.exemplar}",
            f"data.image_size: {config.data.image_size}",
            f"data.crop_size: {config.data.crop_size}",
            f"data.n_frames: {config.data.n_frames}",
            f"data.t_I: {config.data.t_I}",
            f"data.t_S: {config.data.t_S}",
            f"data.t_E: {config.data.t_E}",
            f"model.dim: {config.model.dim}",
            f"model.solver: {config.model.solver}",
            f"rendering.renderer_type: {config.rendering.renderer_type}",
            f"rendering.n_brdf_channels: {config.rendering.n_brdf_channels}",
            f"rendering.n_normal_channels: {config.rendering.n_normal_channels}",
            f"rendering.n_aug_channels: {config.rendering.n_aug_channels}",
            f"rendering.total_channels: {config.rendering.total_channels}",
            f"train.runtime.batch_size: {config.train.runtime.batch_size}",
            f"train.runtime.lr: {config.train.runtime.lr}",
            f"train.runtime.dry_run: {config.train.runtime.dry_run}",
            f"train.stage.n_init_iter: {config.train.stage.n_init_iter}",
            f"train.stage.refresh_rate_init: {config.train.stage.refresh_rate_init}",
            f"train.stage.refresh_rate_local: {config.train.stage.refresh_rate_local}",
            f"train.loss.loss_type: {config.train.loss.loss_type}",
            f"train.loss.n_loss_crops: {config.train.loss.n_loss_crops}",
            f"train.loss.overflow_weight: {config.train.loss.overflow_weight}",
            f"train.loss.init_height_weight: {config.train.loss.init_height_weight}",
            f"train.scheduler.eval_every: {config.train.scheduler.eval_every}",
            f"train.scheduler.scheduler_factor: {config.train.scheduler.scheduler_factor}",
            f"train.scheduler.scheduler_patience_evals: {config.train.scheduler.scheduler_patience_evals}",
            f"train.scheduler.scheduler_min_lr: {config.train.scheduler.scheduler_min_lr}",
            "checkpoint cadence: eval-driven",
            f"workspace: {workspace}",
        ]
    )
