"""Workspace helpers for Lecture 1 dry-run execution."""

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
            "NDAE Lecture 1 Dry Run",
            f"experiment.name: {config.experiment.name}",
            f"experiment.seed: {config.experiment.seed}",
            f"data.root: {config.data.root}",
            f"data.exemplar: {config.data.exemplar}",
            f"data.image_size: {config.data.image_size}",
            f"data.crop_size: {config.data.crop_size}",
            f"data.n_frames: {config.data.n_frames}",
            f"model.dim: {config.model.dim}",
            f"model.n_aug_channels: {config.model.n_aug_channels}",
            f"model.solver: {config.model.solver}",
            f"train.batch_size: {config.train.batch_size}",
            f"train.lr: {config.train.lr}",
            f"train.dry_run: {config.train.dry_run}",
            f"workspace: {workspace}",
        ]
    )
