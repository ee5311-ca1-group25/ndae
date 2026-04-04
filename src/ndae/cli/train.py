"""Lecture 1 training CLI."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from ndae.config import NDAEConfig, load_config
from ndae.utils import create_workspace, format_run_summary, save_resolved_config


def build_argparser() -> argparse.ArgumentParser:
    """Build the Lecture 1 CLI parser."""
    parser = argparse.ArgumentParser(
        description="NDAE Lecture 1 dry-run entry point.",
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
    """Run the Lecture 1 dry-run entry point."""
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

    print("Non-dry-run mode is not implemented in Lecture 1.")
    return 2


def apply_overrides(
    config: NDAEConfig,
    *,
    output_root: str | None,
    force_dry_run: bool,
) -> NDAEConfig:
    """Apply simple Lecture 1 CLI overrides to the config tree."""
    experiment = config.experiment
    train = config.train

    if output_root is not None:
        experiment = replace(experiment, output_root=output_root)
    if force_dry_run:
        train = replace(train, dry_run=True)

    if experiment is config.experiment and train is config.train:
        return config

    return replace(config, experiment=experiment, train=train)
