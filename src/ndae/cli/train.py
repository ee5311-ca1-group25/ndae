"""Training CLI."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from ndae.config import NDAEConfig, load_config
from ndae.losses import VGG19Features
from ndae.training import build_trainer, load_resume_checkpoint, save_checkpoint
from ndae.utils import create_workspace, format_run_summary, save_resolved_config


def build_argparser() -> argparse.ArgumentParser:
    """Build the train CLI parser."""
    parser = argparse.ArgumentParser(
        description="NDAE training entry point.",
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
    """Run the train entry point."""
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

    if config.train.runtime.dry_run:
        print("Dry run completed.")
        return 0

    trainer = build_trainer(
        config,
        workspace,
        vgg_features=VGG19Features(),
    )
    if config.train.runtime.resume_from is not None:
        load_resume_checkpoint(Path(config.train.runtime.resume_from), trainer)
        print(f"Resumed from checkpoint: {Path(config.train.runtime.resume_from).resolve()}")
    trainer.run(eval_callback=make_eval_checkpoint_callback(workspace))
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
        train = replace(
            train,
            runtime=replace(train.runtime, dry_run=True),
        )

    if experiment is config.experiment and train is config.train:
        return config

    return replace(config, experiment=experiment, train=train)


def make_eval_checkpoint_callback(workspace: Path):
    def callback(trainer, eval_metrics: dict[str, float | int | str]) -> None:
        del eval_metrics
        save_checkpoint(workspace, trainer, saved_during_eval=True)

    return callback
