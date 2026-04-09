"""Checkpoint sampling CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ndae.config import load_config
from ndae.data import Timeline
from ndae.evaluation import sample_sequence
from ndae.training import (
    build_svbrdf_system,
    load_sample_checkpoint,
    render_latent_state,
    resolve_checkpoint_dir,
)
from ndae.utils import save_png_image


def build_argparser() -> argparse.ArgumentParser:
    """Build the sample CLI parser."""
    parser = argparse.ArgumentParser(
        description="NDAE checkpoint sampling entry point.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a concrete checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the PNG output directory.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        help="Output spatial resolution used for sample-time rollout.",
    )
    return parser


def run_sample_cli(argv: Sequence[str] | None = None) -> int:
    """Restore a checkpoint and export a PNG sequence."""
    parser = build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    workspace = checkpoint_dir.parent.parent
    config = load_config(
        workspace / "config.resolved.yaml",
        base_dir=workspace,
        validate_dataset=False,
    )
    sample_size = args.sample_size
    if sample_size <= 0:
        parser.error("--sample-size must be greater than 0")

    system = build_svbrdf_system(config)
    load_sample_checkpoint(checkpoint_dir, system.trajectory_model, system.flash_light)
    timeline = Timeline.from_config(config.data)
    states, synthesis_count = sample_sequence(
        system,
        timeline,
        sample_size=sample_size,
        seed=config.experiment.seed,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else workspace / "samples" / checkpoint_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, state in enumerate(states[synthesis_count:, 0]):
        save_png_image(output_dir / f"frames_{index:04d}.png", render_latent_state(system, state))

    print("Sample generation completed.")
    print(f"checkpoint: {checkpoint_dir}")
    print(f"output_dir: {output_dir.resolve()}")
    print(f"transition_frames: {config.data.n_frames}")
    print(f"synthesis_frames: {synthesis_count}")
    print(f"sample_size: {sample_size}")
    return 0


__all__ = ["build_argparser", "run_sample_cli"]
