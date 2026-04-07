#!/usr/bin/env python3
"""Render NDAE training metrics JSONL into a static PNG loss plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


LOSS_KEYS = ("loss_total", "loss_init", "loss_local", "loss_overflow")


def load_metrics(path: str | Path) -> list[dict[str, float]]:
    """Load step metrics from a JSONL file."""
    metrics_path = Path(path)
    records: list[dict[str, float]] = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            record = json.loads(payload)
            records.append(
                {
                    "global_step": float(record["global_step"]),
                    **{key: float(record[key]) for key in LOSS_KEYS},
                }
            )
    if not records:
        raise ValueError(f"No metrics records found in {metrics_path}")
    return records


def plot_metrics(records: list[dict[str, float]], output_path: str | Path) -> Path:
    """Plot loss curves against global step and write a PNG."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    steps = [record["global_step"] for record in records]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for key in LOSS_KEYS:
        axis.plot(steps, [record[key] for record in records], label=key, linewidth=2)
    axis.set_title("NDAE Loss Curves")
    axis.set_xlabel("global_step")
    axis.set_ylabel("loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for metrics plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="PNG path for the rendered loss plot",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the metrics plotting CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    records = load_metrics(args.metrics)
    output_path = plot_metrics(records, args.output)
    print("Rendered loss plot.")
    print(f"metrics: {args.metrics}")
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
