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
CYCLE_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
}


def load_metrics(path: str | Path) -> list[dict[str, float | int]]:
    """Load step metrics from a JSONL file."""
    metrics_path = Path(path)
    records: list[dict[str, float | int]] = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            record = json.loads(payload)
            cycle_step = record.get("cycle_step")
            records.append(
                {
                    "global_step": float(record["global_step"]),
                    "cycle_step": int(cycle_step) if cycle_step is not None else -1,
                    **{key: float(record[key]) for key in LOSS_KEYS},
                }
            )
    if not records:
        raise ValueError(f"No metrics records found in {metrics_path}")
    return records


def cycle_average_records(
    records: list[dict[str, float | int]],
    *,
    refresh_rate: int,
) -> list[dict[str, float]]:
    """Average losses over refresh-sized global-step buckets."""
    grouped: dict[int, list[dict[str, float | int]]] = {}
    for record in records:
        cycle_index = int((float(record["global_step"]) - 1) // refresh_rate)
        grouped.setdefault(cycle_index, []).append(record)

    averaged: list[dict[str, float]] = []
    for cycle_index in sorted(grouped):
        bucket = grouped[cycle_index]
        averaged.append(
            {
                "cycle_index": float(cycle_index),
                **{
                    key: sum(float(record[key]) for record in bucket) / len(bucket)
                    for key in LOSS_KEYS
                },
            }
        )
    return averaged


def plot_metrics(
    records: list[dict[str, float | int]],
    output_path: str | Path,
    *,
    refresh_rate: int = 6,
) -> Path:
    """Plot raw losses, cycle-step grouped losses, and cycle averages to one PNG."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    steps = [record["global_step"] for record in records]
    cycle_averages = cycle_average_records(records, refresh_rate=refresh_rate)

    figure, axes = plt.subplots(3, 1, figsize=(10, 12))
    raw_axis, grouped_axis, average_axis = axes

    for key in LOSS_KEYS:
        raw_axis.plot(steps, [record[key] for record in records], label=key, linewidth=1.8)
    raw_axis.set_title("NDAE Loss Curves")
    raw_axis.set_xlabel("global_step")
    raw_axis.set_ylabel("loss")
    raw_axis.grid(True, alpha=0.25)
    raw_axis.legend()

    cycle_steps = sorted(
        {
            int(record["cycle_step"])
            for record in records
            if int(record["cycle_step"]) >= 0
        }
    )
    for cycle_step in cycle_steps:
        xs = [record["global_step"] for record in records if int(record["cycle_step"]) == cycle_step]
        ys = [record["loss_total"] for record in records if int(record["cycle_step"]) == cycle_step]
        grouped_axis.plot(
            xs,
            ys,
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"cycle_step={cycle_step}",
            color=CYCLE_COLORS.get(cycle_step),
        )
    grouped_axis.set_title("Loss Total Grouped by cycle_step")
    grouped_axis.set_xlabel("global_step")
    grouped_axis.set_ylabel("loss_total")
    grouped_axis.grid(True, alpha=0.25)
    if cycle_steps:
        grouped_axis.legend(ncol=3)

    cycle_indices = [record["cycle_index"] for record in cycle_averages]
    for key in LOSS_KEYS:
        average_axis.plot(
            cycle_indices,
            [record[key] for record in cycle_averages],
            label=key,
            linewidth=1.8,
        )
    average_axis.set_title(f"Cycle-Average Loss Curves (refresh_rate={refresh_rate})")
    average_axis.set_xlabel("cycle_index")
    average_axis.set_ylabel("mean loss")
    average_axis.grid(True, alpha=0.25)
    average_axis.legend()

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
    parser.add_argument(
        "--refresh-rate",
        type=int,
        default=6,
        help="Refresh cycle length used for cycle-average grouping.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the metrics plotting CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    records = load_metrics(args.metrics)
    output_path = plot_metrics(records, args.output, refresh_rate=args.refresh_rate)
    print("Rendered loss plot.")
    print(f"metrics: {args.metrics}")
    print(f"output: {output_path}")
    print(f"refresh_rate: {args.refresh_rate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
