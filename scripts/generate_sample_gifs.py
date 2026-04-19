#!/usr/bin/env python3
"""Generate GIF previews from sampled frame folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


DEFAULT_SAMPLE_SUBDIRS: tuple[str, str] = (
    "step_010000",
    "step_010000_relighted_i0p6_x0p35_y-0p35",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GIF files from sampled frame folders.",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("outputs/full_clay_solidifying_official_like/samples"),
        help="Root directory containing sampled frame subfolders.",
    )
    parser.add_argument(
        "--subdir",
        action="append",
        default=None,
        help=(
            "Sample subdirectory name under --samples-root. "
            "Repeat this flag to render multiple GIFs. "
            "Defaults to the two standard full-run folders."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="frames_*.png",
        help="Glob pattern for frame files inside each sample subdirectory.",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=70,
        help="Frame duration in milliseconds for the generated GIF.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF loop count passed to PIL (0 means infinite).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GIF files.",
    )
    return parser


def collect_frames(sample_dir: Path, pattern: str) -> list[Path]:
    frames = sorted(sample_dir.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No frames matched pattern {pattern!r} in {sample_dir}")
    return frames


def write_gif(
    *,
    frames: list[Path],
    output_path: Path,
    duration_ms: int,
    loop: int,
    overwrite: bool,
) -> bool:
    if output_path.exists() and not overwrite:
        return False

    images = [Image.open(frame_path).convert("RGB") for frame_path in frames]
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=loop,
            optimize=True,
            disposal=2,
        )
    finally:
        for image in images:
            image.close()

    return True


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.duration_ms <= 0:
        parser.error("--duration-ms must be greater than 0")
    if args.loop < 0:
        parser.error("--loop must be greater than or equal to 0")

    samples_root = args.samples_root.expanduser().resolve()
    subdirs = args.subdir if args.subdir is not None else list(DEFAULT_SAMPLE_SUBDIRS)

    print(f"samples_root: {samples_root}")
    print(f"frame_pattern: {args.pattern}")
    print(f"duration_ms: {args.duration_ms}")
    print(f"loop: {args.loop}")

    generated_count = 0
    skipped_count = 0
    for subdir in subdirs:
        sample_dir = samples_root / subdir
        output_path = samples_root / f"{subdir}.gif"
        frames = collect_frames(sample_dir, args.pattern)
        generated = write_gif(
            frames=frames,
            output_path=output_path,
            duration_ms=args.duration_ms,
            loop=args.loop,
            overwrite=args.overwrite,
        )

        if generated:
            generated_count += 1
            print(f"generated: {output_path} ({len(frames)} frames)")
        else:
            skipped_count += 1
            print(f"skipped (exists): {output_path}")

    print(f"done: generated={generated_count}, skipped={skipped_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
