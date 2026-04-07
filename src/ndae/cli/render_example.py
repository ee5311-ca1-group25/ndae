"""Render a synthetic svBRDF example image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from ndae.utils import save_png_image
from ndae.rendering import (
    Camera,
    FlashLight,
    clip_maps,
    diffuse_cook_torrance,
    height_to_normal,
    render_svbrdf,
    tonemapping,
    unpack_brdf_diffuse_cook_torrance,
)

PRESET_NAMES: tuple[str, str] = ("plastic", "coated_metal")


def build_argparser() -> argparse.ArgumentParser:
    """Build the synthetic svBRDF example parser."""
    parser = argparse.ArgumentParser(
        description="Render a synthetic svBRDF example image.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to outputs/render_example/<preset>.png.",
    )
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default="plastic",
        help="Synthetic material preset to render.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square output resolution in pixels.",
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=5.0,
        help="Scale used by height_to_normal when converting the synthetic height map.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="Gamma used by tone mapping.",
    )
    parser.add_argument(
        "--camera-fov",
        type=float,
        default=50.0,
        help="Camera field of view in degrees.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=1.0,
        help="Camera distance from the surface plane.",
    )
    parser.add_argument(
        "--light-intensity",
        type=float,
        default=0.0,
        help="Log light intensity used by FlashLight.",
    )
    parser.add_argument(
        "--light-x",
        type=float,
        default=0.2,
        help="Flash light x offset.",
    )
    parser.add_argument(
        "--light-y",
        type=float,
        default=-0.2,
        help="Flash light y offset.",
    )
    return parser


def run_render_example_cli(argv: Sequence[str] | None = None) -> int:
    """Render a synthetic svBRDF image and save it as PNG."""
    parser = build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.image_size <= 0:
        parser.error("--image-size must be greater than 0")

    output_path = resolve_output_path(args.output, preset=args.preset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    brdf_maps, height_map = build_example_svbrdf_maps(args.image_size, preset=args.preset)
    normal_map = height_to_normal(height_map, scale=args.height_scale)
    camera = Camera(fov=args.camera_fov, distance=args.camera_distance)
    flash_light = FlashLight(
        intensity=args.light_intensity,
        xy_position=(args.light_x, args.light_y),
    )

    rendered = render_svbrdf(
        clip_maps(brdf_maps),
        normal_map,
        camera,
        flash_light,
        diffuse_cook_torrance,
        unpack_brdf_diffuse_cook_torrance,
    )
    tone_mapped = tonemapping(rendered, gamma=args.gamma)
    save_image(output_path, tone_mapped)

    print("Rendered synthetic svBRDF example.")
    print(f"preset: {args.preset}")
    print(f"output: {output_path.resolve()}")
    print(f"image_size: {args.image_size}")
    return 0


def resolve_output_path(output: str | None, *, preset: str) -> Path:
    """Resolve the output path for the selected preset."""
    if output is not None:
        return Path(output)
    return Path("outputs") / "render_example" / f"{preset}.png"


def build_example_svbrdf_maps(
    image_size: int,
    *,
    preset: str = "plastic",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic synthetic svBRDF and height map."""
    coords = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    radial = torch.sqrt(xx.square() + yy.square()).clamp(max=1.0)
    checker = build_checkerboard(xx, yy)

    if preset == "plastic":
        diffuse, specular, alpha_u, alpha_v, height_map = build_plastic_preset(xx, yy, radial, checker)
    elif preset == "coated_metal":
        diffuse, specular, alpha_u, alpha_v, height_map = build_coated_metal_preset(
            xx,
            yy,
            radial,
            checker,
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")

    brdf_maps = torch.cat([diffuse, specular, alpha_u, alpha_v], dim=0)
    return brdf_maps, height_map


def build_checkerboard(xx: torch.Tensor, yy: torch.Tensor, *, frequency: float = 3.0) -> torch.Tensor:
    """Build a tiled checkerboard mask in [0, 1]."""
    return ((torch.sin(frequency * torch.pi * xx) > 0.0) ^ (torch.sin(frequency * torch.pi * yy) > 0.0)).to(
        torch.float32
    )


def build_plastic_preset(
    xx: torch.Tensor,
    yy: torch.Tensor,
    radial: torch.Tensor,
    checker: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a smooth painted-plastic material with a broad isotropic highlight."""
    diffuse = torch.stack(
        [
            0.42 + 0.20 * checker + 0.06 * torch.exp(-2.5 * radial.square()),
            0.10 + 0.10 * (1.0 - checker) + 0.04 * (1.0 - radial),
            0.08 + 0.06 * checker + 0.03 * torch.exp(-3.0 * radial.square()),
        ],
        dim=0,
    ).clamp(0.03, 1.0)

    specular_base = (0.045 + 0.01 * torch.exp(-3.0 * radial.square())).clamp(0.02, 0.10)
    specular = torch.stack(
        [
            specular_base,
            specular_base,
            specular_base,
        ],
        dim=0,
    )

    roughness = (0.24 + 0.04 * torch.exp(-4.0 * radial.square())).clamp(0.18, 0.34)
    alpha_u = roughness.unsqueeze(0)
    alpha_v = roughness.unsqueeze(0)

    height_map = (
        0.12 * torch.exp(-5.0 * radial.square())
        + 0.01 * torch.sin(10.0 * xx) * torch.sin(10.0 * yy)
    ).unsqueeze(0)

    return diffuse, specular, alpha_u, alpha_v, height_map


def build_coated_metal_preset(
    xx: torch.Tensor,
    yy: torch.Tensor,
    radial: torch.Tensor,
    checker: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a dark coated-metal material with a tighter isotropic highlight."""
    diffuse = torch.stack(
        [
            0.05 + 0.05 * checker + 0.015 * torch.exp(-2.0 * radial.square()),
            0.06 + 0.06 * (1.0 - checker) + 0.02 * torch.exp(-2.5 * radial.square()),
            0.08 + 0.05 * checker + 0.02 * torch.exp(-3.0 * radial.square()),
        ],
        dim=0,
    ).clamp(0.02, 0.20)

    specular = torch.stack(
        [
            0.58 + 0.06 * torch.exp(-3.0 * radial.square()),
            0.60 + 0.05 * torch.exp(-3.5 * radial.square()),
            0.64 + 0.04 * torch.exp(-4.0 * radial.square()),
        ],
        dim=0,
    ).clamp(0.25, 0.90)

    roughness = (0.11 + 0.025 * (1.0 - torch.exp(-3.0 * radial.square()))).clamp(0.10, 0.16)
    alpha_u = roughness.unsqueeze(0)
    alpha_v = roughness.unsqueeze(0)

    height_map = (
        0.05 * torch.exp(-6.0 * radial.square())
        + 0.004 * torch.sin(18.0 * xx)
        + 0.003 * torch.cos(16.0 * yy)
    ).unsqueeze(0)

    return diffuse, specular, alpha_u, alpha_v, height_map


def save_image(path: Path, image: torch.Tensor) -> None:
    """Save a CHW tensor as an RGB PNG."""
    save_png_image(path, image)


__all__ = [
    "PRESET_NAMES",
    "build_argparser",
    "build_checkerboard",
    "build_coated_metal_preset",
    "build_example_svbrdf_maps",
    "build_plastic_preset",
    "resolve_output_path",
    "run_render_example_cli",
    "save_image",
]
