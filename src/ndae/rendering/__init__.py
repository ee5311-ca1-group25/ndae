"""Public rendering API for NDAE."""

from __future__ import annotations

from dataclasses import dataclass

from .normal import height_to_normal
from .renderer import (
    EPSILON,
    Camera,
    FlashLight,
    channelwise_normalize,
    compute_directions,
    cook_torrance,
    create_meshgrid,
    diffuse_cook_torrance,
    diffuse_iso_cook_torrance,
    distribution_ggx,
    fresnel_schlick,
    geometry_smith,
    lambertian,
    light_decay,
    localize,
    localize_wiwo,
    normalize,
    reinhard,
    render_svbrdf,
    smith_g1_ggx,
    tonemapping,
    unpack_brdf_diffuse_cook_torrance,
    unpack_brdf_diffuse_iso_cook_torrance,
)
from .maps import clip_maps, i2l, l2i, split_latent_maps


@dataclass(frozen=True, slots=True)
class RendererSpec:
    renderer_type: str
    n_brdf_channels: int


RENDERER_REGISTRY: dict[str, RendererSpec] = {
    "diffuse_cook_torrance": RendererSpec(
        renderer_type="diffuse_cook_torrance",
        n_brdf_channels=8,
    ),
    "diffuse_iso_cook_torrance": RendererSpec(
        renderer_type="diffuse_iso_cook_torrance",
        n_brdf_channels=7,
    ),
    "cook_torrance": RendererSpec(
        renderer_type="cook_torrance",
        n_brdf_channels=5,
    ),
    "iso_cook_torrance": RendererSpec(
        renderer_type="iso_cook_torrance",
        n_brdf_channels=4,
    ),
    "compl_cook_torrance": RendererSpec(
        renderer_type="compl_cook_torrance",
        n_brdf_channels=6,
    ),
    "compl_iso_cook_torrance": RendererSpec(
        renderer_type="compl_iso_cook_torrance",
        n_brdf_channels=5,
    ),
}


def select_renderer(renderer_type: str) -> RendererSpec:
    """Return renderer metadata for the configured renderer type."""
    try:
        return RENDERER_REGISTRY[renderer_type]
    except KeyError as exc:
        raise ValueError(f"Unknown renderer_type: {renderer_type}") from exc


__all__ = [
    "EPSILON",
    "Camera",
    "FlashLight",
    "RENDERER_REGISTRY",
    "RendererSpec",
    "channelwise_normalize",
    "clip_maps",
    "compute_directions",
    "cook_torrance",
    "create_meshgrid",
    "diffuse_cook_torrance",
    "diffuse_iso_cook_torrance",
    "distribution_ggx",
    "fresnel_schlick",
    "geometry_smith",
    "height_to_normal",
    "i2l",
    "lambertian",
    "l2i",
    "light_decay",
    "localize",
    "localize_wiwo",
    "normalize",
    "reinhard",
    "render_svbrdf",
    "select_renderer",
    "smith_g1_ggx",
    "split_latent_maps",
    "tonemapping",
    "unpack_brdf_diffuse_cook_torrance",
    "unpack_brdf_diffuse_iso_cook_torrance",
]
