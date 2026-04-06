"""Rendering metadata registry for NDAE."""

from __future__ import annotations

from dataclasses import dataclass

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
    "RENDERER_REGISTRY",
    "RendererSpec",
    "clip_maps",
    "i2l",
    "l2i",
    "select_renderer",
    "split_latent_maps",
]
