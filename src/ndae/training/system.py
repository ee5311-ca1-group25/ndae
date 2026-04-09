"""Shared svBRDF runtime assembly for training and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from ndae.config import NDAEConfig
from ndae.models import NDAEUNet, ODEFunction, TrajectoryModel
from ndae.rendering import (
    Camera,
    FlashLight,
    clip_maps,
    diffuse_cook_torrance,
    diffuse_iso_cook_torrance,
    height_to_normal,
    render_svbrdf,
    split_latent_maps,
    tonemapping,
    unpack_brdf_diffuse_cook_torrance,
    unpack_brdf_diffuse_iso_cook_torrance,
)

from .solver import SolverConfig


@dataclass(slots=True)
class SVBRDFSystem:
    """Shared runtime components for svBRDF train/sample entry points."""

    trajectory_model: TrajectoryModel
    solver_config: SolverConfig
    camera: Camera
    flash_light: FlashLight
    renderer_pp: Callable[..., torch.Tensor]
    unpack_fn: Callable[..., tuple[torch.Tensor, ...]]
    total_channels: int
    n_brdf_channels: int
    n_normal_channels: int
    height_scale: float
    gamma: float


def build_svbrdf_system(config: NDAEConfig) -> SVBRDFSystem:
    """Build the model, solver, and rendering runtime from config."""
    total_channels = config.rendering.total_channels
    trajectory_model = TrajectoryModel(
        ODEFunction(
            NDAEUNet(
                in_dim=total_channels,
                out_dim=total_channels,
                dim=config.model.dim,
                dim_mults=(1, 2),
                use_attn=False,
            )
        )
    )
    solver_config = SolverConfig(method=resolve_solver_method(config.model.solver))
    renderer_pp, unpack_fn = resolve_renderer_runtime(config)
    camera = Camera(
        fov=config.rendering.camera_fov,
        distance=config.rendering.camera_distance,
    )
    flash_light = FlashLight(
        intensity=nn.Parameter(torch.tensor(config.rendering.light_intensity, dtype=torch.float32)),
        xy_position=config.rendering.light_xy_position,
    )
    return SVBRDFSystem(
        trajectory_model=trajectory_model,
        solver_config=solver_config,
        camera=camera,
        flash_light=flash_light,
        renderer_pp=renderer_pp,
        unpack_fn=unpack_fn,
        total_channels=total_channels,
        n_brdf_channels=config.rendering.n_brdf_channels,
        n_normal_channels=config.rendering.n_normal_channels,
        height_scale=config.rendering.height_scale,
        gamma=config.rendering.gamma,
    )


def resolve_solver_method(solver: str) -> str:
    if solver == "heun":
        return "adaptive_heun"
    if solver == "euler":
        return "euler"
    raise ValueError(
        "Unsupported model.solver for the training runtime: "
        f"{solver!r}. Expected 'heun' or 'euler'."
    )


def resolve_renderer_runtime(
    config: NDAEConfig,
) -> tuple[Callable[..., torch.Tensor], Callable[..., tuple[torch.Tensor, ...]]]:
    renderer_type = config.rendering.renderer_type
    if renderer_type == "diffuse_cook_torrance":
        return diffuse_cook_torrance, unpack_brdf_diffuse_cook_torrance
    if renderer_type == "diffuse_iso_cook_torrance":
        return diffuse_iso_cook_torrance, unpack_brdf_diffuse_iso_cook_torrance
    raise ValueError(
        "Non-dry-run training only supports renderer_type "
        "'diffuse_cook_torrance' and 'diffuse_iso_cook_torrance', got "
        f"{renderer_type!r}."
    )


def render_latent_state(system: SVBRDFSystem, state: torch.Tensor) -> torch.Tensor:
    """Project one latent svBRDF state into a tone-mapped RGB image."""
    brdf_maps, height_map = split_latent_maps(
        state,
        n_brdf_channels=system.n_brdf_channels,
        n_normal_channels=system.n_normal_channels,
    )
    normal_map = height_to_normal(height_map, scale=system.height_scale)
    rendered = render_svbrdf(
        clip_maps(brdf_maps),
        normal_map,
        system.camera,
        system.flash_light,
        system.renderer_pp,
        system.unpack_fn,
    )
    return tonemapping(rendered, gamma=system.gamma)


__all__ = [
    "SVBRDFSystem",
    "build_svbrdf_system",
    "render_latent_state",
    "resolve_renderer_runtime",
    "resolve_solver_method",
]
