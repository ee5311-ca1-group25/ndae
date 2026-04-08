"""Differentiable svBRDF renderer core for NDAE."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ndae.rendering.brdf import (
    cook_torrance,
    diffuse_cook_torrance,
    diffuse_iso_cook_torrance,
    distribution_ggx,
    fresnel_schlick,
    geometry_smith,
    lambertian,
    smith_g1_ggx,
    unpack_brdf_diffuse_cook_torrance,
    unpack_brdf_diffuse_iso_cook_torrance,
)
from ndae.rendering.geometry import (
    EPSILON,
    Camera,
    FlashLight,
    channelwise_normalize,
    compute_directions,
    create_meshgrid,
    localize,
    localize_wiwo,
    normalize,
)
from ndae.rendering.postprocess import light_decay, reinhard, tonemapping


def _ensure_image_batch(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_channels: int | None = None,
) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 3:
        batched = tensor.unsqueeze(0)
        squeeze = True
    elif tensor.ndim == 4:
        batched = tensor
        squeeze = False
    else:
        raise ValueError(f"{name} expects a tensor shaped [C, H, W] or [B, C, H, W]")

    if expected_channels is not None and batched.shape[1] != expected_channels:
        raise ValueError(f"{name} expects channel dimension {expected_channels}")
    return batched, squeeze


def render_svbrdf(
    brdf_maps: torch.Tensor,
    normal_map: torch.Tensor,
    camera: Camera,
    flash_light: FlashLight,
    renderer_pp: Callable,
    unpack_fn: Callable,
    *,
    full_height: int | None = None,
    full_width: int | None = None,
    region: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    """Render a linear-space svBRDF image from BRDF and normal maps."""
    brdf_batch, squeeze = _ensure_image_batch(brdf_maps, name="brdf_maps")
    normal_batch, normal_squeeze = _ensure_image_batch(
        normal_map,
        name="normal_map",
        expected_channels=3,
    )
    if normal_squeeze != squeeze:
        raise ValueError("brdf_maps and normal_map must both be batched or both be unbatched")
    if brdf_batch.shape[0] != normal_batch.shape[0]:
        raise ValueError("brdf_maps and normal_map must share the same batch size")

    batch_size, _, h, w = brdf_batch.shape
    if normal_batch.shape[-2:] != (h, w):
        raise ValueError("brdf_maps and normal_map must share the same spatial size")

    device = brdf_batch.device
    dtype = brdf_batch.dtype

    if region is None:
        positions = create_meshgrid(h, w, camera, device=device)
    else:
        if full_height is None or full_width is None:
            raise ValueError("full_height and full_width are required when region is provided")
        top, left, crop_h, crop_w = region
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError("region crop size must be greater than 0")
        if (crop_h, crop_w) != (h, w):
            raise ValueError("region crop size must match the input map size")
        if top < 0 or left < 0 or top + crop_h > full_height or left + crop_w > full_width:
            raise ValueError("region must lie inside the full image extent")
        positions = create_meshgrid(full_height, full_width, camera, device=device)[
            :, top : top + crop_h, left : left + crop_w
        ]

    positions = positions.to(dtype=dtype)
    wi, wo = compute_directions(positions, camera, flash_light)
    wi = wi.unsqueeze(0).expand(batch_size, -1, -1, -1)
    wo = wo.unsqueeze(0).expand(batch_size, -1, -1, -1)
    local_wi, local_wo = localize_wiwo(wi, wo, normal_batch)

    params = unpack_fn(brdf_batch)
    reflectance = renderer_pp(local_wi, local_wo, *params)

    light_pos = torch.tensor(
        [flash_light.xy_position[0], flash_light.xy_position[1], camera.distance],
        dtype=dtype,
        device=device,
    ).view(3, 1, 1)
    distance = (light_pos - positions).norm(dim=0, keepdim=True)
    light_intensity = torch.exp(_scalar_to_tensor(flash_light.intensity, dtype=dtype, device=device))
    irradiance = light_intensity * light_decay(distance)
    rendered = reflectance * irradiance.unsqueeze(0)

    invalid = (local_wi.narrow(-3, 2, 1) < 0.0) | (local_wo.narrow(-3, 2, 1) < 0.0)
    rendered = torch.where(invalid, torch.zeros_like(rendered), rendered)
    return rendered.squeeze(0) if squeeze else rendered


def _scalar_to_tensor(
    value: torch.Tensor | float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)

__all__ = [
    "EPSILON",
    "Camera",
    "FlashLight",
    "channelwise_normalize",
    "compute_directions",
    "cook_torrance",
    "create_meshgrid",
    "diffuse_cook_torrance",
    "diffuse_iso_cook_torrance",
    "distribution_ggx",
    "fresnel_schlick",
    "geometry_smith",
    "lambertian",
    "light_decay",
    "localize",
    "localize_wiwo",
    "normalize",
    "reinhard",
    "render_svbrdf",
    "smith_g1_ggx",
    "tonemapping",
    "unpack_brdf_diffuse_cook_torrance",
    "unpack_brdf_diffuse_iso_cook_torrance",
]
