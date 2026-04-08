"""Geometry helpers for the differentiable svBRDF renderer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


EPSILON: float = 1e-6


@dataclass(slots=True)
class Camera:
    fov: float = 50.0
    distance: float = 1.0


@dataclass(slots=True)
class FlashLight:
    intensity: torch.Tensor | float = 0.0
    xy_position: tuple[float, float] = (0.0, 0.0)


def normalize(v: torch.Tensor) -> torch.Tensor:
    """Normalize vectors along the last dimension."""
    return v / (v.norm(dim=-1, keepdim=True) + EPSILON)


def channelwise_normalize(m: torch.Tensor) -> torch.Tensor:
    """Normalize vector maps along the channel dimension."""
    return m / (m.norm(dim=-3, keepdim=True) + EPSILON)


def create_meshgrid(
    height: int,
    width: int,
    camera: Camera,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a physical-space position grid for an image plane at z=0."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be greater than 0")

    dtype = torch.float32
    half_width = camera.distance * math.tan(math.radians(camera.fov / 2.0))
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    aspect_ratio = width / height

    xx = xx * half_width
    yy = -yy / aspect_ratio * half_width
    zz = torch.zeros_like(xx)
    return torch.stack([xx, yy, zz], dim=0)


def compute_directions(
    positions: torch.Tensor,
    camera: Camera,
    flash_light: FlashLight,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-pixel incident and outgoing directions in world space."""
    dtype = positions.dtype
    device = positions.device
    light_pos = torch.tensor(
        [flash_light.xy_position[0], flash_light.xy_position[1], camera.distance],
        dtype=dtype,
        device=device,
    ).view(3, 1, 1)
    view_pos = torch.tensor([0.0, 0.0, camera.distance], dtype=dtype, device=device).view(3, 1, 1)
    wi = channelwise_normalize(light_pos - positions)
    wo = channelwise_normalize(view_pos - positions)
    return wi, wo


def _channel_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-3, keepdim=True)


def localize(vec: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Project world-space vectors into the local tangent frame of a normal map."""
    tangent_ref = torch.zeros_like(normal)
    tangent_ref.select(dim=-3, index=0).fill_(1.0)
    tangent = channelwise_normalize(tangent_ref - _channel_dot(tangent_ref, normal) * normal)
    bitangent = channelwise_normalize(torch.cross(normal, tangent, dim=-3))
    return torch.cat(
        [
            _channel_dot(vec, tangent),
            _channel_dot(vec, bitangent),
            _channel_dot(vec, normal),
        ],
        dim=-3,
    )


def localize_wiwo(
    wi: torch.Tensor,
    wo: torch.Tensor,
    normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project incident and outgoing directions into local tangent space."""
    return localize(wi, normal), localize(wo, normal)


__all__ = [
    "EPSILON",
    "Camera",
    "FlashLight",
    "channelwise_normalize",
    "compute_directions",
    "create_meshgrid",
    "localize",
    "localize_wiwo",
    "normalize",
]
