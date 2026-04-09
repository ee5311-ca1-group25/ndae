"""Cook-Torrance BRDF terms and BRDF-map unpack helpers."""

from __future__ import annotations

import torch

from ndae.rendering.geometry import EPSILON, channelwise_normalize


def _channel_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-3, keepdim=True)


def lambertian(wi: torch.Tensor, diffuse: torch.Tensor) -> torch.Tensor:
    """Lambertian diffuse term with the cosine factor included."""
    cos_theta_i = wi.narrow(-3, 2, 1).clamp(min=0.0)
    return diffuse / torch.pi * cos_theta_i


def distribution_ggx(
    h: torch.Tensor,
    alpha_u: torch.Tensor,
    alpha_v: torch.Tensor,
) -> torch.Tensor:
    """Anisotropic GGX normal distribution function."""
    h_x = h.narrow(-3, 0, 1)
    h_y = h.narrow(-3, 1, 1)
    h_z = h.narrow(-3, 2, 1).clamp(min=0.0)
    alpha_uv = alpha_u * alpha_v
    denom = (
        torch.pi
        * alpha_uv
        * (((h_x / alpha_u) ** 2 + (h_y / alpha_v) ** 2 + h_z**2) ** 2)
        + EPSILON
    )
    return 1.0 / denom


def smith_g1_ggx(
    v: torch.Tensor,
    alpha_u: torch.Tensor,
    alpha_v: torch.Tensor,
) -> torch.Tensor:
    """Anisotropic Smith G1 masking-shadowing term."""
    v_x = v.narrow(-3, 0, 1)
    v_y = v.narrow(-3, 1, 1)
    v_z = v.narrow(-3, 2, 1).clamp(min=0.0)
    tan_theta_alpha_2 = ((alpha_u * v_x) ** 2 + (alpha_v * v_y) ** 2) / (v_z**2 + EPSILON)
    return 2.0 / (1.0 + torch.sqrt(1.0 + tan_theta_alpha_2))


def geometry_smith(
    wi: torch.Tensor,
    wo: torch.Tensor,
    alpha_u: torch.Tensor,
    alpha_v: torch.Tensor,
) -> torch.Tensor:
    """Separable Smith masking-shadowing term."""
    return smith_g1_ggx(wi, alpha_u, alpha_v) * smith_g1_ggx(wo, alpha_u, alpha_v)


def fresnel_schlick(cos_theta: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """Schlick Fresnel approximation."""
    return f0 + (1.0 - f0) * (1.0 - cos_theta).clamp(min=0.0, max=1.0).pow(5)


def cook_torrance(
    wi: torch.Tensor,
    wo: torch.Tensor,
    specular: torch.Tensor,
    alpha_u: torch.Tensor,
    alpha_v: torch.Tensor,
) -> torch.Tensor:
    """Cosine-weighted anisotropic Cook-Torrance specular term."""
    h = channelwise_normalize(wi + wo)
    d = distribution_ggx(h, alpha_u, alpha_v)
    g = geometry_smith(wi, wo, alpha_u, alpha_v)
    cos_theta = _channel_dot(h, wi).clamp(min=0.0, max=1.0)
    f = fresnel_schlick(cos_theta, specular)
    denom = 4.0 * wo.narrow(-3, 2, 1).clamp(min=EPSILON)
    return d * g * f / denom


def diffuse_cook_torrance(
    wi: torch.Tensor,
    wo: torch.Tensor,
    diffuse: torch.Tensor,
    specular: torch.Tensor,
    alpha_u: torch.Tensor,
    alpha_v: torch.Tensor,
) -> torch.Tensor:
    """Diffuse plus anisotropic Cook-Torrance specular."""
    return lambertian(wi, diffuse) + cook_torrance(wi, wo, specular, alpha_u, alpha_v)


def diffuse_iso_cook_torrance(
    wi: torch.Tensor,
    wo: torch.Tensor,
    diffuse: torch.Tensor,
    specular: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Diffuse plus isotropic Cook-Torrance specular."""
    return diffuse_cook_torrance(wi, wo, diffuse, specular, alpha, alpha)


def unpack_brdf_diffuse_cook_torrance(
    brdf_maps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack diffuse/specular/aniso-roughness parameters."""
    return (
        brdf_maps.narrow(-3, 0, 3),
        brdf_maps.narrow(-3, 3, 3),
        brdf_maps.narrow(-3, 6, 1),
        brdf_maps.narrow(-3, 7, 1),
    )


def unpack_brdf_diffuse_iso_cook_torrance(
    brdf_maps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack diffuse/specular/isotropic-roughness parameters."""
    return (
        brdf_maps.narrow(-3, 0, 3),
        brdf_maps.narrow(-3, 3, 3),
        brdf_maps.narrow(-3, 6, 1),
    )


__all__ = [
    "cook_torrance",
    "diffuse_cook_torrance",
    "diffuse_iso_cook_torrance",
    "distribution_ggx",
    "fresnel_schlick",
    "geometry_smith",
    "lambertian",
    "smith_g1_ggx",
    "unpack_brdf_diffuse_cook_torrance",
    "unpack_brdf_diffuse_iso_cook_torrance",
]
