"""Latent map extraction helpers for the rendering pipeline."""

from __future__ import annotations

import torch


def l2i(x: torch.Tensor) -> torch.Tensor:
    """Map latent-space values from [-1, 1] into [0, 1]."""
    return x * 0.5 + 0.5


def i2l(x: torch.Tensor) -> torch.Tensor:
    """Map image-space values from [0, 1] into [-1, 1]."""
    return (x - 0.5) * 2.0


def split_latent_maps(
    z: torch.Tensor,
    n_brdf_channels: int,
    n_normal_channels: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split latent maps into BRDF maps and height maps using ...CHW layout."""
    if z.ndim < 3:
        raise ValueError("split_latent_maps expects a tensor shaped [..., C, H, W]")
    if n_brdf_channels <= 0:
        raise ValueError("n_brdf_channels must be greater than 0")
    if n_normal_channels <= 0:
        raise ValueError("n_normal_channels must be greater than 0")

    channel_dim = -3
    required_channels = n_brdf_channels + n_normal_channels
    if z.shape[channel_dim] < required_channels:
        raise ValueError("latent channels must be greater than or equal to n_brdf_channels + n_normal_channels")

    brdf_maps = l2i(z.narrow(channel_dim, 0, n_brdf_channels))
    height_map = z.narrow(channel_dim, n_brdf_channels, n_normal_channels)
    return brdf_maps, height_map


def clip_maps(maps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Clamp BRDF maps into the physically valid [eps, 1] range."""
    return maps.clamp(min=eps, max=1.0)
