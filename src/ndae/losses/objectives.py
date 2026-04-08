"""Wrapper objectives built on top of texture-statistics losses."""

from __future__ import annotations

import torch

from .perceptual import VGG19Features
from .swd import gram_loss, slice_loss


def overflow_loss(
    brdf_maps: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize BRDF values outside the valid [eps, 1] interval."""
    clipped = brdf_maps.clamp(min=eps, max=1.0)
    return ((brdf_maps - clipped) ** 2).mean()


def init_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute the Init-stage per-pixel MSE objective."""
    return ((rendered - target) ** 2).mean()


def local_loss(
    vgg: VGG19Features,
    rendered: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "SW",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Dispatch the Local-stage loss to SWD or Gram statistics."""
    normalized = loss_type.upper()
    if normalized == "SW":
        return slice_loss(vgg, target, rendered, generator=generator)
    if normalized == "GRAM":
        return gram_loss(vgg, target, rendered, generator=generator)
    raise ValueError(f"Unknown loss_type: {loss_type!r}. Must be 'SW' or 'GRAM'.")


__all__ = ["overflow_loss", "init_loss", "local_loss"]
