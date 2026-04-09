"""Post-processing helpers for the differentiable svBRDF renderer."""

from __future__ import annotations

import torch

from ndae.rendering.geometry import EPSILON


def tonemapping(
    img: torch.Tensor,
    gamma: float = 2.2,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Convert linear HDR values into the [0, 1] sRGB range."""
    return img.clamp(min=eps, max=1.0).pow(1.0 / gamma)


def light_decay(distance: torch.Tensor) -> torch.Tensor:
    """Inverse-square distance falloff."""
    return 1.0 / (distance**2 + EPSILON)


def reinhard(img: torch.Tensor) -> torch.Tensor:
    """Alternative Reinhard tone mapping."""
    return img / (1.0 + img)


__all__ = ["light_decay", "reinhard", "tonemapping"]
