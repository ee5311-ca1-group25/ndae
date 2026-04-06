"""Height-to-normal conversion helpers for the rendering pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_EPSILON = 1e-6


def height_to_normal(height: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Convert height maps shaped [..., 1, H, W] into world-space normal maps."""
    if height.ndim < 3:
        raise ValueError("height_to_normal expects a tensor shaped [..., 1, H, W]")
    if height.shape[-3] != 1:
        raise ValueError("height_to_normal expects a singleton channel dimension")

    leading_shape = height.shape[:-3]
    _, h, w = height.shape[-3:]

    scaled = height * scale
    flattened = scaled.reshape(-1, 1, h, w)
    padded = F.pad(flattened, (1, 1, 1, 1), mode="replicate")

    gx = (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2]) / 2.0
    gy = (padded[..., :-2, 1:-1] - padded[..., 2:, 1:-1]) / 2.0

    n_raw = torch.cat([-gx, -gy, torch.ones_like(gx)], dim=1)
    normal = n_raw / (n_raw.norm(dim=1, keepdim=True) + _EPSILON)
    return normal.reshape(*leading_shape, 3, h, w)
