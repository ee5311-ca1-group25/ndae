"""Image writing helpers for NDAE CLIs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_png_image(path: str | Path, image: torch.Tensor) -> Path:
    """Save a CHW image tensor as an RGB PNG."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    plt.imsave(output_path, rgb)
    return output_path


__all__ = ["save_png_image"]
