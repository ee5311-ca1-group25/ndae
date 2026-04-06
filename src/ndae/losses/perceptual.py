"""Perceptual feature extractors used by texture-statistics losses."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19


_BLOCK_RANGES: tuple[tuple[int, int], ...] = (
    (0, 4),
    (5, 9),
    (10, 18),
    (19, 27),
    (28, 36),
)


class VGG19Features(nn.Module):
    """Frozen VGG-19 feature extractor with AvgPool block boundaries."""

    def __init__(self) -> None:
        super().__init__()

        feature_layers = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[feature_layers[layer_index] for layer_index in range(start, stop)]
                )
                for start, stop in _BLOCK_RANGES
            ]
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

        self.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the input image plus 5 pre-pooling VGG block activations."""
        if x.dim() != 4:
            raise ValueError(
                f"VGG19Features expects input shaped (B, 3, H, W), got {tuple(x.shape)}."
            )
        if x.shape[1] != 3:
            raise ValueError(
                f"VGG19Features expects a 3-channel input, got {x.shape[1]} channels."
            )

        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        current = (x - mean) / std

        features = [x]
        for block_index, block in enumerate(self.blocks):
            current = block(current)
            features.append(current)
            if block_index < len(self.blocks) - 1:
                current = self.pool(current)

        return features
