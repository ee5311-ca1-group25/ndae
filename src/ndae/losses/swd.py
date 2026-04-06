"""Gram-based texture statistics losses."""

from __future__ import annotations

import torch

from .perceptual import VGG19Features


def gram_matrix(f: torch.Tensor) -> torch.Tensor:
    """Compute a Gram matrix normalized by the number of spatial samples."""
    if f.dim() == 3:
        f_flat = f.reshape(f.shape[0], -1)
        return (f_flat @ f_flat.transpose(0, 1)) / f_flat.shape[-1]
    if f.dim() == 4:
        f_flat = f.reshape(f.shape[0], f.shape[1], -1)
        return (f_flat @ f_flat.transpose(-2, -1)) / f_flat.shape[-1]
    raise ValueError(
        f"gram_matrix expects input shaped (C, H, W) or (B, C, H, W), got {tuple(f.shape)}."
    )


def gram_loss(
    features: VGG19Features,
    exemplar: torch.Tensor,
    sample: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute the summed per-layer Gram-matrix MSE across VGG features."""
    del generator

    exemplar_features = features(exemplar)
    sample_features = features(sample)

    loss = torch.zeros((), device=sample.device, dtype=sample.dtype)
    for exemplar_feature, sample_feature in zip(exemplar_features, sample_features):
        loss = loss + torch.mean(
            (gram_matrix(exemplar_feature) - gram_matrix(sample_feature)) ** 2
        )

    return loss
