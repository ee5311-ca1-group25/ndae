"""Gram-based texture statistics losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

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


def sliced_wasserstein_loss(
    fe: torch.Tensor,
    fs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute sliced Wasserstein loss between one pair of feature tensors."""
    if fe.dim() not in (3, 4) or fs.dim() not in (3, 4):
        raise ValueError(
            "sliced_wasserstein_loss expects inputs shaped (C, H, W) or "
            f"(B, C, H, W), got {tuple(fe.shape)} and {tuple(fs.shape)}."
        )
    if fe.dim() != fs.dim():
        raise ValueError(
            "sliced_wasserstein_loss expects exemplar and sample to have the same rank, "
            f"got {fe.dim()} and {fs.dim()}."
        )
    if fe.shape[-3] != fs.shape[-3]:
        raise ValueError(
            "sliced_wasserstein_loss expects matching channel counts, "
            f"got {fe.shape[-3]} and {fs.shape[-3]}."
        )

    if fe.dim() == 3:
        exemplar_flat = fe.reshape(fe.shape[0], -1)
        sample_flat = fs.reshape(fs.shape[0], -1)
        channels = sample_flat.shape[0]
        directions = torch.randn(
            channels,
            channels,
            device=fs.device,
            dtype=fs.dtype,
            generator=generator,
        )
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        exemplar_proj = directions @ exemplar_flat
        sample_proj = directions @ sample_flat
    else:
        exemplar_flat = fe.reshape(fe.shape[0], fe.shape[1], -1)
        sample_flat = fs.reshape(fs.shape[0], fs.shape[1], -1)
        channels = sample_flat.shape[1]
        directions = torch.randn(
            channels,
            channels,
            device=fs.device,
            dtype=fs.dtype,
            generator=generator,
        )
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        exemplar_proj = torch.matmul(directions.unsqueeze(0), exemplar_flat)
        sample_proj = torch.matmul(directions.unsqueeze(0), sample_flat)

    exemplar_sorted = torch.sort(exemplar_proj, dim=-1).values
    sample_sorted = torch.sort(sample_proj, dim=-1).values

    if exemplar_sorted.shape[-1] != sample_sorted.shape[-1]:
        if exemplar_sorted.dim() == 2:
            exemplar_sorted = F.interpolate(
                exemplar_sorted.unsqueeze(0),
                size=sample_sorted.shape[-1],
                mode="nearest",
            ).squeeze(0)
        else:
            exemplar_sorted = F.interpolate(
                exemplar_sorted,
                size=sample_sorted.shape[-1],
                mode="nearest",
            )

    return torch.mean((exemplar_sorted - sample_sorted) ** 2)


def slice_loss(
    features: VGG19Features,
    exemplar: torch.Tensor,
    sample: torch.Tensor,
    generator: torch.Generator | None = None,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Compute the weighted multi-layer sliced Wasserstein loss."""
    exemplar_features = features(exemplar)
    sample_features = features(sample)

    if weights is None:
        weights = [1.0] * len(sample_features)
    if len(weights) != len(sample_features):
        raise ValueError(
            f"slice_loss expects {len(sample_features)} weights, got {len(weights)}."
        )

    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        raise ValueError("slice_loss expects weights with a positive sum.")

    normalized_weights = [weight / total_weight * len(weights) for weight in weights]

    loss = torch.zeros((), device=sample.device, dtype=sample.dtype)
    for weight, exemplar_feature, sample_feature in zip(
        normalized_weights, exemplar_features, sample_features
    ):
        loss = loss + sample.new_tensor(weight) * sliced_wasserstein_loss(
            exemplar_feature,
            sample_feature,
            generator=generator,
        )

    return loss
