"""Sampling utilities for crops, pixel shuffles, and frame selection."""

from __future__ import annotations

import torch


def _validate_image_tensor(image: torch.Tensor, *, fn_name: str) -> tuple[int, int, int]:
    if image.ndim != 3:
        raise ValueError(f"{fn_name} expects a 3D tensor shaped [C, H, W]")
    return tuple(image.shape)


def random_crop(
    image: torch.Tensor,
    crop_h: int,
    crop_w: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    c, h, w = _validate_image_tensor(image, fn_name="random_crop")
    del c

    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("crop_h and crop_w must be greater than 0")
    if crop_h > h or crop_w > w:
        raise ValueError("crop size must be less than or equal to image size")

    top = torch.randint(0, h - crop_h + 1, (), generator=generator, device=image.device).item()
    left = torch.randint(0, w - crop_w + 1, (), generator=generator, device=image.device).item()
    return image[:, top : top + crop_h, left : left + crop_w]


def random_take(
    image: torch.Tensor,
    new_h: int,
    new_w: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    c, h, w = _validate_image_tensor(image, fn_name="random_take")

    if new_h <= 0 or new_w <= 0:
        raise ValueError("new_h and new_w must be greater than 0")

    n = new_h * new_w
    if n > h * w:
        raise ValueError("new_h * new_w must be less than or equal to H * W")

    indices = torch.randperm(h * w, generator=generator, device=image.device)[:n]
    return image.reshape(c, -1)[:, indices].reshape(c, new_h, new_w)


def stratified_uniform(
    n: int,
    minval: float,
    maxval: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if n <= 0:
        raise ValueError("n must be greater than 0")
    if not minval < maxval:
        raise ValueError("minval must be less than maxval")

    length = (maxval - minval) / n
    return (
        torch.rand(n, generator=generator, device=device, dtype=torch.float32) * length
        + minval
        + length * torch.arange(n, device=device, dtype=torch.float32)
    )


def sample_frame_indices(
    n_frames: int,
    refresh_rate: int,
    step_in_cycle: int,
    *,
    generator: torch.Generator | None = None,
) -> int:
    if n_frames <= 0:
        raise ValueError("n_frames must be greater than 0")
    if refresh_rate <= 1:
        raise ValueError("refresh_rate must be greater than 1")
    if not 0 <= step_in_cycle < refresh_rate:
        raise ValueError("step_in_cycle must satisfy 0 <= step_in_cycle < refresh_rate")

    if step_in_cycle == 0:
        return 0

    segment = n_frames / (refresh_rate - 1)
    lower = segment * (step_in_cycle - 1)
    upper = segment * step_in_cycle
    sample = torch.rand((), generator=generator, dtype=torch.float32) * (upper - lower) + lower
    return min(int(sample.item()), n_frames - 1)
