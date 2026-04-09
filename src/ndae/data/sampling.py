"""Sampling utilities for crops, pixel shuffles, and frame selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


@dataclass(slots=True)
class CropSampleSpec:
    """Describe one training-space sample shared by target and rendering paths."""

    kind: Literal["rect", "take"]
    height: int
    width: int
    top: int | None = None
    left: int | None = None
    top_ratio: float | None = None
    left_ratio: float | None = None
    indices: torch.Tensor | None = None


def sample_random_crop_spec(
    image: torch.Tensor,
    crop_h: int,
    crop_w: int,
    *,
    generator: torch.Generator | None = None,
) -> CropSampleSpec:
    _, h, w = _validate_image_tensor(image, fn_name="sample_random_crop_spec")
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("crop_h and crop_w must be greater than 0")
    if crop_h > h or crop_w > w:
        raise ValueError("crop size must be less than or equal to image size")

    max_top = h - crop_h
    max_left = w - crop_w
    top = torch.randint(0, max_top + 1, (), generator=generator, device=image.device).item()
    left = torch.randint(0, max_left + 1, (), generator=generator, device=image.device).item()
    top_ratio = 0.0 if max_top == 0 else top / max_top
    left_ratio = 0.0 if max_left == 0 else left / max_left
    return CropSampleSpec(
        kind="rect",
        height=crop_h,
        width=crop_w,
        top=top,
        left=left,
        top_ratio=top_ratio,
        left_ratio=left_ratio,
    )


def apply_crop_spec(image: torch.Tensor, spec: CropSampleSpec) -> torch.Tensor:
    _, h, w = _validate_image_tensor(image, fn_name="apply_crop_spec")
    if spec.kind != "rect":
        raise ValueError("apply_crop_spec expects a rect CropSampleSpec")
    max_top = h - spec.height
    max_left = w - spec.width
    if max_top < 0 or max_left < 0:
        raise ValueError("crop size must be less than or equal to image size")
    if spec.top_ratio is not None and spec.left_ratio is not None:
        top = 0 if max_top == 0 else int(round(spec.top_ratio * max_top))
        left = 0 if max_left == 0 else int(round(spec.left_ratio * max_left))
    elif spec.top is not None and spec.left is not None:
        top = spec.top
        left = spec.left
    else:
        raise ValueError("rect CropSampleSpec requires top/left or ratio fields")
    return image[:, top : top + spec.height, left : left + spec.width]


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

    spec = sample_random_take_spec(image, new_h, new_w, generator=generator)
    return apply_take_spec(image, spec)


def sample_random_take_spec(
    image: torch.Tensor,
    new_h: int,
    new_w: int,
    *,
    generator: torch.Generator | None = None,
) -> CropSampleSpec:
    _, h, w = _validate_image_tensor(image, fn_name="sample_random_take_spec")

    if new_h <= 0 or new_w <= 0:
        raise ValueError("new_h and new_w must be greater than 0")

    n = new_h * new_w
    if n > h * w:
        raise ValueError("new_h * new_w must be less than or equal to H * W")

    indices = torch.randperm(h * w, generator=generator, device=image.device)[:n]
    return CropSampleSpec(kind="take", height=new_h, width=new_w, indices=indices)


def apply_take_spec(image: torch.Tensor, spec: CropSampleSpec) -> torch.Tensor:
    c, h, w = _validate_image_tensor(image, fn_name="apply_take_spec")
    if spec.kind != "take":
        raise ValueError("apply_take_spec expects a take CropSampleSpec")
    if spec.indices is None:
        raise ValueError("take CropSampleSpec requires indices")
    n = spec.height * spec.width
    if spec.indices.numel() != n:
        raise ValueError("take CropSampleSpec indices must have height * width elements")
    source_count = h * w
    indices = spec.indices.to(device=image.device, dtype=torch.long)
    if torch.any(indices < 0) or torch.any(indices >= source_count):
        raise ValueError("take CropSampleSpec indices must lie inside the source image extent")
    return image.reshape(c, -1)[:, indices].reshape(c, spec.height, spec.width)


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
