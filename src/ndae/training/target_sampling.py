"""Training target sampling and rendering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from ..data import (
    CropSampleSpec,
    apply_crop_spec,
    apply_take_spec,
    sample_random_crop_spec,
    sample_random_take_spec,
)
from ..rendering import clip_maps, create_meshgrid, height_to_normal, render_svbrdf, tonemapping

if TYPE_CHECKING:
    from .trainer import Trainer


def sample_target_batch(
    trainer: "Trainer",
    target_index: int,
    *,
    current_stage: Literal["init", "local"],
    brdf_maps: torch.Tensor,
    height_map: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target_frame = trainer.exemplar_frames[target_index]
    zero_normal = height_to_normal(
        torch.zeros_like(height_map),
        scale=trainer.system.height_scale,
    )
    true_normal = height_to_normal(height_map, scale=trainer.system.height_scale)
    targets: list[torch.Tensor] = []
    renderings: list[torch.Tensor] = []

    for batch_index in range(trainer.runtime.batch_size):
        for _ in range(trainer.loss_cfg.n_loss_crops):
            spec = sample_spec(trainer, target_frame, current_stage=current_stage)
            targets.append(apply_target_spec(target_frame, spec))
            renderings.append(
                render_sample(
                    trainer,
                    brdf_maps[batch_index],
                    zero_normal[batch_index] if current_stage == "init" else true_normal[batch_index],
                    spec,
                    image_height=target_frame.shape[-2],
                    image_width=target_frame.shape[-1],
                )
            )

    target_batch = torch.stack(targets, dim=0)
    rendered_batch = tonemapping(
        torch.stack(renderings, dim=0),
        gamma=trainer.runtime.gamma,
    )
    return {"target": target_batch, "rendered": rendered_batch}


def sample_spec(
    trainer: "Trainer",
    target_frame: torch.Tensor,
    *,
    current_stage: Literal["init", "local"],
) -> CropSampleSpec:
    if current_stage == "init":
        return sample_random_take_spec(
            target_frame,
            trainer.crop_size,
            trainer.crop_size,
            generator=trainer.runtime.generator,
        )
    return sample_random_crop_spec(
        target_frame,
        trainer.crop_size,
        trainer.crop_size,
        generator=trainer.runtime.generator,
    )


def apply_target_spec(target_frame: torch.Tensor, spec: CropSampleSpec) -> torch.Tensor:
    if spec.kind == "take":
        return apply_take_spec(target_frame, spec)
    return apply_crop_spec(target_frame, spec)



def render_sample(
    trainer: "Trainer",
    brdf_maps: torch.Tensor,
    normal_map: torch.Tensor,
    spec: CropSampleSpec,
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    clipped_maps = clip_maps(brdf_maps)
    full_positions = create_meshgrid(
        image_height,
        image_width,
        trainer.system.camera,
        device=brdf_maps.device,
    )
    if spec.kind == "take":
        sample_positions = apply_take_spec(full_positions, spec)
        return render_svbrdf(
            clipped_maps,
            normal_map,
            trainer.system.camera,
            trainer.system.flash_light,
            trainer.system.renderer_pp,
            trainer.system.unpack_fn,
            positions=sample_positions,
        )

    sample_brdf = apply_crop_spec(clipped_maps, spec)
    sample_normal = apply_crop_spec(normal_map, spec)
    sample_positions = apply_crop_spec(full_positions, spec)
    return render_svbrdf(
        sample_brdf,
        sample_normal,
        trainer.system.camera,
        trainer.system.flash_light,
        trainer.system.renderer_pp,
        trainer.system.unpack_fn,
        positions=sample_positions,
    )


def region_origin_for(
    image: torch.Tensor,
    spec: CropSampleSpec,
) -> tuple[int, int]:
    _, h, w = image.shape
    max_top = h - spec.height
    max_left = w - spec.width
    if max_top < 0 or max_left < 0:
        raise ValueError("crop size must be less than or equal to image size")
    if spec.top_ratio is not None and spec.left_ratio is not None:
        top = 0 if max_top == 0 else int(round(spec.top_ratio * max_top))
        left = 0 if max_left == 0 else int(round(spec.left_ratio * max_left))
        return top, left
    if spec.top is None or spec.left is None:
        raise ValueError("rect CropSampleSpec requires top/left or ratio fields")
    return spec.top, spec.left


__all__ = [
    "apply_target_spec",
    "region_origin_for",
    "render_sample",
    "sample_spec",
    "sample_target_batch",
]
