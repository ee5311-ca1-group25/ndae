"""Offline sampling helpers for evaluation and checkpoint export."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..data import Timeline

if TYPE_CHECKING:
    from ..training.system import SVBRDFSystem


def build_sample_timeline(
    timeline: Timeline,
    *,
    dtype: torch.dtype,
    synthesis_frames: int = 50,
) -> tuple[torch.Tensor, int]:
    syn_t = timeline.t_S - timeline.t_I
    ts_synthesis = torch.logspace(
        0.0,
        math.log10(1.0 + syn_t),
        synthesis_frames,
        dtype=dtype,
    ) - 1.0 - syn_t
    ts_transition = torch.linspace(
        timeline.t_S,
        timeline.t_E,
        timeline.n_frames,
        dtype=dtype,
    )
    return torch.cat([ts_synthesis, ts_transition]), synthesis_frames


def sample_sequence(
    system: "SVBRDFSystem",
    timeline: Timeline,
    *,
    sample_size: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")

    parameter = next(system.trajectory_model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    z0 = torch.randn(
        1,
        system.total_channels,
        sample_size,
        sample_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    t_eval, synthesis_count = build_sample_timeline(timeline, dtype=dtype)
    t_eval = t_eval.to(device=device)

    system.trajectory_model.eval()
    with torch.no_grad():
        states = system.trajectory_model(
            z0,
            t_eval,
            method=system.solver_config.method,
            rtol=system.solver_config.rtol,
            atol=system.solver_config.atol,
            **(system.solver_config.options or {}),
        )
    return states, synthesis_count


__all__ = ["build_sample_timeline", "sample_sequence"]
