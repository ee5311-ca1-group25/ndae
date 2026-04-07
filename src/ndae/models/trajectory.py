"""Trajectory integration wrapper for NDAE models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint

from .odefunc import ODEFunction


class TrajectoryModel(nn.Module):
    """Integrate an ODE function over evaluation times."""

    def __init__(self, odefunc: ODEFunction) -> None:
        super().__init__()
        self.odefunc = odefunc

    def forward(
        self,
        z0: torch.Tensor,
        t_eval: torch.Tensor,
        **solver_kwargs,
    ) -> torch.Tensor:
        if z0.dim() != 4:
            raise ValueError(
                f"TrajectoryModel expects z0 shaped (B, C, H, W), got {tuple(z0.shape)}."
            )
        if t_eval.dim() != 1:
            raise ValueError(
                f"TrajectoryModel expects t_eval shaped [T], got {tuple(t_eval.shape)}."
            )
        if t_eval.numel() < 2:
            raise ValueError(
                "TrajectoryModel expects t_eval to contain at least 2 time points."
            )

        return odeint(self.odefunc, z0, t_eval, **solver_kwargs)
