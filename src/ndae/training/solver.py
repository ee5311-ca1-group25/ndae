"""Solver configuration and rollout result containers for NDAE training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.trajectory import TrajectoryModel


@dataclass(frozen=True)
class SolverConfig:
    """Numerical solver configuration for ODE integration."""

    method: str = "adaptive_heun"
    rtol: float = 1e-2
    atol: float = 1e-2
    options: dict[str, object] | None = None


@dataclass
class RolloutResult:
    """Result of a single rollout segment."""

    states: torch.Tensor
    final_state: torch.Tensor
    t0: float
    t1: float


def solve_rollout(
    trajectory_model: TrajectoryModel,
    z0: torch.Tensor,
    t0: float,
    t1: float,
    config: SolverConfig,
) -> RolloutResult:
    """Integrate a single rollout segment between two times."""

    t_eval = torch.tensor([t0, t1], dtype=z0.dtype, device=z0.device)
    states = trajectory_model(
        z0,
        t_eval,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        **(config.options or {}),
    )
    return RolloutResult(states=states, final_state=states[-1], t0=t0, t1=t1)


__all__ = ["SolverConfig", "RolloutResult", "solve_rollout"]
