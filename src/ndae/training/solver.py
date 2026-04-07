"""Solver configuration and rollout result containers for NDAE training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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


__all__ = ["SolverConfig", "RolloutResult"]
