"""ODE function adapter for Lecture 6 NDAE models."""

from __future__ import annotations

import torch
import torch.nn as nn


class ODEFunction(nn.Module):
    """Expose a vector field with the solver-friendly ``f(t, state)`` signature."""

    def __init__(self, vector_field: nn.Module) -> None:
        super().__init__()
        self.vector_field = vector_field

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.vector_field(t, state)
