"""Trainer runtime state containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(slots=True)
class TrainerState:
    """Minimal runtime state carried across training steps."""

    global_step: int
    stage: Literal["init", "local"]
    carry_time: float
    carry_state: torch.Tensor | None
    cycle_step: int


__all__ = ["TrainerState"]
