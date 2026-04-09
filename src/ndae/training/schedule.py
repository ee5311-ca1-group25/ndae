"""Time-window configuration objects for NDAE training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..data.sampling import stratified_uniform


@dataclass(frozen=True)
class StageConfig:
    """Time boundaries and refresh rate for NDAE training."""

    t_init: float = -2.0
    t_start: float = 0.0
    t_end: float = 10.0
    refresh_rate: int = 6

    def __post_init__(self) -> None:
        if self.t_init >= self.t_start:
            raise ValueError(
                f"t_init must be < t_start, got {self.t_init} >= {self.t_start}"
            )
        if self.t_start >= self.t_end:
            raise ValueError(
                f"t_start must be < t_end, got {self.t_start} >= {self.t_end}"
            )
        if self.refresh_rate < 2:
            raise ValueError(
                f"refresh_rate must be >= 2, got {self.refresh_rate}"
            )


@dataclass(frozen=True)
class RolloutWindow:
    """Description of one rollout interval in the training cycle."""

    kind: Literal["warmup", "generation"]
    t0: float
    t1: float
    refresh: bool


class RefreshSchedule:
    """Implements Algorithm 1's refresh cycle and stratified time sampling."""

    def __init__(
        self,
        config: StageConfig,
        generator: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self.generator = generator
        self._strata_deltas: torch.Tensor | None = None

    def next(self, iteration: int, carry_time: float) -> RolloutWindow:
        k = iteration % self.config.refresh_rate
        if k == 0:
            self._strata_deltas = self._sample_strata()
            return RolloutWindow(
                kind="warmup",
                t0=self.config.t_init,
                t1=self.config.t_start,
                refresh=True,
            )

        if self._strata_deltas is None:
            raise RuntimeError(
                "RefreshSchedule requires a warmup step before generation steps."
            )

        delta = self._strata_deltas[k - 1].item()
        return RolloutWindow(
            kind="generation",
            t0=carry_time,
            t1=carry_time + delta,
            refresh=False,
        )

    def _sample_strata(self) -> torch.Tensor:
        n = self.config.refresh_rate - 1
        abs_times = stratified_uniform(
            n,
            self.config.t_start,
            self.config.t_end,
            generator=self.generator,
        )
        deltas = abs_times.clone()
        deltas[0] = abs_times[0] - self.config.t_start
        deltas[1:] = abs_times[1:] - abs_times[:-1]
        return deltas


__all__ = ["StageConfig", "RolloutWindow", "RefreshSchedule"]
