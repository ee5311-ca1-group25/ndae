"""Timeline helpers for mapping between frame indices and ODE time."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ndae.config.schema import DataConfig


@dataclass(slots=True)
class Timeline:
    """Continuous-time view over exemplar frame indices."""

    t_I: float
    t_S: float
    t_E: float
    n_frames: int

    def __post_init__(self) -> None:
        if self.n_frames <= 0:
            raise ValueError("n_frames must be greater than 0")
        if not (self.t_I < self.t_S < self.t_E):
            raise ValueError("timeline must satisfy t_I < t_S < t_E")

    @property
    def dt(self) -> float:
        return (self.t_E - self.t_S) / self.n_frames

    @property
    def warmup_duration(self) -> float:
        return self.t_S - self.t_I

    @property
    def generation_duration(self) -> float:
        return self.t_E - self.t_S

    def frame_to_time(self, k: int) -> float:
        if not 0 <= k < self.n_frames:
            raise IndexError(f"frame index out of range: {k}")
        return self.t_S + k * self.dt

    def time_to_frame(self, t: float) -> int:
        relative_t = max(t - self.t_S, 0.0)
        ratio = relative_t / self.dt
        # Preserve frame_to_time/time_to_frame round-trips under float imprecision.
        k = int(math.floor(ratio + 1e-9))
        return min(k, self.n_frames - 1)

    @classmethod
    def from_config(cls, data_config: DataConfig) -> Timeline:
        return cls(
            t_I=data_config.t_I,
            t_S=data_config.t_S,
            t_E=data_config.t_E,
            n_frames=data_config.n_frames,
        )
