"""Time embedding modules for NDAE models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Map a scalar time value to a fixed sinusoidal embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 4 or dim % 2 != 0:
            raise ValueError(
                f"SinusoidalTimeEmbedding expects an even dim >= 4, got {dim}."
            )

        half_dim = dim // 2
        decay = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -decay)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode scalar `[]` or batched `[B]` times as sinusoidal features."""
        if t.dim() not in (0, 1):
            raise ValueError(
                "SinusoidalTimeEmbedding expects time shaped [] or [B], "
                f"got {tuple(t.shape)}."
            )

        freqs = self.freqs.to(device=t.device, dtype=t.dtype)
        phases = t.unsqueeze(-1) * freqs
        return torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)


class TimeMLP(nn.Module):
    """Project sinusoidal time features to the UNet time-conditioning width."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        time_dim = dim * 2
        self.sinusoidal_emb = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return a learned time embedding for scalar `[]` or batched `[B]` times."""
        return self.mlp(self.sinusoidal_emb(t))
