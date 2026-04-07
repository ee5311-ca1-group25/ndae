"""Building blocks for Lecture 5 NDAE models."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_init(module: nn.Module) -> nn.Module:
    """Zero-initialize all learnable parameters in-place."""
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class DefaultConv2d(nn.Module):
    """3x3 convolution with explicit circular padding."""

    def __init__(self, dim: int, dim_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=3, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        return self.conv(x)


class SpatialLinear(nn.Module):
    """1x1 convolution used where the JAX code applies SpatialLinear."""

    def __init__(self, dim: int, dim_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvBlock(nn.Module):
    """Convolution, GroupNorm, optional time scale-shift, then activation."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        kernel_size: int = 3,
        emb_dim: int | None = None,
        act: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ) -> None:
        super().__init__()
        if kernel_size not in (1, 3):
            raise ValueError(f"ConvBlock kernel_size must be 1 or 3, got {kernel_size}.")

        self.proj: nn.Module
        if kernel_size == 3:
            self.proj = DefaultConv2d(dim, dim_out)
        else:
            self.proj = SpatialLinear(dim, dim_out)

        has_emb = emb_dim is not None
        self.norm = nn.GroupNorm(min(dim_out // 4, 32), dim_out, affine=not has_emb)
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                zero_init(nn.Linear(emb_dim, dim_out * 2)),
            )
            if has_emb
            else None
        )
        self.act = act

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if self.mlp is not None and emb is not None:
            scale_shift = self.mlp(emb).unsqueeze(-1).unsqueeze(-1)
            scale, shift = scale_shift.chunk(2, dim=1)
            x = x * (scale + 1) + shift

        return self.act(x)


class Resample(nn.Module):
    """Resize with bilinear interpolation followed by DefaultConv2d."""

    def __init__(self, dim: int, dim_out: int, factor: float) -> None:
        super().__init__()
        self.factor = factor
        self.conv = DefaultConv2d(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=self.factor,
            mode="bilinear",
            align_corners=False,
        )
        return self.conv(x)


class LinearTimeSelfAttention(nn.Module):
    """JAX-aligned multi-head linear attention with zero-initialized output."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.heads = 4
        self.dim_head = 8
        hidden_dim = self.heads * self.dim_head
        self.group_norm = nn.GroupNorm(min(dim // 4, 32), dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1)
        self.to_out = zero_init(nn.Conv2d(hidden_dim, dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = self.group_norm(x)
        qkv = self.to_qkv(x).reshape(b, 3, self.heads, self.dim_head, h * w)
        q, k, v = qkv.unbind(dim=1)

        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.reshape(b, self.heads * self.dim_head, h, w)
        return self.to_out(out)


class Residual(nn.Module):
    """Add a module's output back to its input."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)
