"""UNet assembly for NDAE models."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock, LinearTimeSelfAttention, Residual, Resample, SpatialLinear, zero_init
from .time_embedding import TimeMLP


class NDAEUNet(nn.Module):
    """JAX-aligned UNet used as the NDAE vector field backbone."""

    def __init__(
        self,
        in_dim: int = 9,
        out_dim: int = 9,
        dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2),
        use_attn: bool = False,
    ) -> None:
        super().__init__()
        if not dim_mults:
            raise ValueError("NDAEUNet expects dim_mults to be non-empty.")
        if dim_mults[0] != 1:
            raise ValueError(
                f"NDAEUNet expects dim_mults to start with 1, got {dim_mults}."
            )

        time_dim = dim * 2
        dims = [dim * mult for mult in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.init_conv = ConvBlock(in_dim, dim, kernel_size=1)
        self.time_mlp = TimeMLP(dim)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.downs.append(
                nn.ModuleList(
                    [
                        ConvBlock(dim_in, dim_in, emb_dim=time_dim),
                        Residual(LinearTimeSelfAttention(dim_in))
                        if use_attn
                        else nn.Identity(),
                        Resample(dim_in, dim_out, 0.5),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block = ConvBlock(mid_dim, mid_dim, emb_dim=time_dim)
        self.mid_attn = Residual(LinearTimeSelfAttention(mid_dim)) if use_attn else nn.Identity()

        for dim_in, dim_out in reversed(in_out):
            self.ups.append(
                nn.ModuleList(
                    [
                        Resample(dim_out, dim_in, 2),
                        ConvBlock(dim_in * 2, dim_in, emb_dim=time_dim),
                        Residual(LinearTimeSelfAttention(dim_in))
                        if use_attn
                        else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.ModuleList(
            [
                ConvBlock(dim * 2, dim, kernel_size=1, act=torch.sigmoid),
                zero_init(SpatialLinear(dim, out_dim)),
            ]
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the UNet as f(t, x) with scalar or batched time input."""
        if x.dim() != 4:
            raise ValueError(
                f"NDAEUNet expects x shaped (B, C, H, W), got {tuple(x.shape)}."
            )

        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        elif t.dim() == 1:
            if t.shape[0] != batch_size:
                raise ValueError(
                    f"NDAEUNet expects batched time with length {batch_size}, got {t.shape[0]}."
                )
        else:
            raise ValueError(
                f"NDAEUNet expects time shaped [] or [B], got {tuple(t.shape)}."
            )

        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = [x]
        for block, attn, downsample in self.downs:
            x = block(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block(x, t_emb)
        x = self.mid_attn(x)

        for upsample, block, attn in self.ups:
            x = upsample(x)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block(x, t_emb)
            x = attn(x)

        x = torch.cat((x, skips.pop()), dim=1)
        for layer in self.final_conv:
            x = layer(x)

        assert len(skips) == 0, "all skip connections should be consumed"
        return x
