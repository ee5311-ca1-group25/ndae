import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ndae.models.blocks import (
    ConvBlock,
    DefaultConv2d,
    LinearTimeSelfAttention,
    Residual,
    Resample,
    SpatialLinear,
    zero_init,
)
from ndae.models.time_embedding import SinusoidalTimeEmbedding, TimeMLP
from ndae.models.unet import NDAEUNet


def test_sinusoidal_time_embedding_scalar_shape_and_bounds() -> None:
    embedding = SinusoidalTimeEmbedding(64)

    output = embedding(torch.tensor(0.5))

    assert output.shape == (64,)
    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)


def test_sinusoidal_time_embedding_batch_shape_and_bounds() -> None:
    embedding = SinusoidalTimeEmbedding(64)

    output = embedding(torch.tensor([0.0, 0.5, 1.0]))

    assert output.shape == (3, 64)
    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)


def test_sinusoidal_time_embedding_distinguishes_different_times() -> None:
    embedding = SinusoidalTimeEmbedding(64)

    first = embedding(torch.tensor(0.5))
    second = embedding(torch.tensor(1.0))
    similarity = F.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item()

    assert not torch.allclose(first, second)
    assert similarity < 1.0


def test_sinusoidal_time_embedding_is_smooth_for_nearby_times() -> None:
    embedding = SinusoidalTimeEmbedding(64)

    first = embedding(torch.tensor(0.5))
    second = embedding(torch.tensor(0.501))
    similarity = F.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item()

    assert similarity > 0.999


def test_time_mlp_batch_output_shape() -> None:
    mlp = TimeMLP(32)

    output = mlp(torch.tensor([0.0, 0.5, 1.0]))

    assert output.shape == (3, 64)


def test_time_mlp_gradient_flows_to_input_time() -> None:
    mlp = TimeMLP(32)
    t = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)

    mlp(t).sum().backward()

    assert t.grad is not None
    assert not torch.isnan(t.grad).any()


@pytest.mark.parametrize("dim", [3, 5])
def test_time_embedding_rejects_invalid_dim(dim: int) -> None:
    with pytest.raises(ValueError, match="even dim >= 4"):
        SinusoidalTimeEmbedding(dim)


def test_time_embedding_rejects_dim_smaller_than_four() -> None:
    with pytest.raises(ValueError, match="even dim >= 4"):
        SinusoidalTimeEmbedding(2)


@pytest.mark.parametrize("module_cls", [SinusoidalTimeEmbedding, TimeMLP])
def test_time_embedding_rejects_invalid_input_rank(
    module_cls: type[SinusoidalTimeEmbedding] | type[TimeMLP],
) -> None:
    module = module_cls(32)

    with pytest.raises(ValueError, match=r"expects time shaped \[\] or \[B\]"):
        module(torch.zeros(2, 1))


@pytest.mark.parametrize("module", [nn.Linear(8, 4), nn.Conv2d(3, 5, kernel_size=1)])
def test_zero_init_zeros_all_parameters(module: nn.Module) -> None:
    initialized = zero_init(module)

    assert initialized is module
    for parameter in module.parameters():
        assert torch.count_nonzero(parameter) == 0


def test_default_conv2d_preserves_spatial_shape() -> None:
    module = DefaultConv2d(32, 64)

    output = module(torch.randn(2, 32, 16, 16))

    assert output.shape == (2, 64, 16, 16)


def test_default_conv2d_uses_circular_padding() -> None:
    module = DefaultConv2d(1, 1)
    with torch.no_grad():
        module.conv.weight.zero_()
        module.conv.bias.zero_()
        module.conv.weight[0, 0, 0, 0] = 1.0

    x = torch.zeros(1, 1, 3, 3)
    x[0, 0, 2, 2] = 5.0

    output = module(x)

    assert output[0, 0, 0, 0].item() == pytest.approx(5.0)


def test_spatial_linear_preserves_spatial_shape() -> None:
    module = SpatialLinear(32, 64)

    output = module(torch.randn(2, 32, 16, 16))

    assert output.shape == (2, 64, 16, 16)


def test_conv_block_kernel_size_3_uses_default_conv2d() -> None:
    block = ConvBlock(32, 64, kernel_size=3)

    assert isinstance(block.proj, DefaultConv2d)


def test_conv_block_kernel_size_1_uses_spatial_linear() -> None:
    block = ConvBlock(32, 64, kernel_size=1)

    assert isinstance(block.proj, SpatialLinear)


def test_conv_block_rejects_invalid_kernel_size() -> None:
    with pytest.raises(ValueError, match="kernel_size must be 1 or 3"):
        ConvBlock(32, 64, kernel_size=5)


def test_conv_block_with_zero_init_time_embedding_matches_no_embedding() -> None:
    block_with_t = ConvBlock(32, 32, emb_dim=64)
    block_without_t = ConvBlock(32, 32, emb_dim=None)
    block_without_t.proj.load_state_dict(block_with_t.proj.state_dict())

    x = torch.randn(2, 32, 16, 16)
    emb = torch.randn(2, 64)

    with_t = block_with_t(x, emb)
    without_t = block_without_t(x)

    assert torch.allclose(with_t, without_t, atol=1e-6)


def test_conv_block_gradients_flow_to_input_and_embedding() -> None:
    block = ConvBlock(32, 32, emb_dim=64)
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    emb = torch.randn(2, 64, requires_grad=True)

    block(x, emb).sum().backward()

    assert x.grad is not None
    assert emb.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(emb.grad).any()


def test_resample_downsample_shape() -> None:
    module = Resample(32, 64, factor=0.5)

    output = module(torch.randn(2, 32, 16, 16))

    assert output.shape == (2, 64, 8, 8)


def test_resample_upsample_shape() -> None:
    module = Resample(64, 32, factor=2)

    output = module(torch.randn(2, 64, 16, 16))

    assert output.shape == (2, 32, 32, 32)


def test_attention_zero_init_output_is_near_zero() -> None:
    module = LinearTimeSelfAttention(32)

    output = module(torch.randn(2, 32, 16, 16))

    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)


def test_residual_attention_matches_input_at_initialization() -> None:
    module = Residual(LinearTimeSelfAttention(32))
    x = torch.randn(2, 32, 16, 16)

    output = module(x)

    assert torch.allclose(output, x, atol=1e-6)


def test_residual_attention_gradients_flow_to_input() -> None:
    module = Residual(LinearTimeSelfAttention(32))
    x = torch.randn(2, 32, 16, 16, requires_grad=True)

    module(x).sum().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_unet_forward_shape_default_config() -> None:
    model = NDAEUNet(in_dim=9, out_dim=9, dim=32, dim_mults=(1, 2), use_attn=False)

    output = model(torch.tensor([0.0, 0.5]), torch.randn(2, 9, 64, 64))

    assert output.shape == (2, 9, 64, 64)


def test_unet_forward_shape_with_deeper_dim_mults() -> None:
    model = NDAEUNet(in_dim=9, out_dim=9, dim=32, dim_mults=(1, 2, 4), use_attn=False)

    output = model(torch.tensor([0.0, 0.5]), torch.randn(2, 9, 64, 64))

    assert output.shape == (2, 9, 64, 64)


@pytest.mark.parametrize("use_attn", [False, True])
def test_unet_supports_attention_toggle(use_attn: bool) -> None:
    model = NDAEUNet(in_dim=9, out_dim=9, dim=32, dim_mults=(1, 2), use_attn=use_attn)

    output = model(torch.tensor([0.0, 0.5]), torch.randn(2, 9, 64, 64))

    assert output.shape == (2, 9, 64, 64)


def test_unet_supports_scalar_time_input() -> None:
    model = NDAEUNet(in_dim=9, out_dim=9, dim=32, dim_mults=(1, 2), use_attn=False)

    output = model(torch.tensor(0.5), torch.randn(2, 9, 64, 64))

    assert output.shape == (2, 9, 64, 64)


def test_unet_zero_init_output_is_near_zero() -> None:
    model = NDAEUNet(in_dim=9, out_dim=9, dim=32, dim_mults=(1, 2), use_attn=False)

    output = model(torch.tensor([0.0, 0.5]), torch.randn(2, 9, 64, 64))

    assert output.abs().max().item() < 1e-5


def test_unet_gradients_flow_when_final_layer_is_unfrozen_from_zero() -> None:
    model = NDAEUNet(in_dim=9, out_dim=9, dim=32, dim_mults=(1, 2), use_attn=False)
    with torch.no_grad():
        model.final_conv[-1].conv.weight.fill_(0.01)
        model.final_conv[-1].conv.bias.zero_()

    x = torch.randn(2, 9, 64, 64, requires_grad=True)
    output = model(torch.tensor([0.0, 0.5]), x)
    output.square().mean().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    parameter_grads = [parameter.grad for parameter in model.parameters()]
    assert all(grad is not None for grad in parameter_grads)


def test_unet_rejects_invalid_x_rank() -> None:
    model = NDAEUNet()

    with pytest.raises(ValueError, match=r"expects x shaped \(B, C, H, W\)"):
        model(torch.tensor([0.0]), torch.randn(9, 64, 64))


def test_unet_rejects_invalid_time_rank() -> None:
    model = NDAEUNet()

    with pytest.raises(ValueError, match=r"expects time shaped \[\] or \[B\]"):
        model(torch.zeros(2, 1), torch.randn(2, 9, 64, 64))


def test_unet_rejects_time_batch_mismatch() -> None:
    model = NDAEUNet()

    with pytest.raises(ValueError, match="expects batched time with length 2"):
        model(torch.tensor([0.0]), torch.randn(2, 9, 64, 64))


def test_unet_rejects_dim_mults_without_leading_one() -> None:
    with pytest.raises(ValueError, match="dim_mults to start with 1"):
        NDAEUNet(dim_mults=(2, 4))


def test_unet_rejects_empty_dim_mults() -> None:
    with pytest.raises(ValueError, match="dim_mults to be non-empty"):
        NDAEUNet(dim_mults=())


def test_parameter_summary(capsys: pytest.CaptureFixture[str]) -> None:
    from ndae.models import NDAEUNet as PublicNDAEUNet

    model = PublicNDAEUNet(in_dim=9, out_dim=9, dim=32, dim_mults=(1, 2))
    total = sum(parameter.numel() for parameter in model.parameters())
    subtotal = 0

    for name, module in model.named_children():
        params = sum(parameter.numel() for parameter in module.parameters())
        subtotal += params
        print(f"{name}: {params:,}")
    print(f"Total: {total:,}")

    output = capsys.readouterr().out
    assert "init_conv:" in output
    assert "time_mlp:" in output
    assert "downs:" in output
    assert "mid_block:" in output
    assert "mid_attn:" in output
    assert "ups:" in output
    assert "final_conv:" in output
    assert "Total:" in output
    assert total > 0
    assert subtotal == total
