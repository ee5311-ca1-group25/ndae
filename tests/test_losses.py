import pytest
import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19 as torchvision_vgg19

import ndae.losses.perceptual as perceptual
from ndae.losses import (
    gram_loss,
    gram_matrix,
    slice_loss,
    sliced_wasserstein_loss,
)


def patch_vgg19(monkeypatch: pytest.MonkeyPatch) -> list[VGG19_Weights]:
    requests: list[VGG19_Weights] = []

    def fake_vgg19(*, weights: VGG19_Weights) -> nn.Module:
        requests.append(weights)
        return torchvision_vgg19(weights=None)

    monkeypatch.setattr(perceptual, "vgg19", fake_vgg19)
    return requests


def test_vgg19_output_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()
    x = torch.rand(1, 3, 256, 256)
    features = model(x)

    assert requests == [VGG19_Weights.IMAGENET1K_V1]
    assert [tuple(feature.shape) for feature in features] == [
        (1, 3, 256, 256),
        (1, 64, 256, 256),
        (1, 128, 128, 128),
        (1, 256, 64, 64),
        (1, 512, 32, 32),
        (1, 512, 16, 16),
    ]
    assert isinstance(model.pool, nn.AvgPool2d)
    assert not any(isinstance(module, nn.MaxPool2d) for module in model.modules())


def test_vgg19_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()

    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_vgg19_gradient_to_input(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    features = model(x)
    features[-1].sum().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_vgg19_rejects_invalid_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()

    with pytest.raises(ValueError, match="expects input shaped"):
        model(torch.rand(3, 64, 64))


def test_vgg19_rejects_non_rgb_input(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()

    with pytest.raises(ValueError, match="expects a 3-channel input"):
        model(torch.rand(1, 1, 64, 64))


def test_gram_matrix_shape_and_symmetry() -> None:
    feature = torch.randn(64, 32, 32)

    gram = gram_matrix(feature)

    assert gram.shape == (64, 64)
    assert torch.allclose(gram, gram.transpose(0, 1), atol=1e-6)


def test_gram_matrix_supports_batches() -> None:
    feature = torch.randn(2, 64, 32, 32)

    gram = gram_matrix(feature)

    assert gram.shape == (2, 64, 64)


def test_gram_matrix_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="gram_matrix expects input shaped"):
        gram_matrix(torch.randn(64, 32))


def test_gram_loss_same_input_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()
    image = torch.rand(1, 3, 64, 64)

    loss = gram_loss(model, image, image)

    assert loss.shape == ()
    assert loss.item() < 1e-6


def test_gram_loss_diff_input_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()
    exemplar = torch.zeros(1, 3, 64, 64)
    sample = torch.ones(1, 3, 64, 64)

    loss = gram_loss(model, exemplar, sample)

    assert loss.item() > 0.0


def test_gram_loss_gradient_to_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()
    exemplar = torch.rand(1, 3, 64, 64)
    sample = torch.rand(1, 3, 64, 64, requires_grad=True)

    loss = gram_loss(model, exemplar, sample)
    loss.backward()

    assert sample.grad is not None
    assert not torch.isnan(sample.grad).any()


def test_swd_same_input_zero() -> None:
    feature = torch.randn(32, 8, 8)

    loss = sliced_wasserstein_loss(feature, feature)

    assert loss.shape == ()
    assert loss.item() < 1e-6


def test_swd_diff_input_positive() -> None:
    exemplar = torch.zeros(32, 8, 8)
    sample = torch.ones(32, 8, 8)

    loss = sliced_wasserstein_loss(exemplar, sample)

    assert loss.item() > 0.0


def test_swd_resizes_exemplar_projection_length() -> None:
    exemplar = torch.randn(32, 7, 7)
    sample = torch.randn(32, 9, 9)

    loss = sliced_wasserstein_loss(exemplar, sample)

    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_swd_reproducible() -> None:
    exemplar = torch.randn(32, 8, 8)
    sample = torch.randn(32, 8, 8)

    generator_a = torch.Generator().manual_seed(42)
    generator_b = torch.Generator().manual_seed(42)

    loss_a = sliced_wasserstein_loss(exemplar, sample, generator=generator_a)
    loss_b = sliced_wasserstein_loss(exemplar, sample, generator=generator_b)

    assert torch.allclose(loss_a, loss_b)


def test_swd_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="expects inputs shaped"):
        sliced_wasserstein_loss(torch.randn(8, 8), torch.randn(8, 8))


def test_swd_rejects_channel_mismatch() -> None:
    with pytest.raises(ValueError, match="matching channel counts"):
        sliced_wasserstein_loss(torch.randn(16, 8, 8), torch.randn(32, 8, 8))


def test_slice_loss_gradient_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()
    exemplar = torch.rand(1, 3, 64, 64)
    sample = torch.rand(1, 3, 64, 64, requires_grad=True)

    loss = slice_loss(model, exemplar, sample, generator=torch.Generator().manual_seed(7))
    loss.backward()

    assert sample.grad is not None
    assert not torch.isnan(sample.grad).any()


def test_slice_loss_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()
    exemplar = torch.rand(1, 3, 64, 64)
    sample = torch.rand(1, 3, 64, 64)

    loss_a = slice_loss(
        model,
        exemplar,
        sample,
        generator=torch.Generator().manual_seed(42),
    )
    loss_b = slice_loss(
        model,
        exemplar,
        sample,
        generator=torch.Generator().manual_seed(42),
    )

    assert torch.allclose(loss_a, loss_b)


def test_slice_loss_rejects_invalid_weight_count(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()
    exemplar = torch.rand(1, 3, 64, 64)
    sample = torch.rand(1, 3, 64, 64)

    with pytest.raises(ValueError, match="expects 6 weights"):
        slice_loss(model, exemplar, sample, weights=[1.0, 1.0])


def test_slice_loss_rejects_non_positive_weight_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_vgg19(monkeypatch)

    model = perceptual.VGG19Features()
    exemplar = torch.rand(1, 3, 64, 64)
    sample = torch.rand(1, 3, 64, 64)

    with pytest.raises(ValueError, match="positive sum"):
        slice_loss(model, exemplar, sample, weights=[0.0] * 6)
