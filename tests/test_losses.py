import pytest
import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19 as torchvision_vgg19

import ndae.losses.perceptual as perceptual


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
