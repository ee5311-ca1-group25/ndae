import pytest
import torch

from ndae.rendering import clip_maps, i2l, l2i, split_latent_maps
from ndae.rendering.normal import height_to_normal


def test_l2i_i2l_inverse() -> None:
    latent = torch.linspace(-1.0, 1.0, steps=9)
    image = torch.linspace(0.0, 1.0, steps=9)

    assert torch.allclose(i2l(l2i(latent)), latent)
    assert torch.allclose(l2i(i2l(image)), image)


def test_split_latent_maps_returns_brdf_and_height_for_chw() -> None:
    z = torch.arange(18 * 4 * 5, dtype=torch.float32).reshape(18, 4, 5)

    brdf_maps, height_map = split_latent_maps(z, n_brdf_channels=8, n_normal_channels=1)

    assert brdf_maps.shape == (8, 4, 5)
    assert height_map.shape == (1, 4, 5)
    assert torch.allclose(brdf_maps, l2i(z[:8]))
    assert torch.equal(height_map, z[8:9])


def test_split_latent_maps_supports_leading_batch_dims() -> None:
    z = torch.randn(2, 3, 18, 6, 7)

    brdf_maps, height_map = split_latent_maps(z, n_brdf_channels=8, n_normal_channels=1)

    assert brdf_maps.shape == (2, 3, 8, 6, 7)
    assert height_map.shape == (2, 3, 1, 6, 7)


def test_split_latent_maps_discards_aug_channels() -> None:
    z = torch.zeros(18, 2, 2)
    z[9:] = 123.0

    brdf_maps, height_map = split_latent_maps(z, n_brdf_channels=8, n_normal_channels=1)

    assert brdf_maps.shape == (8, 2, 2)
    assert height_map.shape == (1, 2, 2)
    assert not torch.any(brdf_maps == 123.0)
    assert not torch.any(height_map == 123.0)


def test_split_latent_maps_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="expects a tensor shaped"):
        split_latent_maps(torch.zeros(8, 8), n_brdf_channels=8)


def test_split_latent_maps_rejects_non_positive_channel_counts() -> None:
    z = torch.zeros(9, 4, 4)

    with pytest.raises(ValueError, match="n_brdf_channels must be greater than 0"):
        split_latent_maps(z, n_brdf_channels=0, n_normal_channels=1)

    with pytest.raises(ValueError, match="n_normal_channels must be greater than 0"):
        split_latent_maps(z, n_brdf_channels=8, n_normal_channels=0)


def test_split_latent_maps_rejects_insufficient_channels() -> None:
    z = torch.zeros(8, 4, 4)

    with pytest.raises(ValueError, match="latent channels must be greater than or equal to"):
        split_latent_maps(z, n_brdf_channels=8, n_normal_channels=1)


def test_clip_maps_range() -> None:
    maps = torch.tensor([-3.0, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32)

    clipped = clip_maps(maps, eps=1e-4)

    assert clipped.min().item() == pytest.approx(1e-4)
    assert clipped.max().item() == pytest.approx(1.0)
    assert torch.allclose(clipped, torch.tensor([1e-4, 1e-4, 0.5, 1.0, 1.0]))


def test_height_to_normal_flat() -> None:
    height = torch.zeros(1, 4, 5, dtype=torch.float32)

    normal = height_to_normal(height)

    expected = torch.zeros(3, 4, 5, dtype=torch.float32)
    expected[2] = 1.0
    assert normal.shape == (3, 4, 5)
    assert torch.allclose(normal, expected, atol=1e-6)


def test_height_to_normal_gradient_x() -> None:
    x = torch.arange(5, dtype=torch.float32).view(1, 1, 5).expand(1, 4, 5)

    normal = height_to_normal(x)

    assert torch.all(normal[0] <= 0.0)
    assert torch.all(normal[2] > 0.0)
    center = normal[:, 2, 2]
    expected = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    expected = expected / expected.norm()
    assert torch.allclose(center, expected, atol=1e-5)


def test_height_to_normal_supports_leading_batch_dims() -> None:
    height = torch.zeros(2, 3, 1, 4, 5, dtype=torch.float32)

    normal = height_to_normal(height)

    assert normal.shape == (2, 3, 3, 4, 5)
    assert torch.allclose(normal[..., 0, :, :], torch.zeros(2, 3, 4, 5), atol=1e-6)
    assert torch.allclose(normal[..., 1, :, :], torch.zeros(2, 3, 4, 5), atol=1e-6)
    assert torch.allclose(normal[..., 2, :, :], torch.ones(2, 3, 4, 5), atol=1e-6)


def test_height_to_normal_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="expects a tensor shaped"):
        height_to_normal(torch.zeros(4, 5))


def test_height_to_normal_rejects_non_singleton_channel_dim() -> None:
    with pytest.raises(ValueError, match="singleton channel dimension"):
        height_to_normal(torch.zeros(2, 4, 5))


def test_height_to_normal_backward() -> None:
    height = torch.randn(2, 1, 4, 5, dtype=torch.float32, requires_grad=True)

    normal = height_to_normal(height)
    loss = normal.sum()
    loss.backward()

    assert height.grad is not None
    assert not torch.isnan(height.grad).any()
