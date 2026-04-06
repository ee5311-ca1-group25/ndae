import pytest
import torch

from ndae.rendering import clip_maps, i2l, l2i, split_latent_maps


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
