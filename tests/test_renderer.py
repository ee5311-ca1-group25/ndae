import pytest
import torch

from ndae.rendering import (
    Camera,
    FlashLight,
    clip_maps,
    create_meshgrid,
    diffuse_cook_torrance,
    height_to_normal,
    i2l,
    lambertian,
    l2i,
    render_svbrdf,
    split_latent_maps,
    tonemapping,
    unpack_brdf_diffuse_cook_torrance,
)


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


def test_lambertian_center_pixel() -> None:
    wi = torch.tensor([[[0.0]], [[0.0]], [[1.0]]], dtype=torch.float32)
    diffuse = torch.ones(3, 1, 1, dtype=torch.float32)

    value = lambertian(wi, diffuse)

    assert torch.allclose(value, torch.full((3, 1, 1), 1.0 / torch.pi))


def test_create_meshgrid_center_and_axes() -> None:
    positions = create_meshgrid(5, 5, Camera())

    assert positions.shape == (3, 5, 5)
    assert positions[0, 2, 2].item() == pytest.approx(0.0)
    assert positions[1, 2, 2].item() == pytest.approx(0.0)
    assert positions[2].abs().max().item() == pytest.approx(0.0)
    assert positions[1, 0, 2].item() > 0.0
    assert positions[1, -1, 2].item() < 0.0
    assert positions[0, 2, 0].item() < 0.0
    assert positions[0, 2, -1].item() > 0.0


def test_render_diffuse_only_center_pixel() -> None:
    brdf_maps = torch.zeros(8, 5, 5, dtype=torch.float32)
    brdf_maps[:3] = 1.0
    brdf_maps[6:8] = 0.5
    normal_map = height_to_normal(torch.zeros(1, 5, 5, dtype=torch.float32))

    rendered = render_svbrdf(
        brdf_maps,
        normal_map,
        Camera(),
        FlashLight(),
        diffuse_cook_torrance,
        unpack_brdf_diffuse_cook_torrance,
    )
    tone_mapped = tonemapping(rendered)

    expected_linear = torch.tensor(1.0 / torch.pi, dtype=torch.float32)
    expected_tonemapped = expected_linear.pow(torch.tensor(1.0 / 2.2, dtype=torch.float32))
    assert rendered[:, 2, 2].mean().item() == pytest.approx(expected_linear.item(), rel=1e-5)
    assert tone_mapped[:, 2, 2].mean().item() == pytest.approx(expected_tonemapped.item(), rel=1e-5)


def test_diffuse_cook_torrance_roughness_monotonicity() -> None:
    normal_map = height_to_normal(torch.zeros(1, 5, 5, dtype=torch.float32))
    low_roughness = torch.zeros(8, 5, 5, dtype=torch.float32)
    high_roughness = torch.zeros(8, 5, 5, dtype=torch.float32)
    low_roughness[3:6] = 1.0
    high_roughness[3:6] = 1.0
    low_roughness[6:8] = 0.1
    high_roughness[6:8] = 0.9

    rendered_low = render_svbrdf(
        low_roughness,
        normal_map,
        Camera(),
        FlashLight(),
        diffuse_cook_torrance,
        unpack_brdf_diffuse_cook_torrance,
    )
    rendered_high = render_svbrdf(
        high_roughness,
        normal_map,
        Camera(),
        FlashLight(),
        diffuse_cook_torrance,
        unpack_brdf_diffuse_cook_torrance,
    )

    assert rendered_low[:, 2, 2].mean().item() > rendered_high[:, 2, 2].mean().item()


def test_render_svbrdf_crop_matches_full() -> None:
    brdf_maps = torch.zeros(8, 7, 8, dtype=torch.float32)
    brdf_maps[:3] = 0.8
    brdf_maps[3:6] = 0.2
    brdf_maps[6:8] = 0.3
    normal_map = height_to_normal(torch.zeros(1, 7, 8, dtype=torch.float32))

    full = render_svbrdf(
        brdf_maps,
        normal_map,
        Camera(),
        FlashLight(intensity=0.1, xy_position=(0.1, -0.2)),
        diffuse_cook_torrance,
        unpack_brdf_diffuse_cook_torrance,
    )
    top, left, crop_h, crop_w = 2, 3, 3, 4
    crop = render_svbrdf(
        brdf_maps[:, top : top + crop_h, left : left + crop_w],
        normal_map[:, top : top + crop_h, left : left + crop_w],
        Camera(),
        FlashLight(intensity=0.1, xy_position=(0.1, -0.2)),
        diffuse_cook_torrance,
        unpack_brdf_diffuse_cook_torrance,
        full_height=7,
        full_width=8,
        region=(top, left, crop_h, crop_w),
    )

    assert torch.allclose(crop, full[:, top : top + crop_h, left : left + crop_w], atol=1e-6)


def test_render_svbrdf_explicit_positions_match_region_crop() -> None:
    brdf_maps = torch.zeros(8, 7, 8, dtype=torch.float32)
    brdf_maps[:3] = 0.8
    brdf_maps[3:6] = 0.2
    brdf_maps[6:8] = 0.3
    normal_map = height_to_normal(torch.zeros(1, 7, 8, dtype=torch.float32))
    camera = Camera()
    light = FlashLight(intensity=0.1, xy_position=(0.1, -0.2))
    top, left, crop_h, crop_w = 2, 3, 3, 4
    crop_brdf = brdf_maps[:, top : top + crop_h, left : left + crop_w]
    crop_normal = normal_map[:, top : top + crop_h, left : left + crop_w]

    with_region = render_svbrdf(
        crop_brdf,
        crop_normal,
        camera,
        light,
        diffuse_cook_torrance,
        unpack_brdf_diffuse_cook_torrance,
        full_height=7,
        full_width=8,
        region=(top, left, crop_h, crop_w),
    )
    positions = create_meshgrid(7, 8, camera)[:, top : top + crop_h, left : left + crop_w]
    with_positions = render_svbrdf(
        crop_brdf,
        crop_normal,
        camera,
        light,
        diffuse_cook_torrance,
        unpack_brdf_diffuse_cook_torrance,
        positions=positions,
    )

    assert torch.allclose(with_positions, with_region, atol=1e-6)


def test_render_svbrdf_backward_smoke() -> None:
    z = torch.randn(18, 8, 8, dtype=torch.float32, requires_grad=True)
    brdf_maps, height_map = split_latent_maps(z, n_brdf_channels=8, n_normal_channels=1)
    normal_map = height_to_normal(height_map)

    rendered = render_svbrdf(
        clip_maps(brdf_maps),
        normal_map,
        Camera(),
        FlashLight(),
        diffuse_cook_torrance,
        unpack_brdf_diffuse_cook_torrance,
    )
    loss = tonemapping(rendered).sum()
    loss.backward()

    assert z.grad is not None
    assert not torch.isnan(z.grad).any()
