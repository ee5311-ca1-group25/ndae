import importlib
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_NAMES = [
    "ndae",
    "ndae.config",
    "ndae.cli",
    "ndae.data",
    "ndae.models",
    "ndae.rendering",
    "ndae.losses",
    "ndae.training",
    "ndae.evaluation",
    "ndae.utils",
]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_package_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)


def test_data_package_exports_public_api() -> None:
    from ndae.data import (
        ExemplarDataset,
        Timeline,
        random_crop,
        random_take,
        sample_frame_indices,
        stratified_uniform,
    )
    from ndae.data.exemplar import ExemplarDataset as ExemplarDatasetImpl
    from ndae.data.sampling import (
        random_crop as random_crop_impl,
        random_take as random_take_impl,
        sample_frame_indices as sample_frame_indices_impl,
        stratified_uniform as stratified_uniform_impl,
    )
    from ndae.data.timeline import Timeline as TimelineImpl

    assert ExemplarDataset is ExemplarDatasetImpl
    assert Timeline is TimelineImpl
    assert random_crop is random_crop_impl
    assert random_take is random_take_impl
    assert stratified_uniform is stratified_uniform_impl
    assert sample_frame_indices is sample_frame_indices_impl


def test_rendering_package_exports_renderer_metadata() -> None:
    from ndae.rendering import (
        RENDERER_REGISTRY,
        RendererSpec,
        clip_maps,
        i2l,
        l2i,
        select_renderer,
        split_latent_maps,
    )
    from ndae.rendering.maps import (
        clip_maps as clip_maps_impl,
        i2l as i2l_impl,
        l2i as l2i_impl,
        split_latent_maps as split_latent_maps_impl,
    )

    renderer = select_renderer("diffuse_cook_torrance")

    assert isinstance(renderer, RendererSpec)
    assert renderer is RENDERER_REGISTRY["diffuse_cook_torrance"]
    assert renderer.n_brdf_channels == 8
    assert set(RENDERER_REGISTRY) == {
        "diffuse_cook_torrance",
        "diffuse_iso_cook_torrance",
        "cook_torrance",
        "iso_cook_torrance",
        "compl_cook_torrance",
        "compl_iso_cook_torrance",
    }
    assert l2i is l2i_impl
    assert i2l is i2l_impl
    assert split_latent_maps is split_latent_maps_impl
    assert clip_maps is clip_maps_impl


def test_rendering_normal_module_imports() -> None:
    module = importlib.import_module("ndae.rendering.normal")

    assert module.height_to_normal is not None


def test_rendering_renderer_module_imports() -> None:
    module = importlib.import_module("ndae.rendering.renderer")

    assert module.render_svbrdf is not None


def test_rendering_geometry_module_imports() -> None:
    module = importlib.import_module("ndae.rendering.geometry")
    from ndae.rendering.renderer import Camera, create_meshgrid

    assert module.Camera is Camera
    assert module.create_meshgrid is create_meshgrid


def test_rendering_brdf_module_imports() -> None:
    module = importlib.import_module("ndae.rendering.brdf")
    from ndae.rendering.renderer import diffuse_cook_torrance, lambertian

    assert module.lambertian is lambertian
    assert module.diffuse_cook_torrance is diffuse_cook_torrance


def test_train_cli_stub_returns_success(capsys: pytest.CaptureFixture[str]) -> None:
    from ndae.cli.train import run_train_cli

    assert run_train_cli(["--config", "configs/base.yaml", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "NDAE Lecture 1 Dry Run" in output
    assert "rendering.renderer_type: diffuse_cook_torrance" in output
    assert "Dry run completed." in output


def test_main_entrypoint_smoke() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "main.py", "--config", "configs/base.yaml", "--dry-run"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "NDAE Lecture 1 Dry Run" in result.stdout
    assert "Dry run completed." in result.stdout
