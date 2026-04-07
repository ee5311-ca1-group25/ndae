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


def test_losses_package_exports_public_api() -> None:
    from ndae.losses import (
        VGG19Features,
        gram_loss,
        gram_matrix,
        init_loss,
        local_loss,
        overflow_loss,
        slice_loss,
        sliced_wasserstein_loss,
    )
    from ndae.losses.objectives import (
        init_loss as init_loss_impl,
        local_loss as local_loss_impl,
        overflow_loss as overflow_loss_impl,
    )
    from ndae.losses.perceptual import VGG19Features as VGG19FeaturesImpl
    from ndae.losses.swd import (
        gram_loss as gram_loss_impl,
        gram_matrix as gram_matrix_impl,
        slice_loss as slice_loss_impl,
        sliced_wasserstein_loss as sliced_wasserstein_loss_impl,
    )

    assert VGG19Features is VGG19FeaturesImpl
    assert gram_matrix is gram_matrix_impl
    assert gram_loss is gram_loss_impl
    assert sliced_wasserstein_loss is sliced_wasserstein_loss_impl
    assert slice_loss is slice_loss_impl
    assert overflow_loss is overflow_loss_impl
    assert init_loss is init_loss_impl
    assert local_loss is local_loss_impl


def test_models_package_exports_public_api() -> None:
    from ndae.models import (
        ConvBlock,
        DefaultConv2d,
        LinearTimeSelfAttention,
        NDAEUNet,
        ODEFunction,
        Residual,
        Resample,
        SinusoidalTimeEmbedding,
        SpatialLinear,
        TimeMLP,
        TrajectoryModel,
        zero_init,
    )
    from ndae.models.blocks import (
        ConvBlock as ConvBlockImpl,
        DefaultConv2d as DefaultConv2dImpl,
        LinearTimeSelfAttention as LinearTimeSelfAttentionImpl,
        Residual as ResidualImpl,
        Resample as ResampleImpl,
        SpatialLinear as SpatialLinearImpl,
        zero_init as zero_init_impl,
    )
    from ndae.models.odefunc import ODEFunction as ODEFunctionImpl
    from ndae.models.time_embedding import (
        SinusoidalTimeEmbedding as SinusoidalTimeEmbeddingImpl,
        TimeMLP as TimeMLPImpl,
    )
    from ndae.models.trajectory import TrajectoryModel as TrajectoryModelImpl
    from ndae.models.unet import NDAEUNet as NDAEUNetImpl

    assert NDAEUNet is NDAEUNetImpl
    assert ODEFunction is ODEFunctionImpl
    assert TrajectoryModel is TrajectoryModelImpl
    assert SinusoidalTimeEmbedding is SinusoidalTimeEmbeddingImpl
    assert TimeMLP is TimeMLPImpl
    assert ConvBlock is ConvBlockImpl
    assert DefaultConv2d is DefaultConv2dImpl
    assert SpatialLinear is SpatialLinearImpl
    assert Resample is ResampleImpl
    assert LinearTimeSelfAttention is LinearTimeSelfAttentionImpl
    assert Residual is ResidualImpl
    assert zero_init is zero_init_impl


def test_training_package_exports_public_api() -> None:
    from ndae.training import (
        RefreshSchedule,
        RolloutResult,
        RolloutWindow,
        SolverConfig,
        SVBRDFSystem,
        StageConfig,
        Trainer,
        TrainerComponents,
        TrainerConfig,
        TrainerState,
        build_svbrdf_system,
        build_trainer,
        load_resume_checkpoint,
        load_sample_checkpoint,
        render_latent_state,
        resolve_checkpoint_dir,
        save_checkpoint,
        rollout_generation,
        rollout_warmup,
        solve_rollout,
    )
    from ndae.training.checkpoint import (
        load_resume_checkpoint as load_resume_checkpoint_impl,
        load_sample_checkpoint as load_sample_checkpoint_impl,
        resolve_checkpoint_dir as resolve_checkpoint_dir_impl,
        save_checkpoint as save_checkpoint_impl,
    )
    from ndae.training.schedule import (
        RefreshSchedule as RefreshScheduleImpl,
        RolloutWindow as RolloutWindowImpl,
        StageConfig as StageConfigImpl,
    )
    from ndae.training.solver import (
        RolloutResult as RolloutResultImpl,
        SolverConfig as SolverConfigImpl,
        rollout_generation as rollout_generation_impl,
        rollout_warmup as rollout_warmup_impl,
        solve_rollout as solve_rollout_impl,
    )
    from ndae.training.system import (
        SVBRDFSystem as SVBRDFSystemImpl,
        build_svbrdf_system as build_svbrdf_system_impl,
        render_latent_state as render_latent_state_impl,
    )
    from ndae.training.factory import build_trainer as build_trainer_impl
    from ndae.training.trainer import (
        Trainer as TrainerImpl,
        TrainerComponents as TrainerComponentsImpl,
        TrainerConfig as TrainerConfigImpl,
        TrainerState as TrainerStateImpl,
    )

    assert SVBRDFSystem is SVBRDFSystemImpl
    assert build_svbrdf_system is build_svbrdf_system_impl
    assert render_latent_state is render_latent_state_impl
    assert build_trainer is build_trainer_impl
    assert SolverConfig is SolverConfigImpl
    assert RolloutResult is RolloutResultImpl
    assert solve_rollout is solve_rollout_impl
    assert rollout_warmup is rollout_warmup_impl
    assert rollout_generation is rollout_generation_impl
    assert StageConfig is StageConfigImpl
    assert RolloutWindow is RolloutWindowImpl
    assert RefreshSchedule is RefreshScheduleImpl
    assert resolve_checkpoint_dir is resolve_checkpoint_dir_impl
    assert save_checkpoint is save_checkpoint_impl
    assert load_resume_checkpoint is load_resume_checkpoint_impl
    assert load_sample_checkpoint is load_sample_checkpoint_impl
    assert Trainer is TrainerImpl
    assert TrainerComponents is TrainerComponentsImpl
    assert TrainerConfig is TrainerConfigImpl
    assert TrainerState is TrainerStateImpl


def test_rendering_package_exports_renderer_metadata() -> None:
    from ndae.rendering import (
        EPSILON,
        Camera,
        FlashLight,
        RENDERER_REGISTRY,
        RendererSpec,
        channelwise_normalize,
        clip_maps,
        compute_directions,
        cook_torrance,
        create_meshgrid,
        diffuse_cook_torrance,
        diffuse_iso_cook_torrance,
        distribution_ggx,
        fresnel_schlick,
        geometry_smith,
        height_to_normal,
        i2l,
        lambertian,
        l2i,
        light_decay,
        localize,
        localize_wiwo,
        normalize,
        reinhard,
        render_svbrdf,
        select_renderer,
        smith_g1_ggx,
        split_latent_maps,
        tonemapping,
        unpack_brdf_diffuse_cook_torrance,
        unpack_brdf_diffuse_iso_cook_torrance,
    )
    from ndae.rendering.maps import (
        clip_maps as clip_maps_impl,
        i2l as i2l_impl,
        l2i as l2i_impl,
        split_latent_maps as split_latent_maps_impl,
    )
    from ndae.rendering.normal import height_to_normal as height_to_normal_impl
    from ndae.rendering.geometry import (
        EPSILON as epsilon_impl,
        Camera as camera_impl,
        FlashLight as flash_light_impl,
        channelwise_normalize as channelwise_normalize_impl,
        compute_directions as compute_directions_impl,
        create_meshgrid as create_meshgrid_impl,
        localize as localize_impl,
        localize_wiwo as localize_wiwo_impl,
        normalize as normalize_impl,
    )
    from ndae.rendering.brdf import (
        cook_torrance as cook_torrance_impl,
        diffuse_cook_torrance as diffuse_cook_torrance_impl,
        diffuse_iso_cook_torrance as diffuse_iso_cook_torrance_impl,
        distribution_ggx as distribution_ggx_impl,
        fresnel_schlick as fresnel_schlick_impl,
        geometry_smith as geometry_smith_impl,
        lambertian as lambertian_impl,
        smith_g1_ggx as smith_g1_ggx_impl,
    )
    from ndae.rendering.postprocess import (
        light_decay as light_decay_impl,
        reinhard as reinhard_impl,
        tonemapping as tonemapping_impl,
    )
    from ndae.rendering.renderer import (
        render_svbrdf as render_svbrdf_impl,
        unpack_brdf_diffuse_cook_torrance as unpack_brdf_diffuse_cook_torrance_impl,
        unpack_brdf_diffuse_iso_cook_torrance as unpack_brdf_diffuse_iso_cook_torrance_impl,
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
    assert EPSILON == epsilon_impl
    assert Camera is camera_impl
    assert FlashLight is flash_light_impl
    assert l2i is l2i_impl
    assert i2l is i2l_impl
    assert split_latent_maps is split_latent_maps_impl
    assert clip_maps is clip_maps_impl
    assert height_to_normal is height_to_normal_impl
    assert normalize is normalize_impl
    assert channelwise_normalize is channelwise_normalize_impl
    assert create_meshgrid is create_meshgrid_impl
    assert compute_directions is compute_directions_impl
    assert localize is localize_impl
    assert localize_wiwo is localize_wiwo_impl
    assert lambertian is lambertian_impl
    assert distribution_ggx is distribution_ggx_impl
    assert smith_g1_ggx is smith_g1_ggx_impl
    assert geometry_smith is geometry_smith_impl
    assert fresnel_schlick is fresnel_schlick_impl
    assert cook_torrance is cook_torrance_impl
    assert diffuse_cook_torrance is diffuse_cook_torrance_impl
    assert diffuse_iso_cook_torrance is diffuse_iso_cook_torrance_impl
    assert render_svbrdf is render_svbrdf_impl
    assert unpack_brdf_diffuse_cook_torrance is unpack_brdf_diffuse_cook_torrance_impl
    assert unpack_brdf_diffuse_iso_cook_torrance is unpack_brdf_diffuse_iso_cook_torrance_impl
    assert tonemapping is tonemapping_impl
    assert light_decay is light_decay_impl
    assert reinhard is reinhard_impl


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


def test_rendering_postprocess_module_imports() -> None:
    module = importlib.import_module("ndae.rendering.postprocess")
    from ndae.rendering.renderer import light_decay, tonemapping

    assert module.tonemapping is tonemapping
    assert module.light_decay is light_decay


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
