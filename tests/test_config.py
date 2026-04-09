from dataclasses import replace
from pathlib import Path

import pytest

from ndae.config import (
    ConfigError,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    NDAEConfig,
    RenderingConfig,
    TrainConfig,
    TrainLossConfig,
    TrainRuntimeConfig,
    TrainSchedulerConfig,
    TrainStageConfig,
    load_config,
    to_dict,
    validate_config,
)
from ndae.rendering import select_renderer


def make_rendering_config(**overrides: object) -> RenderingConfig:
    renderer_type = overrides.pop("renderer_type", "diffuse_cook_torrance")
    spec = select_renderer(str(renderer_type))
    config = RenderingConfig(
        renderer_type=spec.renderer_type,
        n_brdf_channels=spec.n_brdf_channels,
    )
    return replace(config, **overrides)


def make_config(
    *,
    root: str,
    exemplar: str = "clay_solidifying",
    image_size: int = 256,
    crop_size: int = 128,
    n_frames: int = 1,
    t_S: float = 0.0,
    t_E: float = 10.0,
    rendering: RenderingConfig | None = None,
) -> NDAEConfig:
    return NDAEConfig(
        experiment=ExperimentConfig(name="demo", output_root="outputs", seed=7),
        data=DataConfig(
            root=root,
            exemplar=exemplar,
            image_size=image_size,
            crop_size=crop_size,
            n_frames=n_frames,
            t_S=t_S,
            t_E=t_E,
        ),
        model=ModelConfig(dim=64, solver="heun"),
        rendering=rendering or make_rendering_config(),
        train=TrainConfig(
            runtime=TrainRuntimeConfig(
                batch_size=1,
                lr=0.001,
                dry_run=True,
                n_iter=2,
                log_every=1,
                checkpoint_every=1,
                resume_from=None,
            ),
            stage=TrainStageConfig(
                n_init_iter=1,
                refresh_rate_init=2,
                refresh_rate_local=6,
            ),
            loss=TrainLossConfig(
                loss_type="SW",
                n_loss_crops=32,
                overflow_weight=100.0,
                init_height_weight=1.0,
            ),
            scheduler=TrainSchedulerConfig(
                eval_every=500,
                scheduler_factor=0.5,
                scheduler_patience_evals=5,
                scheduler_min_lr=1e-4,
            ),
        ),
    )


def write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_base_config_loads_into_dataclasses() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "base.yaml")

    assert isinstance(config, NDAEConfig)
    assert isinstance(config.experiment, ExperimentConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.rendering, RenderingConfig)
    assert isinstance(config.train, TrainConfig)
    assert config.experiment.name == "smoke_run"
    assert config.data.root == "data_local/svbrdf_mini"
    assert config.data.exemplar == "clay_solidifying"
    assert config.data.n_frames == 8
    assert config.data.t_I == -2.0
    assert config.data.t_S == 0.0
    assert config.data.t_E == 10.0
    assert config.model.dim == 64
    assert config.model.solver == "heun"
    assert config.rendering.renderer_type == "diffuse_cook_torrance"
    assert config.rendering.n_brdf_channels == 8
    assert config.rendering.n_normal_channels == 1
    assert config.rendering.n_aug_channels == 9
    assert config.rendering.light_xy_position == (0.0, 0.0)
    assert config.rendering.total_channels == 18
    assert config.train.runtime.dry_run is True
    assert config.train.runtime.n_iter == 2
    assert config.train.stage.n_init_iter == 1
    assert config.train.runtime.log_every == 1
    assert config.train.runtime.checkpoint_every == 1
    assert config.train.loss.loss_type == "SW"
    assert config.train.stage.refresh_rate_init == 2
    assert config.train.stage.refresh_rate_local == 6
    assert config.train.scheduler.eval_every == 500
    assert config.train.loss.n_loss_crops == 32
    assert config.train.loss.overflow_weight == pytest.approx(100.0)
    assert config.train.loss.init_height_weight == pytest.approx(1.0)
    assert config.train.scheduler.scheduler_factor == pytest.approx(0.5)
    assert config.train.scheduler.scheduler_patience_evals == 5
    assert config.train.scheduler.scheduler_min_lr == pytest.approx(1e-4)
    assert config.train.runtime.resume_from is None


def test_debug_config_loads_without_dataset_validation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "debug.yaml",
        validate_dataset=False,
    )

    assert config.experiment.name == "debug_clay"
    assert config.data.root == "data_local/svbrdf_full"
    assert config.data.exemplar == "clay_solidifying"
    assert config.data.n_frames == 16
    assert config.train.runtime.dry_run is False
    assert config.train.runtime.n_iter == 7
    assert config.train.stage.n_init_iter == 2
    assert config.train.runtime.checkpoint_every == 1
    assert config.train.stage.refresh_rate_init == 2
    assert config.train.stage.refresh_rate_local == 6


def test_to_dict_returns_plain_dictionary_tree() -> None:
    config = NDAEConfig(
        experiment=ExperimentConfig(name="demo", output_root="outputs", seed=7),
        data=DataConfig(
            root="data_local/svbrdf_mini",
            exemplar="clay_solidifying",
            image_size=256,
            crop_size=128,
            n_frames=8,
        ),
        model=ModelConfig(dim=64, solver="heun"),
        rendering=make_rendering_config(),
        train=TrainConfig(
            runtime=TrainRuntimeConfig(
                batch_size=1,
                lr=0.001,
                dry_run=True,
                n_iter=2,
                log_every=1,
                checkpoint_every=1,
                resume_from=None,
            ),
            stage=TrainStageConfig(
                n_init_iter=1,
                refresh_rate_init=2,
                refresh_rate_local=6,
            ),
            loss=TrainLossConfig(
                loss_type="SW",
                n_loss_crops=32,
                overflow_weight=100.0,
                init_height_weight=1.0,
            ),
            scheduler=TrainSchedulerConfig(
                eval_every=500,
                scheduler_factor=0.5,
                scheduler_patience_evals=5,
                scheduler_min_lr=1e-4,
            ),
        ),
    )

    assert to_dict(config) == {
        "experiment": {"name": "demo", "output_root": "outputs", "seed": 7},
        "data": {
            "root": "data_local/svbrdf_mini",
            "exemplar": "clay_solidifying",
            "image_size": 256,
            "crop_size": 128,
            "n_frames": 8,
            "t_S": 0.0,
            "t_E": 10.0,
        },
        "model": {"dim": 64, "solver": "heun"},
        "rendering": {
            "renderer_type": "diffuse_cook_torrance",
            "n_brdf_channels": 8,
            "n_normal_channels": 1,
            "n_aug_channels": 9,
            "camera_fov": 50.0,
            "camera_distance": 1.0,
            "light_intensity": 0.0,
            "light_xy_position": (0.0, 0.0),
            "height_scale": 1.0,
            "gamma": 2.2,
        },
        "train": {
            "runtime": {
                "batch_size": 1,
                "lr": 0.001,
                "dry_run": True,
                "n_iter": 2,
                "log_every": 1,
                "checkpoint_every": 1,
                "resume_from": None,
            },
            "stage": {
                "n_init_iter": 1,
                "refresh_rate_init": 2,
                "refresh_rate_local": 6,
            },
            "loss": {
                "loss_type": "SW",
                "n_loss_crops": 32,
                "overflow_weight": 100.0,
                "init_height_weight": 1.0,
            },
            "scheduler": {
                "eval_every": 500,
                "scheduler_factor": 0.5,
                "scheduler_patience_evals": 5,
                "scheduler_min_lr": 1e-4,
            },
        },
    }


def test_full_clay_config_loads_without_dataset_validation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "full_clay.yaml",
        validate_dataset=False,
    )

    assert config.experiment.name == "full_clay_solidifying_1"
    assert config.data.root == "data_local/svbrdf_full"
    assert config.data.exemplar == "clay_solidifying"
    assert config.data.n_frames == 100
    assert config.train.runtime.dry_run is False
    assert config.train.runtime.n_iter == 3000
    assert config.train.stage.n_init_iter == 400
    assert config.train.runtime.log_every == 1
    assert config.train.runtime.checkpoint_every == 120
    assert config.train.stage.refresh_rate_init == 2
    assert config.train.stage.refresh_rate_local == 6


def test_full_clay_official_like_config_loads_without_dataset_validation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "full_clay_official_like.yaml",
        validate_dataset=False,
    )

    assert config.experiment.name == "full_clay_solidifying_official_like"
    assert config.data.root == "data_local/svbrdf_full"
    assert config.data.exemplar == "clay_solidifying"
    assert config.data.crop_size == 128
    assert config.data.n_frames == 100
    assert config.train.runtime.dry_run is False
    assert config.train.runtime.lr == 0.0005
    assert config.train.runtime.n_iter == 60000
    assert config.train.stage.n_init_iter == 20000
    assert config.train.runtime.log_every == 1
    assert config.train.runtime.checkpoint_every == 500
    assert config.train.stage.refresh_rate_init == 2
    assert config.train.stage.refresh_rate_local == 6


def test_load_config_supports_default_rendering_block(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_run
  output_root: outputs
  seed: 42

data:
  root: {tmp_path / "svbrdf_mini"}
  exemplar: clay_solidifying
  image_size: 256
  crop_size: 128
  n_frames: 1

model:
  dim: 64
  solver: heun

train:
  runtime:
    batch_size: 1
    lr: 0.0005
    dry_run: true
    n_iter: 2
    log_every: 1
    checkpoint_every: 1
  stage:
    n_init_iter: 1
  loss: {{}}
  scheduler: {{}}
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.data.t_I == -2.0
    assert config.data.t_S == 0.0
    assert config.data.t_E == 10.0
    assert config.rendering.renderer_type == "diffuse_cook_torrance"
    assert config.rendering.n_brdf_channels == 8
    assert config.rendering.total_channels == 18
    assert config.train.runtime.resume_from is None


def test_validate_config_rejects_invalid_semantics(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(
        root=str(tmp_path / "svbrdf_mini"),
        image_size=128,
        crop_size=256,
    )

    with pytest.raises(ConfigError, match="crop_size"):
        validate_config(config)


def test_validate_config_rejects_missing_exemplar_directory(tmp_path: Path) -> None:
    root = tmp_path / "svbrdf_mini"
    root.mkdir()
    config = make_config(root=str(root), exemplar="missing_exemplar")

    with pytest.raises(ConfigError, match="data.exemplar directory does not exist"):
        validate_config(config)


def test_validate_config_rejects_excessive_frame_count(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    for idx in range(2):
        write_frame(exemplar_dir / f"frame_{idx:04d}.jpg")

    config = make_config(root=str(tmp_path / "svbrdf_mini"), n_frames=3)

    with pytest.raises(ConfigError, match="data.n_frames exceeds available images"):
        validate_config(config)


def test_validate_config_prefers_manifest_selected_files_over_directory_count(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    for name in ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]:
        write_frame(exemplar_dir / name)

    (exemplar_dir / "_manifest.json").write_text(
        """
{
  "exemplar": "clay_solidifying",
  "selected_files": [
    "clay_solidifying/a.jpg",
    "clay_solidifying/b.jpg"
  ]
}
""".strip(),
        encoding="utf-8",
    )

    config = make_config(root=str(tmp_path / "svbrdf_mini"), n_frames=3)

    with pytest.raises(ConfigError, match="requested 3, found 2"):
        validate_config(config)


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
experiment:
  name: smoke_run
  output_root: outputs
  seed: 42

data:
  root: data_local/svbrdf_mini
  exemplar: clay_solidifying
  image_size: 256
  crop_size: 128
  n_frames: 8

model:
  dim: 64
  solver: heun
  unknown: nope

train:
  runtime:
    batch_size: 1
    lr: 0.0005
    dry_run: true
    n_iter: 2
    log_every: 1
    checkpoint_every: 1
  stage:
    n_init_iter: 1
  loss:
  scheduler:
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown keys: unknown"):
        load_config(config_path)


def test_load_config_rejects_legacy_model_n_aug_channels(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
experiment:
  name: smoke_run
  output_root: outputs
  seed: 42

data:
  root: data_local/svbrdf_mini
  exemplar: clay_solidifying
  image_size: 256
  crop_size: 128
  n_frames: 8

model:
  dim: 64
  n_aug_channels: 9
  solver: heun

train:
  runtime:
    batch_size: 1
    lr: 0.0005
    dry_run: true
    n_iter: 2
    log_every: 1
    checkpoint_every: 1
  stage:
    n_init_iter: 1
  loss:
  scheduler:
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown keys: n_aug_channels"):
        load_config(config_path)


def test_load_config_rejects_invalid_renderer_type(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")
    config_path = tmp_path / "invalid_renderer.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_run
  output_root: outputs
  seed: 42

data:
  root: {tmp_path / "svbrdf_mini"}
  exemplar: clay_solidifying
  image_size: 256
  crop_size: 128
  n_frames: 1

model:
  dim: 64
  solver: heun

rendering:
  renderer_type: not_a_renderer

train:
  runtime:
    batch_size: 1
    lr: 0.0005
    dry_run: true
    n_iter: 2
    log_every: 1
    checkpoint_every: 1
  stage:
    n_init_iter: 1
  loss:
  scheduler:
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="rendering.renderer_type"):
        load_config(config_path)


def test_load_config_rejects_invalid_light_xy_position(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")
    config_path = tmp_path / "invalid_light.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_run
  output_root: outputs
  seed: 42

data:
  root: {tmp_path / "svbrdf_mini"}
  exemplar: clay_solidifying
  image_size: 256
  crop_size: 128
  n_frames: 1

model:
  dim: 64
  solver: heun

rendering:
  light_xy_position: [0.0, oops]

train:
  runtime:
    batch_size: 1
    lr: 0.0005
    dry_run: true
  stage:
  loss:
  scheduler:
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="rendering.light_xy_position"):
        load_config(config_path)


def test_validate_config_rejects_invalid_rendering_channel_count(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(
        root=str(tmp_path / "svbrdf_mini"),
        rendering=make_rendering_config(
            renderer_type="diffuse_cook_torrance",
            n_brdf_channels=7,
        ),
    )

    with pytest.raises(ConfigError, match="rendering.n_brdf_channels"):
        validate_config(config)


def test_validate_config_rejects_non_finite_light_xy_position(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(
        root=str(tmp_path / "svbrdf_mini"),
        rendering=make_rendering_config(light_xy_position=(float("inf"), 0.0)),
    )

    with pytest.raises(ConfigError, match="rendering.light_xy_position"):
        validate_config(config)


def test_validate_config_rejects_invalid_timeline_ordering(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(
        root=str(tmp_path / "svbrdf_mini"),
        t_S=0.0,
        t_E=0.0,
    )

    with pytest.raises(ConfigError, match="t_I < t_S < t_E"):
        validate_config(config)


def test_validate_config_rejects_n_init_iter_greater_than_n_iter(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(root=str(tmp_path / "svbrdf_mini"))
    config.train = replace(
        config.train,
        runtime=replace(config.train.runtime, n_iter=2),
        stage=replace(config.train.stage, n_init_iter=3),
    )

    with pytest.raises(
        ConfigError,
        match="train.stage.n_init_iter must be less than or equal to train.runtime.n_iter",
    ):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "field_name", "field_value", "message"),
    [
        ("runtime", "log_every", 0, "train.runtime.log_every"),
        ("runtime", "checkpoint_every", 0, "train.runtime.checkpoint_every"),
        ("scheduler", "eval_every", 0, "train.scheduler.eval_every"),
        ("loss", "n_loss_crops", 0, "train.loss.n_loss_crops"),
        ("scheduler", "scheduler_patience_evals", 0, "train.scheduler.scheduler_patience_evals"),
    ],
)
def test_validate_config_rejects_non_positive_train_runtime_fields(
    tmp_path: Path,
    section: str,
    field_name: str,
    field_value: int,
    message: str,
) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(root=str(tmp_path / "svbrdf_mini"))
    current_section = getattr(config.train, section)
    config.train = replace(
        config.train,
        **{section: replace(current_section, **{field_name: field_value})},
    )

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_validate_config_rejects_empty_resume_from(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(root=str(tmp_path / "svbrdf_mini"))
    config.train = replace(
        config.train,
        runtime=replace(config.train.runtime, resume_from=""),
    )

    with pytest.raises(ConfigError, match="train.runtime.resume_from"):
        validate_config(config)


def test_load_config_rejects_unknown_train_keys(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config_path = tmp_path / "invalid_train.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_run
  output_root: outputs
  seed: 42

data:
  root: {tmp_path / "svbrdf_mini"}
  exemplar: clay_solidifying
  image_size: 256
  crop_size: 128
  n_frames: 1

model:
  dim: 64
  solver: heun

train:
  runtime:
    batch_size: 1
    lr: 0.0005
    dry_run: true
    n_iter: 2
    log_every: 1
    checkpoint_every: 1
  extra_train_flag: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown keys: extra_train_flag"):
        load_config(config_path)


def test_load_config_rejects_explicit_data_t_i_key(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config_path = tmp_path / "invalid_t_i.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_run
  output_root: outputs
  seed: 42

data:
  root: {tmp_path / "svbrdf_mini"}
  exemplar: clay_solidifying
  image_size: 256
  crop_size: 128
  n_frames: 1
  t_I: -2.0

model:
  dim: 64
  solver: heun

train:
  runtime:
    batch_size: 1
    lr: 0.0005
    dry_run: true
    n_iter: 2
    log_every: 1
  stage:
    n_init_iter: 1
  loss:
  scheduler:
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown keys: t_I"):
        load_config(config_path)


def test_load_config_canonicalizes_uppercase_loss_type(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config_path = tmp_path / "loss_type.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_run
  output_root: outputs
  seed: 42

data:
  root: {tmp_path / "svbrdf_mini"}
  exemplar: clay_solidifying
  image_size: 256
  crop_size: 128
  n_frames: 1

model:
  dim: 64
  solver: heun

train:
  runtime:
    batch_size: 1
    lr: 0.0005
    dry_run: true
    n_iter: 2
    log_every: 1
  stage:
    n_init_iter: 1
  loss:
    loss_type: gram
  scheduler: {{}}
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.train.loss.loss_type == "GRAM"


@pytest.mark.parametrize(
    ("section", "field_name", "field_value", "message"),
    [
        ("stage", "refresh_rate_init", 1, "train.stage.refresh_rate_init"),
        ("stage", "refresh_rate_local", 1, "train.stage.refresh_rate_local"),
        ("loss", "overflow_weight", -1.0, "train.loss.overflow_weight"),
        ("loss", "init_height_weight", -1.0, "train.loss.init_height_weight"),
        ("scheduler", "scheduler_factor", 1.0, "train.scheduler.scheduler_factor"),
        ("scheduler", "scheduler_factor", 0.0, "train.scheduler.scheduler_factor"),
    ],
)
def test_validate_config_rejects_invalid_phase1_fields(
    tmp_path: Path,
    section: str,
    field_name: str,
    field_value: object,
    message: str,
) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(root=str(tmp_path / "svbrdf_mini"))
    current_section = getattr(config.train, section)
    config.train = replace(
        config.train,
        **{section: replace(current_section, **{field_name: field_value})},
    )

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_validate_config_rejects_scheduler_min_lr_above_lr(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(root=str(tmp_path / "svbrdf_mini"))
    config.train = replace(
        config.train,
        scheduler=replace(config.train.scheduler, scheduler_min_lr=1.0),
    )

    with pytest.raises(ConfigError, match="train.scheduler.scheduler_min_lr"):
        validate_config(config)
