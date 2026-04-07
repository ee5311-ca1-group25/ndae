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
    t_I: float = -2.0,
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
            t_I=t_I,
            t_S=t_S,
            t_E=t_E,
        ),
        model=ModelConfig(dim=64, solver="heun"),
        rendering=rendering or make_rendering_config(),
        train=TrainConfig(
            batch_size=1,
            lr=0.001,
            dry_run=True,
            n_iter=2,
            n_init_iter=1,
            log_every=1,
            checkpoint_every=1,
            sample_every=1,
            sample_size=64,
            resume_from=None,
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
    assert config.experiment.name == "lecture1_smoke"
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
    assert config.train.dry_run is True
    assert config.train.n_iter == 2
    assert config.train.n_init_iter == 1
    assert config.train.log_every == 1
    assert config.train.checkpoint_every == 1
    assert config.train.sample_every == 1
    assert config.train.sample_size == 64
    assert config.train.resume_from is None


def test_debug_config_loads_without_dataset_validation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "debug.yaml",
        validate_dataset=False,
    )

    assert config.experiment.name == "lecture09_debug_clay"
    assert config.data.root == "data_local/svbrdf_full"
    assert config.data.exemplar == "clay_solidifying"
    assert config.data.n_frames == 16
    assert config.train.dry_run is False
    assert config.train.n_iter == 7
    assert config.train.n_init_iter == 2
    assert config.train.checkpoint_every == 1
    assert config.train.sample_size == 64


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
            batch_size=1,
            lr=0.001,
            dry_run=True,
            n_iter=2,
            n_init_iter=1,
            log_every=1,
            checkpoint_every=1,
            sample_every=1,
            sample_size=64,
            resume_from=None,
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
            "t_I": -2.0,
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
            "batch_size": 1,
            "lr": 0.001,
            "dry_run": True,
            "n_iter": 2,
            "n_init_iter": 1,
            "log_every": 1,
            "checkpoint_every": 1,
            "sample_every": 1,
            "sample_size": 64,
            "resume_from": None,
        },
    }


def test_load_config_supports_default_rendering_block(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        f"""
experiment:
  name: lecture1_smoke
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
  batch_size: 1
  lr: 0.0005
  dry_run: true
  n_iter: 2
  n_init_iter: 1
  log_every: 1
  checkpoint_every: 1
  sample_every: 1
  sample_size: 64
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
    assert config.train.resume_from is None


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
  name: lecture1_smoke
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
  batch_size: 1
  lr: 0.0005
  dry_run: true
  n_iter: 2
  n_init_iter: 1
  log_every: 1
  checkpoint_every: 1
  sample_every: 1
  sample_size: 64
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
  name: lecture1_smoke
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
  batch_size: 1
  lr: 0.0005
  dry_run: true
  n_iter: 2
  n_init_iter: 1
  log_every: 1
  checkpoint_every: 1
  sample_every: 1
  sample_size: 64
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
  name: lecture1_smoke
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
  batch_size: 1
  lr: 0.0005
  dry_run: true
  n_iter: 2
  n_init_iter: 1
  log_every: 1
  checkpoint_every: 1
  sample_every: 1
  sample_size: 64
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
  name: lecture1_smoke
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
  batch_size: 1
  lr: 0.0005
  dry_run: true
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
        t_I=0.0,
        t_S=0.0,
        t_E=1.0,
    )

    with pytest.raises(ConfigError, match="t_I < t_S < t_E"):
        validate_config(config)


def test_validate_config_rejects_n_init_iter_greater_than_n_iter(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(root=str(tmp_path / "svbrdf_mini"))
    config.train = replace(config.train, n_iter=2, n_init_iter=3)

    with pytest.raises(ConfigError, match="train.n_init_iter must be less than or equal to train.n_iter"):
        validate_config(config)


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("log_every", 0, "train.log_every"),
        ("checkpoint_every", 0, "train.checkpoint_every"),
        ("sample_every", 0, "train.sample_every"),
        ("sample_size", 0, "train.sample_size"),
    ],
)
def test_validate_config_rejects_non_positive_train_runtime_fields(
    tmp_path: Path,
    field_name: str,
    field_value: int,
    message: str,
) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(root=str(tmp_path / "svbrdf_mini"))
    config.train = replace(config.train, **{field_name: field_value})

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_validate_config_rejects_empty_resume_from(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config = make_config(root=str(tmp_path / "svbrdf_mini"))
    config.train = replace(config.train, resume_from="")

    with pytest.raises(ConfigError, match="train.resume_from"):
        validate_config(config)


def test_load_config_rejects_unknown_train_keys(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "svbrdf_mini" / "clay_solidifying"
    write_frame(exemplar_dir / "frame_0000.jpg")

    config_path = tmp_path / "invalid_train.yaml"
    config_path.write_text(
        f"""
experiment:
  name: lecture1_smoke
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
  batch_size: 1
  lr: 0.0005
  dry_run: true
  n_iter: 2
  n_init_iter: 1
  log_every: 1
  checkpoint_every: 1
  sample_every: 1
  sample_size: 64
  extra_train_flag: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown keys: extra_train_flag"):
        load_config(config_path)
