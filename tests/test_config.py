from pathlib import Path

import pytest

from ndae.config import (
    ConfigError,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    NDAEConfig,
    TrainConfig,
    load_config,
    to_dict,
    validate_config,
)


def test_base_config_loads_into_dataclasses() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "base.yaml")

    assert isinstance(config, NDAEConfig)
    assert isinstance(config.experiment, ExperimentConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.train, TrainConfig)
    assert config.experiment.name == "lecture1_smoke"
    assert config.data.root == "data_local/svbrdf_mini"
    assert config.model.dim == 64
    assert config.train.dry_run is True


def test_to_dict_returns_plain_dictionary_tree() -> None:
    config = NDAEConfig(
        experiment=ExperimentConfig(name="demo", output_root="outputs", seed=7),
        data=DataConfig(root="data_local/svbrdf_mini", image_size=256, crop_size=128, n_frames=12),
        model=ModelConfig(dim=64, n_aug_channels=9, solver="heun"),
        train=TrainConfig(batch_size=1, lr=0.001, dry_run=True),
    )

    assert to_dict(config) == {
        "experiment": {"name": "demo", "output_root": "outputs", "seed": 7},
        "data": {
            "root": "data_local/svbrdf_mini",
            "image_size": 256,
            "crop_size": 128,
            "n_frames": 12,
        },
        "model": {"dim": 64, "n_aug_channels": 9, "solver": "heun"},
        "train": {"batch_size": 1, "lr": 0.001, "dry_run": True},
    }


def test_validate_config_rejects_invalid_semantics() -> None:
    config = NDAEConfig(
        experiment=ExperimentConfig(name="demo", output_root="outputs", seed=7),
        data=DataConfig(root="data_local/svbrdf_mini", image_size=128, crop_size=256, n_frames=12),
        model=ModelConfig(dim=64, n_aug_channels=9, solver="heun"),
        train=TrainConfig(batch_size=1, lr=0.001, dry_run=True),
    )

    with pytest.raises(ConfigError, match="crop_size"):
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
  image_size: 256
  crop_size: 128
  n_frames: 100

model:
  dim: 64
  n_aug_channels: 9
  solver: heun
  unknown: nope

train:
  batch_size: 1
  lr: 0.0005
  dry_run: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown keys: unknown"):
        load_config(config_path)
