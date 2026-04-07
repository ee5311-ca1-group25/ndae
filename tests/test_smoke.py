import numpy as np
from pathlib import Path

from PIL import Image
import pytest
import torch
import torch.nn as nn

import ndae.cli.train as train_cli
from ndae.cli.render_example import run_render_example_cli
from ndae.cli.train import run_train_cli
from tests.support import write_image


def test_dry_run_creates_workspace_and_resolved_config(tmp_path: Path, capsys) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "base.yaml"

    exit_code = run_train_cli(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0

    workspace = tmp_path / "lecture1_smoke"
    resolved_config = workspace / "config.resolved.yaml"

    assert workspace.is_dir()
    assert resolved_config.is_file()

    output = capsys.readouterr().out
    assert "workspace:" in output
    assert str(workspace) in output
    assert "data.exemplar: clay_solidifying" in output
    assert "rendering.renderer_type: diffuse_cook_torrance" in output
    assert "rendering.total_channels: 18" in output
    assert "Dry run completed." in output

    resolved_text = resolved_config.read_text(encoding="utf-8")
    assert "lecture1_smoke" in resolved_text
    assert "exemplar: clay_solidifying" in resolved_text
    assert "rendering:" in resolved_text
    assert "renderer_type: diffuse_cook_torrance" in resolved_text
    assert "n_brdf_channels: 8" in resolved_text
    assert f"output_root: '{tmp_path}'" not in resolved_text
    assert f"output_root: {tmp_path}" in resolved_text


@pytest.mark.parametrize("preset", ["plastic", "coated_metal"])
def test_render_example_cli_writes_png(tmp_path: Path, capsys, preset: str) -> None:
    output_path = tmp_path / f"{preset}.png"

    exit_code = run_render_example_cli(
        [
            "--preset",
            preset,
            "--output",
            str(output_path),
            "--image-size",
            "48",
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()

    with Image.open(output_path) as image:
        pixels = np.asarray(image)
        assert image.size == (48, 48)
        assert pixels.shape[0] == 48
        assert pixels.shape[1] == 48
        assert float(pixels.std()) > 0.0

    output = capsys.readouterr().out
    assert "Rendered synthetic svBRDF example." in output
    assert f"preset: {preset}" in output
    assert str(output_path) in output


class DummyFeatures(nn.Module):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x]


def test_non_dry_run_cli_executes_trainer_and_writes_metrics(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_cli, "VGG19Features", DummyFeatures)

    data_root = tmp_path / "data_root"
    exemplar_dir = data_root / "example"
    write_image(exemplar_dir / "frame_0000.png", size=(24, 24), color=(20, 40, 60))
    write_image(exemplar_dir / "frame_0001.png", size=(24, 24), color=(60, 80, 100))

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""
experiment:
  name: lecture8_cli_smoke
  output_root: {tmp_path}
  seed: 7

data:
  root: {data_root}
  exemplar: example
  image_size: 16
  crop_size: 8
  n_frames: 2
  t_I: -2.0
  t_S: 0.0
  t_E: 2.0

model:
  dim: 8
  solver: euler

rendering:
  renderer_type: diffuse_cook_torrance
  n_normal_channels: 1
  n_aug_channels: 9
  camera_fov: 50.0
  camera_distance: 1.0
  light_intensity: 0.0
  light_xy_position: [0.0, 0.0]
  height_scale: 1.0
  gamma: 2.2

train:
  batch_size: 1
  lr: 0.001
  dry_run: false
  n_iter: 2
  n_init_iter: 1
  log_every: 1
  checkpoint_every: 1
  sample_every: 1
  sample_size: 8
  resume_from: null
""".strip(),
        encoding="utf-8",
    )

    exit_code = run_train_cli(["--config", str(config_path)])

    assert exit_code == 0

    workspace = tmp_path / "lecture8_cli_smoke"
    metrics_path = workspace / "metrics.jsonl"
    resolved_config = workspace / "config.resolved.yaml"

    assert workspace.is_dir()
    assert resolved_config.is_file()
    assert metrics_path.is_file()
    assert len(metrics_path.read_text(encoding="utf-8").strip().splitlines()) == 2

    output = capsys.readouterr().out
    assert "workspace:" in output
    assert str(workspace) in output
    assert "Training completed." in output


def test_train_cli_writes_boundary_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_cli, "VGG19Features", DummyFeatures)

    data_root = tmp_path / "data_root"
    exemplar_dir = data_root / "example"
    write_image(exemplar_dir / "frame_0000.png", size=(24, 24), color=(20, 40, 60))
    write_image(exemplar_dir / "frame_0001.png", size=(24, 24), color=(60, 80, 100))

    config_path = tmp_path / "checkpoint_train.yaml"
    config_path.write_text(
        f"""
experiment:
  name: checkpoint_smoke
  output_root: {tmp_path}
  seed: 7

data:
  root: {data_root}
  exemplar: example
  image_size: 16
  crop_size: 8
  n_frames: 2
  t_I: -2.0
  t_S: 0.0
  t_E: 2.0

model:
  dim: 8
  solver: euler

rendering:
  renderer_type: diffuse_cook_torrance
  n_normal_channels: 1
  n_aug_channels: 9
  camera_fov: 50.0
  camera_distance: 1.0
  light_intensity: 0.0
  light_xy_position: [0.0, 0.0]
  height_scale: 1.0
  gamma: 2.2

train:
  batch_size: 1
  lr: 0.001
  dry_run: false
  n_iter: 7
  n_init_iter: 1
  log_every: 1
  checkpoint_every: 1
  sample_every: 1
  sample_size: 8
  resume_from: null
""".strip(),
        encoding="utf-8",
    )

    exit_code = run_train_cli(["--config", str(config_path)])

    assert exit_code == 0
    workspace = tmp_path / "checkpoint_smoke"
    latest = workspace / "checkpoints" / "latest"
    assert (latest / "model.pt").is_file()
    assert (latest / "optimizer.pt").is_file()
    assert (latest / "trainer_state.pt").is_file()
    assert (latest / "meta.json").is_file()


def test_resume_cli_appends_metrics_jsonl(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_cli, "VGG19Features", DummyFeatures)

    data_root = tmp_path / "data_root"
    exemplar_dir = data_root / "example"
    write_image(exemplar_dir / "frame_0000.png", size=(24, 24), color=(20, 40, 60))
    write_image(exemplar_dir / "frame_0001.png", size=(24, 24), color=(60, 80, 100))

    first_config_path = tmp_path / "resume_first.yaml"
    first_config_path.write_text(
        f"""
experiment:
  name: resume_smoke
  output_root: {tmp_path}
  seed: 7

data:
  root: {data_root}
  exemplar: example
  image_size: 16
  crop_size: 8
  n_frames: 2
  t_I: -2.0
  t_S: 0.0
  t_E: 2.0

model:
  dim: 8
  solver: euler

rendering:
  renderer_type: diffuse_cook_torrance
  n_normal_channels: 1
  n_aug_channels: 9
  camera_fov: 50.0
  camera_distance: 1.0
  light_intensity: 0.0
  light_xy_position: [0.0, 0.0]
  height_scale: 1.0
  gamma: 2.2

train:
  batch_size: 1
  lr: 0.001
  dry_run: false
  n_iter: 7
  n_init_iter: 1
  log_every: 1
  checkpoint_every: 1
  sample_every: 1
  sample_size: 8
  resume_from: null
""".strip(),
        encoding="utf-8",
    )

    assert run_train_cli(["--config", str(first_config_path)]) == 0

    workspace = tmp_path / "resume_smoke"
    metrics_path = workspace / "metrics.jsonl"
    assert len(metrics_path.read_text(encoding="utf-8").strip().splitlines()) == 7

    second_config_path = tmp_path / "resume_second.yaml"
    second_config_path.write_text(
        f"""
experiment:
  name: resume_smoke
  output_root: {tmp_path}
  seed: 7

data:
  root: {data_root}
  exemplar: example
  image_size: 16
  crop_size: 8
  n_frames: 2
  t_I: -2.0
  t_S: 0.0
  t_E: 2.0

model:
  dim: 8
  solver: euler

rendering:
  renderer_type: diffuse_cook_torrance
  n_normal_channels: 1
  n_aug_channels: 9
  camera_fov: 50.0
  camera_distance: 1.0
  light_intensity: 0.0
  light_xy_position: [0.0, 0.0]
  height_scale: 1.0
  gamma: 2.2

train:
  batch_size: 1
  lr: 0.001
  dry_run: false
  n_iter: 1
  n_init_iter: 1
  log_every: 1
  checkpoint_every: 1
  sample_every: 1
  sample_size: 8
  resume_from: {workspace / "checkpoints" / "latest"}
""".strip(),
        encoding="utf-8",
    )

    assert run_train_cli(["--config", str(second_config_path)]) == 0
    assert len(metrics_path.read_text(encoding="utf-8").strip().splitlines()) == 8

    output = capsys.readouterr().out
    assert "Resumed from checkpoint:" in output
