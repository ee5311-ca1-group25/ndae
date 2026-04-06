import numpy as np
from pathlib import Path

from PIL import Image
import pytest

from ndae.cli.render_example import run_render_example_cli
from ndae.cli.train import run_train_cli


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
