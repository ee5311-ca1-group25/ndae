from pathlib import Path

from PIL import Image
import pytest

from ndae.cli.sample import run_sample_cli
from ndae.training import save_checkpoint
from ndae.utils import save_resolved_config

from tests.support import make_config, make_trainer


def _create_sample_checkpoint(
    tmp_path: Path,
    *,
    n_frames: int = 3,
) -> tuple[Path, Path]:
    workspace = tmp_path / "sample_workspace"
    workspace.mkdir()
    config = make_config(
        output_root=str(tmp_path),
        name="sample_workspace",
        n_frames=n_frames,
        n_init_iter=0,
    )
    save_resolved_config(config, workspace)
    trainer = make_trainer(workspace, config=config)
    for _ in range(3):
        trainer.step()
    checkpoint_dir = save_checkpoint(workspace, trainer)
    return workspace, checkpoint_dir


def test_sample_cli_writes_png_sequence_from_checkpoint(tmp_path: Path) -> None:
    workspace, checkpoint_dir = _create_sample_checkpoint(tmp_path, n_frames=3)

    exit_code = run_sample_cli(["--checkpoint", str(checkpoint_dir), "--sample-size", "12"])

    assert exit_code == 0
    output_dir = workspace / "samples" / checkpoint_dir.name
    assert output_dir.is_dir()
    assert [path.name for path in sorted(output_dir.glob("frames_*.png"))] == [
        "frames_0000.png",
        "frames_0001.png",
        "frames_0002.png",
    ]


def test_sample_cli_requires_explicit_sample_size(tmp_path: Path) -> None:
    _, checkpoint_dir = _create_sample_checkpoint(tmp_path, n_frames=2)

    with pytest.raises(SystemExit, match="2"):
        run_sample_cli(["--checkpoint", str(checkpoint_dir)])


def test_sample_cli_uses_explicit_sample_size(tmp_path: Path) -> None:
    workspace, checkpoint_dir = _create_sample_checkpoint(tmp_path, n_frames=2)

    run_sample_cli(["--checkpoint", str(checkpoint_dir), "--sample-size", "14"])

    with Image.open(workspace / "samples" / checkpoint_dir.name / "frames_0000.png") as image:
        assert image.size == (14, 14)


def test_sample_cli_accepts_explicit_output_dir(tmp_path: Path) -> None:
    _, checkpoint_dir = _create_sample_checkpoint(tmp_path, n_frames=2)
    output_dir = tmp_path / "custom_samples"

    exit_code = run_sample_cli(
        [
            "--checkpoint",
            str(checkpoint_dir),
            "--output-dir",
            str(output_dir),
            "--sample-size",
            "10",
        ]
    )

    assert exit_code == 0
    assert output_dir.is_dir()
    assert len(list(output_dir.glob("frames_*.png"))) == 2


def test_sample_cli_requires_flashlight_checkpoint_state(tmp_path: Path) -> None:
    _, checkpoint_dir = _create_sample_checkpoint(tmp_path, n_frames=2)
    (checkpoint_dir / "flashlight.pt").unlink()

    with pytest.raises(FileNotFoundError, match="flashlight state"):
        run_sample_cli(["--checkpoint", str(checkpoint_dir), "--sample-size", "10"])
