from pathlib import Path

from PIL import Image
import pytest
import torch

from ndae.cli.sample import run_sample_cli
from ndae.data import Timeline
from ndae.evaluation import build_sample_timeline, sample_sequence
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
    checkpoint_dir = save_checkpoint(workspace, trainer, saved_during_eval=True)
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


def test_sample_cli_reads_non_resume_ready_checkpoint(tmp_path: Path) -> None:
    workspace = tmp_path / "sample_workspace_nonresume"
    workspace.mkdir()
    config = make_config(
        output_root=str(tmp_path),
        name="sample_workspace_nonresume",
        n_frames=2,
        n_init_iter=0,
    )
    save_resolved_config(config, workspace)
    trainer = make_trainer(workspace, config=config)
    trainer.step()
    checkpoint_dir = save_checkpoint(workspace, trainer, saved_during_eval=True)

    run_sample_cli(["--checkpoint", str(checkpoint_dir), "--sample-size", "10"])

    assert len(list((workspace / "samples" / checkpoint_dir.name).glob("frames_*.png"))) == 2


def test_build_sample_timeline_uses_synthesis_and_transition_segments() -> None:
    timeline = Timeline.from_config(make_config(output_root="unused", n_frames=4).data)
    tensor, synthesis_frames = build_sample_timeline(timeline, dtype=torch.float32)

    assert synthesis_frames == 50
    assert tensor.shape[0] == 54
    assert tensor[0].item() == pytest.approx(timeline.t_I)
    assert tensor[49].item() < 0.0
    assert tensor[50].item() == pytest.approx(timeline.t_S)
    assert tensor[-1].item() == pytest.approx(timeline.t_E)


def test_sample_sequence_returns_synthesis_and_transition_states(tmp_path: Path) -> None:
    config = make_config(output_root=str(tmp_path), n_frames=4, n_init_iter=0)
    workspace = tmp_path / "sample_sequence"
    workspace.mkdir()
    trainer = make_trainer(workspace, config=config)
    timeline = Timeline.from_config(config.data)

    states, synthesis_frames = sample_sequence(
        trainer.system,
        timeline,
        sample_size=10,
        seed=config.experiment.seed,
    )

    assert synthesis_frames == 50
    assert states.shape == (
        synthesis_frames + timeline.n_frames,
        1,
        trainer.system.total_channels,
        10,
        10,
    )
