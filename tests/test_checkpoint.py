import json
from pathlib import Path

import pytest
import torch

from ndae.training import load_resume_checkpoint, save_checkpoint

from tests.support import make_config, make_trainer


def _advance_to_boundary(trainer, *, n_steps: int = 3) -> None:
    for _ in range(n_steps):
        trainer.step()


def test_save_checkpoint_writes_step_and_latest_layout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trainer = make_trainer(workspace, config=make_config(output_root=str(tmp_path), name="workspace", n_init_iter=0))

    _advance_to_boundary(trainer)
    checkpoint_dir = save_checkpoint(workspace, trainer)

    latest_dir = workspace / "checkpoints" / "latest"
    assert checkpoint_dir == workspace / "checkpoints" / "step_000003"
    for directory in (checkpoint_dir, latest_dir):
        assert (directory / "flashlight.pt").is_file()
        assert (directory / "model.pt").is_file()
        assert (directory / "optimizer.pt").is_file()
        assert (directory / "trainer_state.pt").is_file()
        assert (directory / "meta.json").is_file()

    meta = json.loads((checkpoint_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["step"] == 3
    assert meta["resume_ready"] is True
    assert meta["saved_at_refresh_boundary"] is True
    assert meta["cycle_step"] == 0
    assert "flashlight.pt" in meta["files"]


def test_resume_checkpoint_round_trip_restores_model_optimizer_and_trainer_state(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = make_config(output_root=str(tmp_path), name="workspace", n_init_iter=0)
    trainer = make_trainer(workspace, config=config)

    _advance_to_boundary(trainer)
    checkpoint_dir = save_checkpoint(workspace, trainer)
    expected_model = {
        name: tensor.detach().clone()
        for name, tensor in trainer.trajectory_model.state_dict().items()
    }
    expected_optimizer = trainer.optimizer.state_dict()
    expected_state = trainer.state
    expected_flashlight = trainer.system.flash_light.intensity.detach().clone()

    restored_trainer = make_trainer(workspace, config=config)
    restored_state = load_resume_checkpoint(checkpoint_dir, restored_trainer)

    for name, tensor in restored_trainer.trajectory_model.state_dict().items():
        assert torch.allclose(tensor, expected_model[name])
    assert restored_trainer.optimizer.state_dict()["state"].keys() == expected_optimizer["state"].keys()
    assert restored_state.global_step == expected_state.global_step
    assert restored_state.stage == expected_state.stage
    assert restored_state.cycle_step == expected_state.cycle_step
    assert restored_state.carry_time == pytest.approx(expected_state.carry_time)
    assert restored_state.carry_state is not None
    assert expected_state.carry_state is not None
    assert torch.allclose(restored_state.carry_state, expected_state.carry_state)
    assert torch.allclose(restored_trainer.system.flash_light.intensity.detach(), expected_flashlight)


def test_load_resume_checkpoint_rejects_non_resume_ready_meta(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trainer = make_trainer(workspace, config=make_config(output_root=str(tmp_path), name="workspace", n_init_iter=0))

    trainer.step()
    checkpoint_dir = save_checkpoint(workspace, trainer)
    restored_trainer = make_trainer(workspace, config=make_config(output_root=str(tmp_path), name="workspace", n_init_iter=0))

    with pytest.raises(ValueError, match="not resume-ready"):
        load_resume_checkpoint(checkpoint_dir, restored_trainer)


def test_load_resume_checkpoint_requires_flashlight_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = make_config(output_root=str(tmp_path), name="workspace", n_init_iter=0)
    trainer = make_trainer(workspace, config=config)

    _advance_to_boundary(trainer)
    checkpoint_dir = save_checkpoint(workspace, trainer)
    (checkpoint_dir / "flashlight.pt").unlink()

    restored_trainer = make_trainer(workspace, config=config)

    with pytest.raises(FileNotFoundError, match="flashlight state"):
        load_resume_checkpoint(checkpoint_dir, restored_trainer)
