"""Checkpoint helpers for training and sampling."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .schedule import RefreshSchedule
from .state import TrainerState

if TYPE_CHECKING:
    from .trainer import Trainer


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Resolve and validate a concrete checkpoint directory."""
    checkpoint_dir = Path(path).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint path must be a directory: {checkpoint_dir}")
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(
            "Checkpoint path must point inside a checkpoints directory, got "
            f"{checkpoint_dir}"
        )
    if checkpoint_dir.name != "latest" and not checkpoint_dir.name.startswith("step_"):
        raise ValueError(
            "Checkpoint directory must be 'latest' or 'step_XXXXXX', got "
            f"{checkpoint_dir.name!r}"
        )
    return checkpoint_dir


def save_checkpoint(
    workspace: Path,
    trainer: Trainer,
    *,
    saved_during_eval: bool,
) -> Path:
    """Persist model, optimizer, trainer state, and metadata."""
    checkpoint_root = workspace / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    step = trainer.state.global_step
    checkpoint_dir = checkpoint_root / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "model.pt"
    optimizer_path = checkpoint_dir / "optimizer.pt"
    scheduler_path = checkpoint_dir / "scheduler.pt"
    trainer_state_path = checkpoint_dir / "trainer_state.pt"
    flashlight_path = checkpoint_dir / "flashlight.pt"
    meta_path = checkpoint_dir / "meta.json"

    torch.save(trainer.trajectory_model.state_dict(), model_path)
    torch.save(trainer.optimizer.state_dict(), optimizer_path)
    torch.save(trainer.scheduler.state_dict(), scheduler_path)
    torch.save(_trainer_state_payload(trainer.state), trainer_state_path)
    torch.save(_flashlight_payload(trainer.system.flash_light), flashlight_path)

    resume_ready = trainer.state.cycle_step == 0
    meta = {
        "step": step,
        "stage": trainer.state.stage,
        "resume_ready": resume_ready,
        "cycle_step": trainer.state.cycle_step,
        "saved_at_refresh_boundary": resume_ready,
        "saved_during_eval": saved_during_eval,
        "effective_lr": float(trainer.optimizer.param_groups[0]["lr"]),
        "files": [
            "flashlight.pt",
            "meta.json",
            "model.pt",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.pt",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    latest_dir = checkpoint_root / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(checkpoint_dir, latest_dir)

    return checkpoint_dir


def load_resume_checkpoint(checkpoint_dir: Path, trainer: Trainer) -> TrainerState:
    """Restore model, optimizer, and runtime state for boundary resume."""
    resolved_dir = resolve_checkpoint_dir(checkpoint_dir)
    meta = _load_meta(resolved_dir)
    if not meta.get("resume_ready", False):
        raise ValueError(f"Checkpoint is not resume-ready: {resolved_dir}")

    trainer_state = _load_trainer_state(
        resolved_dir / "trainer_state.pt",
        device=trainer.device,
        dtype=trainer.dtype,
    )
    if trainer_state.cycle_step != 0:
        raise ValueError(
            "Resume-ready checkpoints must have trainer_state.cycle_step == 0, got "
            f"{trainer_state.cycle_step}"
        )

    trainer.trajectory_model.load_state_dict(
        torch.load(resolved_dir / "model.pt", map_location=trainer.device)
    )
    _load_flashlight_state(
        resolved_dir / "flashlight.pt",
        trainer.system.flash_light,
        device=trainer.device,
        dtype=trainer.dtype,
    )
    trainer.optimizer.load_state_dict(
        torch.load(resolved_dir / "optimizer.pt", map_location=trainer.device)
    )
    trainer.scheduler.load_state_dict(
        torch.load(resolved_dir / "scheduler.pt", map_location=trainer.device)
    )
    trainer.state = trainer_state
    stage_config = trainer.init_stage_config if trainer.state.stage == "init" else trainer.local_stage_config
    trainer.schedule = RefreshSchedule(stage_config, generator=trainer.generator)
    return trainer.state


def load_sample_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    flash_light,
) -> dict[str, object]:
    """Restore model weights for sample-only inference and return checkpoint metadata."""
    resolved_dir = resolve_checkpoint_dir(checkpoint_dir)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.load_state_dict(torch.load(resolved_dir / "model.pt", map_location=device))
    _load_flashlight_state(
        resolved_dir / "flashlight.pt",
        flash_light,
        device=device,
        dtype=dtype,
    )
    return {
        "checkpoint_dir": resolved_dir,
        "meta": _load_meta(resolved_dir),
    }


def _flashlight_payload(flash_light) -> dict[str, torch.Tensor]:
    intensity = flash_light.intensity
    if isinstance(intensity, torch.Tensor):
        return {"intensity": intensity.detach().cpu()}
    return {"intensity": torch.tensor(float(intensity), dtype=torch.float32)}


def _trainer_state_payload(state: TrainerState) -> dict[str, object]:
    return {
        "global_step": state.global_step,
        "stage": state.stage,
        "carry_time": state.carry_time,
        "carry_state": state.carry_state.detach().cpu() if state.carry_state is not None else None,
        "cycle_step": state.cycle_step,
    }


def _load_trainer_state(
    trainer_state_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TrainerState:
    payload = torch.load(trainer_state_path, map_location=device)
    carry_state = payload["carry_state"]
    if carry_state is not None:
        carry_state = carry_state.to(device=device, dtype=dtype)
    return TrainerState(
        global_step=int(payload["global_step"]),
        stage=str(payload["stage"]),
        carry_time=float(payload["carry_time"]),
        carry_state=carry_state,
        cycle_step=int(payload["cycle_step"]),
    )


def _load_meta(checkpoint_dir: Path) -> dict[str, object]:
    return json.loads((checkpoint_dir / "meta.json").read_text(encoding="utf-8"))


def _load_flashlight_state(
    flashlight_path: Path,
    flash_light,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not flashlight_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing flashlight state: {flashlight_path}")
    payload = torch.load(flashlight_path, map_location=device)
    intensity = payload["intensity"].to(device=device, dtype=dtype)
    if isinstance(flash_light.intensity, torch.Tensor):
        flash_light.intensity.data.copy_(intensity)
        return
    flash_light.intensity = float(intensity.item())


__all__ = [
    "load_resume_checkpoint",
    "load_sample_checkpoint",
    "resolve_checkpoint_dir",
    "save_checkpoint",
]
