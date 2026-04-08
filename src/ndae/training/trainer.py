"""Minimal trainer runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn as nn

from ..data import Timeline, random_crop
from ..losses import init_loss, local_loss, overflow_loss
from ..rendering import split_latent_maps
from .schedule import RefreshSchedule, StageConfig
from .solver import rollout_generation, rollout_warmup
from .system import SVBRDFSystem, render_latent_state


@dataclass(slots=True)
class TrainerState:
    """Minimal runtime state carried across training steps."""

    global_step: int
    stage: Literal["init", "local"]
    carry_time: float
    carry_state: torch.Tensor | None
    cycle_step: int


@dataclass(slots=True)
class TrainerComponents:
    """Long-lived runtime objects injected into the trainer."""

    system: SVBRDFSystem
    optimizer_factory: Callable[[], torch.optim.Optimizer]
    schedule: RefreshSchedule
    init_stage_config: StageConfig
    local_stage_config: StageConfig
    vgg_features: nn.Module


@dataclass(slots=True)
class TrainerConfig:
    """Config and data inputs for one trainer instance."""

    exemplar_frames: torch.Tensor
    timeline: Timeline
    crop_size: int
    batch_size: int
    workspace: Path
    n_iter: int
    n_init_iter: int
    log_every: int
    generator: torch.Generator | None = None
    device: torch.device | None = None


class Trainer:
    """Coordinate schedule, rollout, rendering, loss, and optimization."""

    def __init__(
        self,
        *,
        components: TrainerComponents,
        config: TrainerConfig,
    ) -> None:
        self.system = components.system
        self.trajectory_model = self.system.trajectory_model
        self.optimizer_factory = components.optimizer_factory
        self.optimizer = self.optimizer_factory()
        self.schedule = components.schedule
        self.init_stage_config = components.init_stage_config
        self.local_stage_config = components.local_stage_config
        self.vgg_features = components.vgg_features
        self.timeline = config.timeline
        self.crop_size = config.crop_size
        self.batch_size = config.batch_size
        self.workspace = config.workspace
        self.n_iter = config.n_iter
        self.n_init_iter = config.n_init_iter
        self.log_every = config.log_every
        self.generator = config.generator
        self.device = config.device or next(self.trajectory_model.parameters()).device
        self.dtype = next(self.trajectory_model.parameters()).dtype
        self.exemplar_frames = config.exemplar_frames.to(device=self.device, dtype=self.dtype)
        self.vgg_features = self.vgg_features.to(self.device, dtype=self.dtype)
        self.metrics_path = self.workspace / "metrics.jsonl"
        if not self.metrics_path.exists():
            self.metrics_path.write_text("", encoding="utf-8")
        initial_stage = "init" if self.n_init_iter > 0 else "local"
        initial_stage_config = self._stage_config_for(initial_stage)
        self.state = TrainerState(
            global_step=0,
            stage=initial_stage,
            carry_time=initial_stage_config.t_start,
            carry_state=None,
            cycle_step=0,
        )

    def step(self) -> dict[str, float | int | str]:
        """Run one optimization step and update the trainer state."""

        current_stage = self._current_stage()
        if current_stage == "local" and self.state.stage != "local":
            self._enter_local_stage()
        else:
            self.state.stage = current_stage

        cycle_step = self.state.cycle_step
        window = self.schedule.next(
            iteration=cycle_step,
            carry_time=self.state.carry_time,
        )
        z0 = self._resolve_initial_state(refresh=window.refresh)
        rollout = (
            rollout_warmup(
                self.trajectory_model,
                z0,
                window,
                self.system.solver_config,
            )
            if window.kind == "warmup"
            else rollout_generation(
                self.trajectory_model,
                z0,
                window,
                self.system.solver_config,
            )
        )

        rendered, brdf_maps = self._project_state(rollout.final_state)
        target_index = self.timeline.time_to_frame(window.t1)
        target = self._sample_target_batch(target_index)

        loss_init = torch.zeros((), device=self.device, dtype=self.dtype)
        loss_local = torch.zeros((), device=self.device, dtype=self.dtype)
        loss_overflow = overflow_loss(brdf_maps)
        if current_stage == "init":
            loss_init = init_loss(rendered, target)
            loss_total = loss_init + loss_overflow
        else:
            loss_local = local_loss(
                self.vgg_features,
                rendered,
                target,
                loss_type="SW",
                generator=self.generator,
            )
            loss_total = loss_local + loss_overflow

        self.optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        self.optimizer.step()

        self.state.carry_state = rollout.final_state.detach()
        self.state.carry_time = window.t1
        self.state.global_step += 1
        self.state.stage = current_stage
        self.state.cycle_step = (cycle_step + 1) % self._stage_config_for(current_stage).refresh_rate

        return {
            "global_step": self.state.global_step,
            "stage": current_stage,
            "cycle_step": cycle_step,
            "window_kind": window.kind,
            "t0": float(window.t0),
            "t1": float(window.t1),
            "target_index": target_index,
            "loss_total": float(loss_total.detach().item()),
            "loss_init": float(loss_init.detach().item()),
            "loss_local": float(loss_local.detach().item()),
            "loss_overflow": float(loss_overflow.detach().item()),
        }

    def run(
        self,
        checkpoint_callback: Callable[[Trainer], None] | None = None,
    ) -> None:
        """Run the configured number of training steps."""

        for _ in range(self.n_iter):
            metrics = self.step()
            if self.state.global_step % self.log_every == 0:
                self._log_metrics(metrics)
            if checkpoint_callback is not None:
                checkpoint_callback(self)

    def _current_stage(self) -> Literal["init", "local"]:
        return "init" if self.state.global_step < self.n_init_iter else "local"

    def _enter_local_stage(self) -> None:
        self.optimizer = self.optimizer_factory()
        self.schedule = RefreshSchedule(self.local_stage_config, generator=self.generator)
        self.state.stage = "local"
        self.state.carry_state = None
        self.state.carry_time = self.local_stage_config.t_start
        self.state.cycle_step = 0

    def _stage_config_for(self, stage: Literal["init", "local"]) -> StageConfig:
        if stage == "init":
            return self.init_stage_config
        return self.local_stage_config

    def _resolve_initial_state(self, *, refresh: bool) -> torch.Tensor:
        if refresh:
            return torch.randn(
                self.batch_size,
                self.system.total_channels,
                self.crop_size,
                self.crop_size,
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )

        if self.state.carry_state is None:
            raise RuntimeError("Trainer requires carry_state for generation windows.")
        return self.state.carry_state.detach()

    def _sample_target_batch(self, target_index: int) -> torch.Tensor:
        target_frame = self.exemplar_frames[target_index]
        return torch.stack(
            [
                random_crop(
                    target_frame,
                    self.crop_size,
                    self.crop_size,
                    generator=self.generator,
                )
                for _ in range(self.batch_size)
            ],
            dim=0,
        )

    def _project_state(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        brdf_maps, _ = split_latent_maps(
            state,
            n_brdf_channels=self.system.n_brdf_channels,
            n_normal_channels=self.system.n_normal_channels,
        )
        return render_latent_state(self.system, state), brdf_maps

    def _log_metrics(self, metrics: dict[str, float | int | str]) -> None:
        line = json.dumps(metrics, sort_keys=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

        print(
            "step={global_step} stage={stage} window={window_kind} "
            "loss_total={loss_total:.6f} t0={t0:.3f} t1={t1:.3f}".format(**metrics)
        )


__all__ = ["Trainer", "TrainerComponents", "TrainerConfig", "TrainerState"]
