"""Trainer orchestration runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..evaluation import run_eval, should_eval
from ..losses import init_loss, local_loss, overflow_loss
from ..rendering import split_latent_maps
from .config import (
    TrainerConfig,
    TrainerLossConfig,
    TrainerRuntimeConfig,
    TrainerSchedulerConfig,
    TrainerStageConfig,
)
from .schedule import RefreshSchedule, StageConfig
from .solver import rollout_generation, rollout_warmup
from .state import TrainerState
from .system import SVBRDFSystem
from .target_sampling import sample_target_batch


@dataclass(slots=True)
class TrainerComponents:
    """Long-lived runtime objects injected into the trainer."""

    system: SVBRDFSystem
    optimizer_factory: Callable[[], torch.optim.Optimizer]
    schedule: RefreshSchedule
    init_stage_config: StageConfig
    local_stage_config: StageConfig
    vgg_features: nn.Module


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
        self.runtime = config.runtime
        self.stage_cfg = config.stage
        self.loss_cfg = config.loss
        self.scheduler_cfg = config.scheduler
        self.device = self.runtime.device or next(self.trajectory_model.parameters()).device
        self.dtype = next(self.trajectory_model.parameters()).dtype
        self.exemplar_frames = config.exemplar_frames.to(device=self.device, dtype=self.dtype)
        self.vgg_features = self.vgg_features.to(self.device, dtype=self.dtype)
        self.metrics_path = self.runtime.workspace / "metrics.jsonl"
        if not self.metrics_path.exists():
            self.metrics_path.write_text("", encoding="utf-8")
        self.scheduler = self._build_scheduler()
        initial_stage = "init" if self.stage_cfg.n_init_iter > 0 else "local"
        initial_stage_config = self._stage_config_for(initial_stage)
        self.state = TrainerState(
            global_step=0,
            stage=initial_stage,
            carry_time=initial_stage_config.t_start,
            carry_state=None,
            cycle_step=0,
        )

    @property
    def batch_size(self) -> int:
        return self.runtime.batch_size

    @property
    def workspace(self) -> Path:
        return self.runtime.workspace

    @property
    def generator(self) -> torch.Generator | None:
        return self.runtime.generator

    @property
    def n_init_iter(self) -> int:
        return self.stage_cfg.n_init_iter

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

        brdf_maps, height_map = self._project_state(rollout.final_state)
        target_index = self.timeline.time_to_frame(window.t1)
        target = self._sample_target_batch(
            target_index,
            current_stage=current_stage,
            brdf_maps=brdf_maps,
            height_map=height_map,
        )

        loss_init = torch.zeros((), device=self.device, dtype=self.dtype)
        loss_local = torch.zeros((), device=self.device, dtype=self.dtype)
        loss_overflow = self.loss_cfg.overflow_weight * overflow_loss(brdf_maps)
        if current_stage == "init":
            rendered = target["rendered"]
            loss_height = self.loss_cfg.init_height_weight * (height_map.square().mean())
            loss_init = init_loss(rendered, target["target"]) + loss_height
            loss_total = loss_init + loss_overflow
        else:
            rendered = target["rendered"]
            loss_local = local_loss(
                self.vgg_features,
                rendered,
                target["target"],
                loss_type=self.loss_cfg.loss_type,
                generator=self.runtime.generator,
            )
            loss_total = loss_local + loss_overflow

        self.optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        self._normalize_gradients()
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
        eval_callback: Callable[[Trainer, dict[str, float | int | str]], None] | None = None,
    ) -> None:
        """Run the configured number of training steps."""

        for _ in range(self.runtime.n_iter):
            iteration = self.state.global_step
            metrics = self.step()
            if self.state.global_step % self.runtime.log_every == 0:
                self._log_metrics(metrics)
            if should_eval(self, iteration):
                eval_metrics = run_eval(self, iteration=iteration)
                self._log_metrics(eval_metrics)
                if eval_callback is not None:
                    eval_callback(self, eval_metrics)

    def _current_stage(self) -> Literal["init", "local"]:
        return "init" if self.state.global_step < self.stage_cfg.n_init_iter else "local"

    def _enter_local_stage(self) -> None:
        self.optimizer = self.optimizer_factory()
        self.scheduler = self._build_scheduler()
        self.schedule = RefreshSchedule(self.local_stage_config, generator=self.runtime.generator)
        self.state.stage = "local"
        self.state.carry_state = None
        self.state.carry_time = self.local_stage_config.t_start
        self.state.cycle_step = 0

    def _build_scheduler(self) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.scheduler_cfg.scheduler_factor,
            patience=self.scheduler_cfg.scheduler_patience_evals,
            min_lr=self.scheduler_cfg.scheduler_min_lr,
        )

    def _stage_config_for(self, stage: Literal["init", "local"]) -> StageConfig:
        if stage == "init":
            return self.init_stage_config
        return self.local_stage_config

    def _resolve_initial_state(self, *, refresh: bool) -> torch.Tensor:
        if refresh:
            return torch.randn(
                self.runtime.batch_size,
                self.system.total_channels,
                self.crop_size,
                self.crop_size,
                generator=self.runtime.generator,
                device=self.device,
                dtype=self.dtype,
            )

        if self.state.carry_state is None:
            raise RuntimeError("Trainer requires carry_state for generation windows.")
        return self.state.carry_state.detach()

    def _project_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        brdf_maps, height_map = split_latent_maps(
            state,
            n_brdf_channels=self.system.n_brdf_channels,
            n_normal_channels=self.system.n_normal_channels,
        )
        return brdf_maps, height_map

    def _sample_target_batch(
        self,
        target_index: int,
        *,
        current_stage: Literal["init", "local"],
        brdf_maps: torch.Tensor,
        height_map: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return sample_target_batch(
            self,
            target_index,
            current_stage=current_stage,
            brdf_maps=brdf_maps,
            height_map=height_map,
        )

    def _normalize_gradients(self) -> None:
        for parameter in self.trajectory_model.parameters():
            if parameter.grad is None:
                continue
            parameter.grad.div_(parameter.grad.norm() + 1e-8)
        intensity = self.system.flash_light.intensity
        if isinstance(intensity, torch.Tensor) and intensity.grad is not None:
            intensity.grad.div_(intensity.grad.norm() + 1e-8)

    def _log_metrics(self, metrics: dict[str, float | int | str]) -> None:
        line = json.dumps(metrics, sort_keys=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

        if metrics.get("event") == "eval":
            message = (
                "step={global_step} stage={stage} event=eval scope={eval_scope} "
                "effective_lr={effective_lr:.6f}"
            )
            if "inference_loss" in metrics:
                message += " inference_loss={inference_loss:.6f}"
            print(message.format(**metrics))
            return

        print(
            "step={global_step} stage={stage} window={window_kind} "
            "loss_total={loss_total:.6f} t0={t0:.3f} t1={t1:.3f}".format(**metrics)
        )


__all__ = ["Trainer", "TrainerComponents"]
