"""Evaluation helpers used by the trainer runtime."""

from __future__ import annotations

import torch

from ..losses import local_loss


def should_eval(trainer, iteration: int) -> bool:
    return (
        iteration == 0
        or (iteration + 1) % trainer.scheduler_cfg.eval_every == 0
        or iteration == trainer.stage_cfg.n_init_iter
    )


def run_eval(trainer, *, iteration: int) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {
        "global_step": trainer.state.global_step,
        "stage": trainer.state.stage,
        "event": "eval",
        "eval_iteration": iteration,
        "effective_lr": effective_lr(trainer),
    }
    if trainer.state.stage != "local":
        metrics["eval_scope"] = "init"
        return metrics

    inference_loss = compute_inference_loss(trainer)
    trainer.scheduler.step(inference_loss)
    metrics["eval_scope"] = "local"
    metrics["inference_loss"] = float(inference_loss)
    metrics["effective_lr"] = effective_lr(trainer)
    return metrics


def compute_inference_loss(trainer) -> float:
    from ..training.system import render_latent_state

    height, width = trainer.exemplar_frames.shape[-2:]
    z0 = torch.randn(
        1,
        trainer.system.total_channels,
        height,
        width,
        generator=trainer.runtime.generator,
        device=trainer.device,
        dtype=trainer.dtype,
    )
    t_eval = torch.cat(
        [
            torch.tensor([trainer.timeline.t_I], device=trainer.device, dtype=trainer.dtype),
            torch.linspace(
                trainer.timeline.t_S,
                trainer.timeline.t_E,
                trainer.timeline.n_frames,
                device=trainer.device,
                dtype=trainer.dtype,
            ),
        ]
    )
    states = trainer.trajectory_model(
        z0,
        t_eval,
        method=trainer.system.solver_config.method,
        rtol=trainer.system.solver_config.rtol,
        atol=trainer.system.solver_config.atol,
        **(trainer.system.solver_config.options or {}),
    )
    frame_losses: list[torch.Tensor] = []
    for frame_index, state in enumerate(states[1:]):
        rendered = render_latent_state(trainer.system, state)
        target = trainer.exemplar_frames[frame_index].unsqueeze(0)
        frame_losses.append(
            local_loss(
                trainer.vgg_features,
                rendered,
                target,
                loss_type=trainer.loss_cfg.loss_type,
                generator=trainer.runtime.generator,
            )
        )
    return float(torch.stack(frame_losses).mean().detach().item())


def effective_lr(trainer) -> float:
    return float(trainer.optimizer.param_groups[0]["lr"])


__all__ = ["compute_inference_loss", "effective_lr", "run_eval", "should_eval"]
