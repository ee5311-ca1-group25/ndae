import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import ndae.evaluation.runtime as evaluation_runtime_module
import ndae.training.target_sampling as target_sampling_module
from ndae.config import RenderingConfig
from ndae.data import Timeline
from ndae.models import NDAEUNet, ODEFunction, TrajectoryModel
from ndae.rendering import (
    Camera,
    FlashLight,
    create_meshgrid,
    diffuse_cook_torrance,
    select_renderer,
    unpack_brdf_diffuse_cook_torrance,
)
import ndae.training.trainer as trainer_module
from ndae.training import (
    RefreshSchedule,
    RolloutResult,
    SVBRDFSystem,
    SolverConfig,
    StageConfig,
    Trainer,
    TrainerComponents,
    TrainerConfig,
    TrainerLossConfig,
    TrainerRuntimeConfig,
    TrainerSchedulerConfig,
    TrainerStageConfig,
)


class DummyFeatures(nn.Module):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x]


def make_trainer(
    tmp_path: Path,
    *,
    batch_size: int = 1,
    n_iter: int = 3,
    n_init_iter: int = 1,
    log_every: int = 1,
    loss_type: str = "SW",
    n_loss_crops: int = 2,
    overflow_weight: float = 100.0,
    init_height_weight: float = 1.0,
    eval_every: int = 500,
    scheduler_factor: float = 0.5,
    scheduler_patience_evals: int = 5,
    scheduler_min_lr: float = 1e-4,
) -> Trainer:
    renderer_spec = select_renderer("diffuse_cook_torrance")
    rendering = RenderingConfig(
        renderer_type=renderer_spec.renderer_type,
        n_brdf_channels=renderer_spec.n_brdf_channels,
    )
    model = TrajectoryModel(
        ODEFunction(
            NDAEUNet(
                in_dim=rendering.total_channels,
                out_dim=rendering.total_channels,
                dim=8,
                dim_mults=(1, 2),
                use_attn=False,
            )
        )
    )
    timeline = Timeline(t_I=-2.0, t_S=0.0, t_E=2.0, n_frames=4)
    generator = torch.Generator().manual_seed(7)
    exemplar_frames = torch.rand(
        timeline.n_frames,
        3,
        16,
        16,
        generator=torch.Generator().manual_seed(11),
    )
    init_stage_config = StageConfig(
        t_init=timeline.t_I,
        t_start=timeline.t_S,
        t_end=timeline.t_E,
        refresh_rate=2,
    )
    local_stage_config = StageConfig(
        t_init=timeline.t_I,
        t_start=timeline.t_S,
        t_end=timeline.t_E,
        refresh_rate=3,
    )
    schedule = RefreshSchedule(
        init_stage_config if n_init_iter > 0 else local_stage_config,
        generator=generator,
    )
    flash_light = FlashLight(intensity=torch.nn.Parameter(torch.tensor(0.0)))

    def optimizer_factory() -> torch.optim.Optimizer:
        return torch.optim.Adam([*model.parameters(), flash_light.intensity], lr=1e-2)

    return Trainer(
        components=TrainerComponents(
            system=SVBRDFSystem(
                trajectory_model=model,
                solver_config=SolverConfig(method="euler"),
                camera=Camera(),
                flash_light=flash_light,
                renderer_pp=diffuse_cook_torrance,
                unpack_fn=unpack_brdf_diffuse_cook_torrance,
                total_channels=rendering.total_channels,
                n_brdf_channels=rendering.n_brdf_channels,
                n_normal_channels=rendering.n_normal_channels,
                height_scale=rendering.height_scale,
                gamma=rendering.gamma,
            ),
            optimizer_factory=optimizer_factory,
            schedule=schedule,
            init_stage_config=init_stage_config,
            local_stage_config=local_stage_config,
            vgg_features=DummyFeatures(),
        ),
        config=TrainerConfig(
            exemplar_frames=exemplar_frames,
            timeline=timeline,
            crop_size=8,
            runtime=TrainerRuntimeConfig(
                batch_size=batch_size,
                workspace=tmp_path,
                n_iter=n_iter,
                log_every=log_every,
                gamma=rendering.gamma,
                generator=generator,
            ),
            stage=TrainerStageConfig(
                n_init_iter=n_init_iter,
                refresh_rate_init=2,
                refresh_rate_local=3,
            ),
            loss=TrainerLossConfig(
                loss_type=loss_type,
                n_loss_crops=n_loss_crops,
                overflow_weight=overflow_weight,
                init_height_weight=init_height_weight,
            ),
            scheduler=TrainerSchedulerConfig(
                eval_every=eval_every,
                scheduler_factor=scheduler_factor,
                scheduler_patience_evals=scheduler_patience_evals,
                scheduler_min_lr=scheduler_min_lr,
            ),
        ),
    )


def test_trainer_single_step_updates_parameters(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path)
    before = [parameter.detach().clone() for parameter in trainer.trajectory_model.parameters()]

    metrics = trainer.step()

    assert metrics["stage"] == "init"
    assert trainer.state.global_step == 1
    assert any(
        not torch.allclose(before_param, after_param.detach())
        for before_param, after_param in zip(before, trainer.trajectory_model.parameters())
    )


def test_trainer_switches_stage_after_n_init_iter(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=1)

    first = trainer.step()
    second = trainer.step()

    assert first["stage"] == "init"
    assert second["stage"] == "local"
    assert second["window_kind"] == "warmup"
    assert second["cycle_step"] == 0
    assert trainer.state.stage == "local"
    assert trainer.state.global_step == 2
    assert trainer.state.cycle_step == 1


def test_trainer_uses_distinct_init_and_local_stage_configs(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=1)

    assert trainer.init_stage_config.refresh_rate == 2
    assert trainer.local_stage_config.refresh_rate == 3


def test_trainer_detaches_and_advances_carry_state(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=0)

    first = trainer.step()
    carry_after_first = trainer.state.carry_state
    assert carry_after_first is not None
    assert carry_after_first.requires_grad is False
    assert trainer.state.carry_time == pytest.approx(first["t1"])

    second = trainer.step()

    assert second["stage"] == "local"
    assert second["window_kind"] == "generation"
    assert second["t0"] == pytest.approx(first["t1"])
    assert trainer.state.carry_state is not None
    assert trainer.state.carry_state.requires_grad is False
    assert trainer.state.carry_time == pytest.approx(second["t1"])
    assert trainer.state.carry_time > second["t0"]


def test_trainer_run_writes_metrics_jsonl(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, n_iter=3, n_init_iter=1, log_every=1)

    trainer.run()

    metrics_path = tmp_path / "metrics.jsonl"
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in lines]
    step_payloads = [payload for payload in payloads if payload.get("event") != "eval"]
    eval_payloads = [payload for payload in payloads if payload.get("event") == "eval"]

    assert metrics_path.is_file()
    assert len(step_payloads) == 3
    assert [payload["global_step"] for payload in step_payloads] == [1, 2, 3]
    assert step_payloads[0]["stage"] == "init"
    assert step_payloads[1]["stage"] == "local"
    assert len(eval_payloads) == 2
    assert eval_payloads[0]["eval_scope"] == "init"
    assert eval_payloads[1]["eval_scope"] == "local"
    assert {
        "global_step",
        "stage",
        "cycle_step",
        "window_kind",
        "t0",
        "t1",
        "target_index",
        "loss_total",
        "loss_init",
        "loss_local",
        "loss_overflow",
    } <= step_payloads[0].keys()


def test_trainer_init_stage_uses_random_take_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=1, n_loss_crops=3)
    calls = {"take": 0, "crop": 0}

    def fake_take_spec(*args: object, **kwargs: object) -> target_sampling_module.CropSampleSpec:
        del args, kwargs
        calls["take"] += 1
        return target_sampling_module.CropSampleSpec(
            kind="take",
            height=trainer.crop_size,
            width=trainer.crop_size,
            indices=torch.arange(trainer.crop_size * trainer.crop_size),
        )

    def fake_crop_spec(*args: object, **kwargs: object) -> target_sampling_module.CropSampleSpec:
        del args, kwargs
        calls["crop"] += 1
        return target_sampling_module.CropSampleSpec(
            kind="rect",
            height=trainer.crop_size,
            width=trainer.crop_size,
            top=0,
            left=0,
        )

    monkeypatch.setattr(target_sampling_module, "sample_random_take_spec", fake_take_spec)
    monkeypatch.setattr(target_sampling_module, "sample_random_crop_spec", fake_crop_spec)

    trainer.step()

    assert calls["take"] == 3
    assert calls["crop"] == 0


def test_trainer_local_stage_uses_random_crop_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=0, n_loss_crops=4)
    calls = {"take": 0, "crop": 0}

    def fake_take_spec(*args: object, **kwargs: object) -> target_sampling_module.CropSampleSpec:
        del args, kwargs
        calls["take"] += 1
        return target_sampling_module.CropSampleSpec(
            kind="take",
            height=trainer.crop_size,
            width=trainer.crop_size,
            indices=torch.arange(trainer.crop_size * trainer.crop_size),
        )

    def fake_crop_spec(*args: object, **kwargs: object) -> target_sampling_module.CropSampleSpec:
        del args, kwargs
        calls["crop"] += 1
        return target_sampling_module.CropSampleSpec(
            kind="rect",
            height=trainer.crop_size,
            width=trainer.crop_size,
            top=0,
            left=0,
        )

    monkeypatch.setattr(target_sampling_module, "sample_random_take_spec", fake_take_spec)
    monkeypatch.setattr(target_sampling_module, "sample_random_crop_spec", fake_crop_spec)

    trainer.step()

    assert calls["take"] == 0
    assert calls["crop"] == 4


def test_trainer_multicrop_uses_all_samples(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, batch_size=2, n_init_iter=0, n_loss_crops=3)
    brdf_maps = torch.full((2, 8, 8, 8), 0.5)
    height_map = torch.zeros(2, 1, 8, 8)

    sampled = trainer._sample_target_batch(
        0,
        current_stage="local",
        brdf_maps=brdf_maps,
        height_map=height_map,
    )

    assert sampled["target"].shape == (6, 3, 8, 8)
    assert sampled["rendered"].shape == (6, 3, 8, 8)


def test_render_sample_uses_exemplar_image_size_for_positions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=0)
    spec = target_sampling_module.CropSampleSpec(
        kind="rect",
        height=trainer.crop_size,
        width=trainer.crop_size,
        top=4,
        left=5,
        top_ratio=0.5,
        left_ratio=0.625,
    )
    brdf_maps = torch.full(
        (trainer.system.n_brdf_channels, trainer.crop_size, trainer.crop_size),
        0.5,
        dtype=trainer.dtype,
        device=trainer.device,
    )
    normal_map = torch.zeros(
        3,
        trainer.crop_size,
        trainer.crop_size,
        dtype=trainer.dtype,
        device=trainer.device,
    )
    captured: dict[str, torch.Tensor] = {}

    def fake_render_svbrdf(*args: object, **kwargs: object) -> torch.Tensor:
        del args
        captured["positions"] = kwargs["positions"]
        return torch.zeros(
            3,
            trainer.crop_size,
            trainer.crop_size,
            dtype=trainer.dtype,
            device=trainer.device,
        )

    monkeypatch.setattr(target_sampling_module, "render_svbrdf", fake_render_svbrdf)

    target_sampling_module.render_sample(
        trainer,
        brdf_maps,
        normal_map,
        spec,
        image_height=trainer.exemplar_frames.shape[-2],
        image_width=trainer.exemplar_frames.shape[-1],
    )

    expected = create_meshgrid(
        trainer.exemplar_frames.shape[-2],
        trainer.exemplar_frames.shape[-1],
        trainer.system.camera,
        device=trainer.device,
    )[:, 4 : 4 + trainer.crop_size, 5 : 5 + trainer.crop_size].to(dtype=trainer.dtype)
    assert torch.allclose(captured["positions"], expected)


def test_trainer_init_loss_includes_height_and_weighted_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(
        tmp_path,
        n_init_iter=1,
        n_loss_crops=1,
        overflow_weight=7.0,
        init_height_weight=3.0,
    )
    window = trainer.schedule.next(iteration=0, carry_time=trainer.state.carry_time)
    final_state = torch.zeros(
        trainer.batch_size,
        trainer.system.total_channels,
        trainer.crop_size,
        trainer.crop_size,
        dtype=trainer.dtype,
        device=trainer.device,
        requires_grad=True,
    )

    def fake_rollout(*args: object, **kwargs: object) -> RolloutResult:
        del args, kwargs
        return RolloutResult(states=final_state.unsqueeze(0), final_state=final_state, t0=window.t0, t1=window.t1)

    def fake_project_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del state
        brdf = torch.full(
            (trainer.batch_size, trainer.system.n_brdf_channels, trainer.crop_size, trainer.crop_size),
            -1.0,
            dtype=trainer.dtype,
            device=trainer.device,
        )
        height = torch.ones(
            trainer.batch_size,
            trainer.system.n_normal_channels,
            trainer.crop_size,
            trainer.crop_size,
            dtype=trainer.dtype,
            device=trainer.device,
        )
        return brdf, height

    def fake_sample_target_batch(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        del args, kwargs
        rendered = torch.zeros(
            1,
            3,
            trainer.crop_size,
            trainer.crop_size,
            dtype=trainer.dtype,
            device=trainer.device,
            requires_grad=True,
        )
        return {"target": torch.zeros_like(rendered), "rendered": rendered}

    monkeypatch.setattr(trainer_module, "rollout_warmup", fake_rollout)
    monkeypatch.setattr(trainer, "_project_state", fake_project_state)
    monkeypatch.setattr(trainer, "_sample_target_batch", fake_sample_target_batch)

    metrics = trainer.step()

    expected_overflow = 7.0 * float(((-1.0 - 1e-6) ** 2))
    expected_height = 3.0
    assert metrics["loss_overflow"] == pytest.approx(expected_overflow)
    assert metrics["loss_init"] == pytest.approx(expected_height)
    assert metrics["loss_total"] == pytest.approx(expected_overflow + expected_height)


def test_trainer_local_stage_supports_gram_loss(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=0, loss_type="GRAM", n_loss_crops=1)

    metrics = trainer.step()

    assert metrics["stage"] == "local"
    assert metrics["loss_local"] >= 0.0


def test_normalize_gradients_scales_nonzero_grads_and_keeps_zero_grads_finite(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path)
    first_param = next(trainer.trajectory_model.parameters())
    first_param.grad = torch.full_like(first_param, 3.0)
    intensity = trainer.system.flash_light.intensity
    intensity.grad = torch.zeros_like(intensity)

    trainer._normalize_gradients()

    assert first_param.grad is not None
    assert first_param.grad.norm().item() == pytest.approx(1.0, rel=1e-5)
    assert intensity.grad is not None
    assert torch.all(torch.isfinite(intensity.grad))
    assert intensity.grad.item() == pytest.approx(0.0)


def test_run_triggers_eval_at_iteration_zero_watershed_and_period(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(tmp_path, n_iter=4, n_init_iter=2, eval_every=2)
    seen: list[int] = []

    def fake_eval(current_trainer: Trainer, *, iteration: int) -> dict[str, float | int | str]:
        assert current_trainer is trainer
        seen.append(iteration)
        return {
            "global_step": trainer.state.global_step,
            "stage": trainer.state.stage,
            "event": "eval",
            "eval_iteration": iteration,
            "eval_scope": trainer.state.stage,
            "effective_lr": trainer.optimizer.param_groups[0]["lr"],
        }

    monkeypatch.setattr(trainer_module, "run_eval", fake_eval)

    trainer.run()

    assert seen == [0, 1, 2, 3]


def test_init_eval_does_not_step_scheduler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=2)
    calls: list[float] = []

    def fake_step(value: float) -> None:
        calls.append(value)

    monkeypatch.setattr(trainer.scheduler, "step", fake_step)

    metrics = evaluation_runtime_module.run_eval(trainer, iteration=0)

    assert metrics["eval_scope"] == "init"
    assert calls == []


def test_local_eval_steps_scheduler_and_reports_inference_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=0)
    calls: list[float] = []

    def fake_inference_loss(_trainer: Trainer) -> float:
        return 0.25

    def fake_step(value: float) -> None:
        calls.append(value)

    monkeypatch.setattr(evaluation_runtime_module, "compute_inference_loss", fake_inference_loss)
    monkeypatch.setattr(trainer.scheduler, "step", fake_step)

    metrics = evaluation_runtime_module.run_eval(trainer, iteration=0)

    assert metrics["eval_scope"] == "local"
    assert metrics["inference_loss"] == pytest.approx(0.25)
    assert calls == [0.25]


def test_enter_local_stage_resets_scheduler(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=1)
    before = trainer.scheduler

    trainer.step()
    trainer.step()

    assert trainer.state.stage == "local"
    assert trainer.scheduler is not before


def test_run_invokes_eval_callback_only_on_eval_steps(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, n_iter=4, n_init_iter=2, eval_every=2)
    calls: list[int] = []

    def on_eval(current_trainer: Trainer, metrics: dict[str, float | int | str]) -> None:
        assert current_trainer is trainer
        calls.append(int(metrics["eval_iteration"]))

    trainer.run(eval_callback=on_eval)

    assert calls == [0, 1, 2, 3]
