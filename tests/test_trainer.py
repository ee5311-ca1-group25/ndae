import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ndae.config import RenderingConfig
from ndae.data import Timeline
from ndae.models import NDAEUNet, ODEFunction, TrajectoryModel
from ndae.rendering import (
    Camera,
    FlashLight,
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
            batch_size=batch_size,
            workspace=tmp_path,
            n_iter=n_iter,
            n_init_iter=n_init_iter,
            log_every=log_every,
            loss_type=loss_type,
            n_loss_crops=n_loss_crops,
            overflow_weight=overflow_weight,
            init_height_weight=init_height_weight,
            gamma=rendering.gamma,
            generator=generator,
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

    assert metrics_path.is_file()
    assert len(payloads) == 3
    assert [payload["global_step"] for payload in payloads] == [1, 2, 3]
    assert payloads[0]["stage"] == "init"
    assert payloads[1]["stage"] == "local"
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
    } <= payloads[0].keys()


def test_trainer_init_stage_uses_random_take_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=1, n_loss_crops=3)
    calls = {"take": 0, "crop": 0}

    def fake_take_spec(*args: object, **kwargs: object) -> trainer_module.CropSampleSpec:
        del args, kwargs
        calls["take"] += 1
        return trainer_module.CropSampleSpec(
            kind="take",
            height=trainer.crop_size,
            width=trainer.crop_size,
            indices=torch.arange(trainer.crop_size * trainer.crop_size),
        )

    def fake_crop_spec(*args: object, **kwargs: object) -> trainer_module.CropSampleSpec:
        del args, kwargs
        calls["crop"] += 1
        return trainer_module.CropSampleSpec(
            kind="rect",
            height=trainer.crop_size,
            width=trainer.crop_size,
            top=0,
            left=0,
        )

    monkeypatch.setattr(trainer_module, "sample_random_take_spec", fake_take_spec)
    monkeypatch.setattr(trainer_module, "sample_random_crop_spec", fake_crop_spec)

    trainer.step()

    assert calls["take"] == 3
    assert calls["crop"] == 0


def test_trainer_local_stage_uses_random_crop_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(tmp_path, n_init_iter=0, n_loss_crops=4)
    calls = {"take": 0, "crop": 0}

    def fake_take_spec(*args: object, **kwargs: object) -> trainer_module.CropSampleSpec:
        del args, kwargs
        calls["take"] += 1
        return trainer_module.CropSampleSpec(
            kind="take",
            height=trainer.crop_size,
            width=trainer.crop_size,
            indices=torch.arange(trainer.crop_size * trainer.crop_size),
        )

    def fake_crop_spec(*args: object, **kwargs: object) -> trainer_module.CropSampleSpec:
        del args, kwargs
        calls["crop"] += 1
        return trainer_module.CropSampleSpec(
            kind="rect",
            height=trainer.crop_size,
            width=trainer.crop_size,
            top=0,
            left=0,
        )

    monkeypatch.setattr(trainer_module, "sample_random_take_spec", fake_take_spec)
    monkeypatch.setattr(trainer_module, "sample_random_crop_spec", fake_crop_spec)

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
