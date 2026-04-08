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
from ndae.training import (
    RefreshSchedule,
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
