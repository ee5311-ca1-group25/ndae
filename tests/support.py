from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from ndae.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    NDAEConfig,
    RenderingConfig,
    TrainConfig,
    TrainLossConfig,
    TrainRuntimeConfig,
    TrainSchedulerConfig,
    TrainStageConfig,
)
from ndae.data import Timeline
from ndae.models import NDAEUNet, ODEFunction, TrajectoryModel
from ndae.rendering import Camera, FlashLight, select_renderer
from ndae.training import (
    RefreshSchedule,
    SVBRDFSystem,
    SolverConfig,
    StageConfig,
    Trainer,
    TrainerComponents,
    TrainerConfig,
)
from ndae.training.system import resolve_renderer_runtime


class DummyFeatures(nn.Module):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x]


def write_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def make_config(
    *,
    output_root: str,
    name: str = "demo",
    data_root: str = "unused_data_root",
    exemplar: str = "example",
    image_size: int = 16,
    crop_size: int = 8,
    n_frames: int = 4,
    t_S: float = 0.0,
    t_E: float = 2.0,
    dry_run: bool = False,
    n_iter: int = 3,
    n_init_iter: int = 1,
    log_every: int = 1,
    checkpoint_every: int = 1,
    loss_type: str = "SW",
    refresh_rate_init: int = 2,
    refresh_rate_local: int = 6,
    eval_every: int = 500,
    n_loss_crops: int = 32,
    overflow_weight: float = 100.0,
    init_height_weight: float = 1.0,
    scheduler_factor: float = 0.5,
    scheduler_patience_evals: int = 5,
    scheduler_min_lr: float = 1e-4,
    resume_from: str | None = None,
) -> NDAEConfig:
    renderer_spec = select_renderer("diffuse_cook_torrance")
    return NDAEConfig(
        experiment=ExperimentConfig(name=name, output_root=output_root, seed=7),
        data=DataConfig(
            root=data_root,
            exemplar=exemplar,
            image_size=image_size,
            crop_size=crop_size,
            n_frames=n_frames,
            t_S=t_S,
            t_E=t_E,
        ),
        model=ModelConfig(dim=8, solver="euler"),
        rendering=RenderingConfig(
            renderer_type=renderer_spec.renderer_type,
            n_brdf_channels=renderer_spec.n_brdf_channels,
        ),
        train=TrainConfig(
            runtime=TrainRuntimeConfig(
                batch_size=1,
                lr=1e-2,
                dry_run=dry_run,
                n_iter=n_iter,
                log_every=log_every,
                checkpoint_every=checkpoint_every,
                resume_from=resume_from,
            ),
            stage=TrainStageConfig(
                n_init_iter=n_init_iter,
                refresh_rate_init=refresh_rate_init,
                refresh_rate_local=refresh_rate_local,
            ),
            loss=TrainLossConfig(
                loss_type=loss_type,
                n_loss_crops=n_loss_crops,
                overflow_weight=overflow_weight,
                init_height_weight=init_height_weight,
            ),
            scheduler=TrainSchedulerConfig(
                eval_every=eval_every,
                scheduler_factor=scheduler_factor,
                scheduler_patience_evals=scheduler_patience_evals,
                scheduler_min_lr=scheduler_min_lr,
            ),
        ),
    )


def make_trainer(
    workspace: Path,
    *,
    config: NDAEConfig | None = None,
    refresh_rate: int = 3,
) -> Trainer:
    config = config or make_config(output_root=str(workspace), name=workspace.name)
    model = TrajectoryModel(
        ODEFunction(
            NDAEUNet(
                in_dim=config.rendering.total_channels,
                out_dim=config.rendering.total_channels,
                dim=config.model.dim,
                dim_mults=(1, 2),
                use_attn=False,
            )
        )
    )
    timeline = Timeline.from_config(config.data)
    generator = torch.Generator().manual_seed(config.experiment.seed)
    exemplar_frames = torch.rand(
        timeline.n_frames,
        3,
        config.data.image_size,
        config.data.image_size,
        generator=torch.Generator().manual_seed(11),
    )
    init_stage_config = StageConfig(
        t_init=timeline.t_I,
        t_start=timeline.t_S,
        t_end=timeline.t_E,
        refresh_rate=refresh_rate,
    )
    local_stage_config = StageConfig(
        t_init=timeline.t_I,
        t_start=timeline.t_S,
        t_end=timeline.t_E,
        refresh_rate=refresh_rate,
    )
    schedule = RefreshSchedule(
        init_stage_config if config.train.stage.n_init_iter > 0 else local_stage_config,
        generator=generator,
    )
    renderer_pp, unpack_fn = resolve_renderer_runtime(config)
    flash_light = FlashLight(intensity=torch.nn.Parameter(torch.tensor(0.0)))

    def optimizer_factory() -> torch.optim.Optimizer:
        return torch.optim.Adam(
            [*model.parameters(), flash_light.intensity],
            lr=config.train.runtime.lr,
        )

    return Trainer(
        components=TrainerComponents(
            system=SVBRDFSystem(
                trajectory_model=model,
                solver_config=SolverConfig(method="euler"),
                camera=Camera(),
                flash_light=flash_light,
                renderer_pp=renderer_pp,
                unpack_fn=unpack_fn,
                total_channels=config.rendering.total_channels,
                n_brdf_channels=config.rendering.n_brdf_channels,
                n_normal_channels=config.rendering.n_normal_channels,
                height_scale=config.rendering.height_scale,
                gamma=config.rendering.gamma,
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
            crop_size=config.data.crop_size,
            batch_size=config.train.runtime.batch_size,
            workspace=workspace,
            n_iter=config.train.runtime.n_iter,
            n_init_iter=config.train.stage.n_init_iter,
            log_every=config.train.runtime.log_every,
            generator=generator,
        ),
    )
