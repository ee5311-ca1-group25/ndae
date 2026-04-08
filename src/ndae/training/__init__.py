"""Public training API for NDAE."""

from .config import (
    TrainerConfig,
    TrainerLossConfig,
    TrainerRuntimeConfig,
    TrainerSchedulerConfig,
    TrainerStageConfig,
)
from .factory import build_trainer
from .schedule import RefreshSchedule, RolloutWindow, StageConfig
from .solver import (
    RolloutResult,
    SolverConfig,
    rollout_generation,
    rollout_warmup,
    solve_rollout,
)
from .system import SVBRDFSystem, build_svbrdf_system, render_latent_state
from .checkpoint import (
    load_resume_checkpoint,
    load_sample_checkpoint,
    resolve_checkpoint_dir,
    save_checkpoint,
)
from .state import TrainerState
from .trainer import Trainer, TrainerComponents

__all__ = [
    "SVBRDFSystem",
    "build_svbrdf_system",
    "render_latent_state",
    "build_trainer",
    "SolverConfig",
    "RolloutResult",
    "solve_rollout",
    "rollout_warmup",
    "rollout_generation",
    "StageConfig",
    "RolloutWindow",
    "RefreshSchedule",
    "resolve_checkpoint_dir",
    "save_checkpoint",
    "load_resume_checkpoint",
    "load_sample_checkpoint",
    "Trainer",
    "TrainerComponents",
    "TrainerConfig",
    "TrainerRuntimeConfig",
    "TrainerStageConfig",
    "TrainerLossConfig",
    "TrainerSchedulerConfig",
    "TrainerState",
]
