from dataclasses import FrozenInstanceError

import pytest
import torch

from ndae.training import RolloutResult, SolverConfig
from ndae.training.solver import RolloutResult as RolloutResultImpl
from ndae.training.solver import SolverConfig as SolverConfigImpl


def test_solver_config_defaults_match_lecture7_plan() -> None:
    config = SolverConfig()

    assert config.method == "adaptive_heun"
    assert config.rtol == pytest.approx(1e-2)
    assert config.atol == pytest.approx(1e-2)
    assert config.options is None


def test_solver_config_is_frozen() -> None:
    config = SolverConfig()

    with pytest.raises(FrozenInstanceError):
        config.method = "euler"


def test_solver_config_is_exported_from_training_package() -> None:
    assert SolverConfig is SolverConfigImpl


def test_rollout_result_exposes_fields() -> None:
    states = torch.randn(2, 3, 9, 8, 8)
    final_state = states[-1]
    result = RolloutResult(states=states, final_state=final_state, t0=-2.0, t1=0.0)

    assert result.states is states
    assert result.final_state is final_state
    assert result.t0 == pytest.approx(-2.0)
    assert result.t1 == pytest.approx(0.0)


def test_rollout_result_is_exported_from_training_package() -> None:
    assert RolloutResult is RolloutResultImpl
