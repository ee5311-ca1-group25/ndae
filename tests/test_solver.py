from dataclasses import FrozenInstanceError

import pytest
import torch

from ndae.models.odefunc import ODEFunction
from ndae.models.trajectory import TrajectoryModel
from ndae.models.unet import NDAEUNet
from ndae.training import RolloutResult, SolverConfig
from ndae.training.solver import RolloutResult as RolloutResultImpl
from ndae.training.solver import SolverConfig as SolverConfigImpl
from ndae.training.solver import solve_rollout


def make_trajectory_model() -> TrajectoryModel:
    model = NDAEUNet(in_dim=9, out_dim=9, dim=16, dim_mults=(1, 2), use_attn=False)
    return TrajectoryModel(ODEFunction(model))


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


def test_solve_rollout_returns_rollout_result_with_expected_shapes() -> None:
    trajectory_model = make_trajectory_model()
    z0 = torch.randn(2, 9, 8, 8)

    result = solve_rollout(trajectory_model, z0, t0=-2.0, t1=0.0, config=SolverConfig(method="euler"))

    assert isinstance(result, RolloutResult)
    assert result.states.shape == (2, 2, 9, 8, 8)
    assert result.final_state.shape == (2, 9, 8, 8)


def test_solve_rollout_preserves_initial_state_and_records_times() -> None:
    trajectory_model = make_trajectory_model()
    z0 = torch.randn(2, 9, 8, 8)

    result = solve_rollout(trajectory_model, z0, t0=-2.0, t1=0.5, config=SolverConfig(method="euler"))

    assert torch.allclose(result.states[0], z0)
    assert result.t0 == pytest.approx(-2.0)
    assert result.t1 == pytest.approx(0.5)


def test_solve_rollout_final_state_matches_last_state() -> None:
    trajectory_model = make_trajectory_model()
    z0 = torch.randn(2, 9, 8, 8)

    result = solve_rollout(trajectory_model, z0, t0=0.0, t1=1.0, config=SolverConfig(method="euler"))

    assert torch.allclose(result.final_state, result.states[-1])


def test_solve_rollout_zero_init_vector_field_keeps_state_nearly_static() -> None:
    trajectory_model = make_trajectory_model()
    z0 = torch.randn(2, 9, 8, 8)

    result = solve_rollout(
        trajectory_model,
        z0,
        t0=0.0,
        t1=1.0,
        config=SolverConfig(method="adaptive_heun"),
    )

    assert torch.allclose(result.final_state, z0, atol=2e-5)


def test_solve_rollout_gradients_flow_to_wrapped_unet_parameters() -> None:
    trajectory_model = make_trajectory_model()
    model = trajectory_model.odefunc.vector_field
    with torch.no_grad():
        model.final_conv[-1].conv.weight.fill_(0.01)
        model.final_conv[-1].conv.bias.zero_()

    z0 = torch.randn(2, 9, 8, 8, requires_grad=True)

    result = solve_rollout(
        trajectory_model,
        z0,
        t0=0.0,
        t1=1.0,
        config=SolverConfig(method="euler"),
    )
    result.final_state.sum().backward()

    assert z0.grad is not None
    assert not torch.isnan(z0.grad).any()
    nonzero_grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
    ]
    assert nonzero_grads


@pytest.mark.parametrize("method", ["euler", "adaptive_heun"])
def test_solve_rollout_supports_configured_solver_methods(method: str) -> None:
    trajectory_model = make_trajectory_model()
    z0 = torch.randn(1, 9, 8, 8)

    result = solve_rollout(trajectory_model, z0, t0=0.0, t1=1.0, config=SolverConfig(method=method))

    assert result.states.shape == (2, 1, 9, 8, 8)
