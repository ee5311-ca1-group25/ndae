from dataclasses import FrozenInstanceError

import pytest
import torch

from ndae.models.odefunc import ODEFunction
from ndae.models.trajectory import TrajectoryModel
from ndae.models.unet import NDAEUNet
from ndae.training import RolloutResult, RolloutWindow, SolverConfig
from ndae.training.solver import RolloutResult as RolloutResultImpl
from ndae.training.solver import SolverConfig as SolverConfigImpl
from ndae.training.solver import rollout_generation, rollout_warmup, solve_rollout


def make_trajectory_model() -> TrajectoryModel:
    model = NDAEUNet(in_dim=9, out_dim=9, dim=16, dim_mults=(1, 2), use_attn=False)
    return TrajectoryModel(ODEFunction(model))


def test_solver_config_defaults_match_runtime_defaults() -> None:
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


def test_rollout_warmup_uses_warmup_window_times() -> None:
    trajectory_model = make_trajectory_model()
    z0 = torch.randn(2, 9, 8, 8)
    window = RolloutWindow(kind="warmup", t0=-2.0, t1=0.0, refresh=True)

    result = rollout_warmup(
        trajectory_model,
        z0,
        window,
        SolverConfig(method="euler"),
    )

    assert result.t0 == pytest.approx(-2.0)
    assert result.t1 == pytest.approx(0.0)


def test_rollout_generation_uses_generation_window_times() -> None:
    trajectory_model = make_trajectory_model()
    z0 = torch.randn(2, 9, 8, 8)
    window = RolloutWindow(kind="generation", t0=0.0, t1=1.5, refresh=False)

    result = rollout_generation(
        trajectory_model,
        z0,
        window,
        SolverConfig(method="euler"),
    )

    assert result.t0 == pytest.approx(0.0)
    assert result.t1 == pytest.approx(1.5)


@pytest.mark.parametrize(
    "window",
    [
        RolloutWindow(kind="generation", t0=0.0, t1=1.0, refresh=False),
        RolloutWindow(kind="warmup", t0=-2.0, t1=0.0, refresh=False),
    ],
)
def test_rollout_warmup_rejects_non_warmup_window(window: RolloutWindow) -> None:
    with pytest.raises(
        ValueError,
        match="rollout_warmup expected a warmup window with refresh=True",
    ):
        rollout_warmup(
            make_trajectory_model(),
            torch.randn(1, 9, 8, 8),
            window,
            SolverConfig(method="euler"),
        )


@pytest.mark.parametrize(
    "window",
    [
        RolloutWindow(kind="warmup", t0=-2.0, t1=0.0, refresh=True),
        RolloutWindow(kind="generation", t0=0.0, t1=1.0, refresh=True),
    ],
)
def test_rollout_generation_rejects_non_generation_window(window: RolloutWindow) -> None:
    with pytest.raises(
        ValueError,
        match="rollout_generation expected a generation window with refresh=False",
    ):
        rollout_generation(
            make_trajectory_model(),
            torch.randn(1, 9, 8, 8),
            window,
            SolverConfig(method="euler"),
        )


def test_rollout_stage_wrappers_preserve_zero_init_static_behavior() -> None:
    trajectory_model = make_trajectory_model()
    z0 = torch.randn(2, 9, 8, 8)

    warmup_result = rollout_warmup(
        trajectory_model,
        z0,
        RolloutWindow(kind="warmup", t0=-2.0, t1=0.0, refresh=True),
        SolverConfig(method="adaptive_heun"),
    )
    generation_result = rollout_generation(
        trajectory_model,
        z0,
        RolloutWindow(kind="generation", t0=0.0, t1=1.0, refresh=False),
        SolverConfig(method="adaptive_heun"),
    )

    assert torch.allclose(warmup_result.final_state, z0, atol=2e-5)
    assert torch.allclose(generation_result.final_state, z0, atol=2e-5)


def test_rollout_generation_allows_gradient_flow() -> None:
    trajectory_model = make_trajectory_model()
    model = trajectory_model.odefunc.vector_field
    with torch.no_grad():
        model.final_conv[-1].conv.weight.fill_(0.01)
        model.final_conv[-1].conv.bias.zero_()

    z0 = torch.randn(2, 9, 8, 8, requires_grad=True)
    window = RolloutWindow(kind="generation", t0=0.0, t1=1.0, refresh=False)

    result = rollout_generation(
        trajectory_model,
        z0,
        window,
        SolverConfig(method="euler"),
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
