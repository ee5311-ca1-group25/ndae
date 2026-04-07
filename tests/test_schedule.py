import pytest
import torch

from ndae.training.schedule import RefreshSchedule, RolloutWindow, StageConfig


def test_stage_config_defaults_match_lecture7_plan() -> None:
    config = StageConfig()

    assert config.t_init == pytest.approx(-2.0)
    assert config.t_start == pytest.approx(0.0)
    assert config.t_end == pytest.approx(10.0)
    assert config.refresh_rate == 6


def test_stage_config_rejects_equal_t_init_and_t_start() -> None:
    with pytest.raises(ValueError, match="t_init must be < t_start"):
        StageConfig(t_init=0.0, t_start=0.0)


def test_stage_config_rejects_t_start_not_before_t_end() -> None:
    with pytest.raises(ValueError, match="t_start must be < t_end"):
        StageConfig(t_start=10.0, t_end=5.0)


def test_stage_config_rejects_refresh_rate_smaller_than_two() -> None:
    with pytest.raises(ValueError, match="refresh_rate must be >= 2"):
        StageConfig(refresh_rate=1)


@pytest.mark.parametrize(
    ("kind", "t0", "t1", "refresh"),
    [
        ("warmup", -2.0, 0.0, True),
        ("generation", 0.0, 1.5, False),
    ],
)
def test_rollout_window_exposes_fields(
    kind: str,
    t0: float,
    t1: float,
    refresh: bool,
) -> None:
    window = RolloutWindow(kind=kind, t0=t0, t1=t1, refresh=refresh)

    assert window.kind == kind
    assert window.t0 == pytest.approx(t0)
    assert window.t1 == pytest.approx(t1)
    assert window.refresh is refresh


def test_refresh_schedule_returns_warmup_window_on_cycle_start() -> None:
    config = StageConfig(refresh_rate=4)
    schedule = RefreshSchedule(config, generator=torch.Generator().manual_seed(0))

    first = schedule.next(iteration=0, carry_time=123.0)
    second = schedule.next(iteration=4, carry_time=-9.0)

    assert first.kind == "warmup"
    assert first.refresh is True
    assert first.t0 == pytest.approx(config.t_init)
    assert first.t1 == pytest.approx(config.t_start)
    assert second.kind == "warmup"
    assert second.refresh is True
    assert second.t0 == pytest.approx(config.t_init)
    assert second.t1 == pytest.approx(config.t_start)


def test_refresh_schedule_returns_generation_window_between_refresh_steps() -> None:
    config = StageConfig(refresh_rate=4)
    schedule = RefreshSchedule(config, generator=torch.Generator().manual_seed(1))

    warmup = schedule.next(iteration=0, carry_time=0.0)
    carry_time = warmup.t1
    windows = []
    for iteration in range(1, config.refresh_rate):
        window = schedule.next(iteration=iteration, carry_time=carry_time)
        windows.append(window)
        carry_time = window.t1

    assert len(windows) == config.refresh_rate - 1
    assert all(window.kind == "generation" for window in windows)
    assert all(window.refresh is False for window in windows)


def test_refresh_schedule_generation_uses_carry_time_as_t0() -> None:
    schedule = RefreshSchedule(StageConfig(), generator=torch.Generator().manual_seed(2))

    schedule.next(iteration=0, carry_time=0.0)
    window = schedule.next(iteration=1, carry_time=3.25)

    assert window.kind == "generation"
    assert window.t0 == pytest.approx(3.25)


def test_refresh_schedule_generation_t1_is_strictly_increasing_within_cycle() -> None:
    config = StageConfig(refresh_rate=6)
    schedule = RefreshSchedule(config, generator=torch.Generator().manual_seed(3))

    warmup = schedule.next(iteration=0, carry_time=0.0)
    carry_time = warmup.t1
    t1_values = []
    for iteration in range(1, config.refresh_rate):
        window = schedule.next(iteration=iteration, carry_time=carry_time)
        t1_values.append(window.t1)
        carry_time = window.t1

    assert all(curr > prev for prev, curr in zip(t1_values, t1_values[1:]))


def test_refresh_schedule_generation_cycle_covers_generation_interval() -> None:
    config = StageConfig(refresh_rate=6)
    schedule = RefreshSchedule(config, generator=torch.Generator().manual_seed(4))

    warmup = schedule.next(iteration=0, carry_time=0.0)
    deltas = schedule._strata_deltas

    assert deltas is not None
    assert torch.all(deltas > 0).item()

    carry_time = warmup.t1
    for iteration in range(1, config.refresh_rate):
        carry_time = schedule.next(iteration=iteration, carry_time=carry_time).t1

    n = config.refresh_rate - 1
    last_stratum_lower = config.t_start + (n - 1) * (config.t_end - config.t_start) / n
    assert carry_time == pytest.approx(config.t_start + deltas.sum().item())
    assert carry_time >= last_stratum_lower
    assert carry_time <= config.t_end


def test_refresh_schedule_is_deterministic_with_seeded_generator() -> None:
    config = StageConfig(refresh_rate=6)
    schedule_a = RefreshSchedule(config, generator=torch.Generator().manual_seed(42))
    schedule_b = RefreshSchedule(config, generator=torch.Generator().manual_seed(42))

    carry_a = schedule_a.next(iteration=0, carry_time=0.0).t1
    carry_b = schedule_b.next(iteration=0, carry_time=0.0).t1
    t1_values_a = []
    t1_values_b = []
    for iteration in range(1, config.refresh_rate):
        window_a = schedule_a.next(iteration=iteration, carry_time=carry_a)
        window_b = schedule_b.next(iteration=iteration, carry_time=carry_b)
        t1_values_a.append(window_a.t1)
        t1_values_b.append(window_b.t1)
        carry_a = window_a.t1
        carry_b = window_b.t1

    assert torch.allclose(torch.tensor(t1_values_a), torch.tensor(t1_values_b))


def test_refresh_schedule_resamples_for_next_cycle() -> None:
    config = StageConfig(refresh_rate=4)
    schedule = RefreshSchedule(config, generator=torch.Generator().manual_seed(5))

    carry_time = schedule.next(iteration=0, carry_time=0.0).t1
    first_cycle_deltas = schedule._strata_deltas
    assert first_cycle_deltas is not None
    for iteration in range(1, config.refresh_rate):
        carry_time = schedule.next(iteration=iteration, carry_time=carry_time).t1

    second_warmup = schedule.next(iteration=config.refresh_rate, carry_time=carry_time)
    second_cycle_deltas = schedule._strata_deltas
    assert second_cycle_deltas is not None
    assert second_cycle_deltas is not first_cycle_deltas
    assert second_warmup.kind == "warmup"
    assert second_warmup.refresh is True

    next_generation = schedule.next(iteration=config.refresh_rate + 1, carry_time=7.0)
    assert next_generation.kind == "generation"
    assert next_generation.t0 == pytest.approx(7.0)
    assert next_generation.t1 > 7.0


def test_refresh_schedule_rejects_generation_before_warmup() -> None:
    schedule = RefreshSchedule(StageConfig())

    with pytest.raises(
        RuntimeError,
        match="RefreshSchedule requires a warmup step before generation steps.",
    ):
        schedule.next(iteration=1, carry_time=0.0)
