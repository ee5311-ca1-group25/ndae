from pathlib import Path

import pytest

from ndae.config import load_config
from ndae.data.timeline import Timeline


def test_timeline_from_config() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "base.yaml")

    timeline = Timeline.from_config(config.data)

    assert timeline.t_I == -2.0
    assert timeline.t_S == 0.0
    assert timeline.t_E == 10.0
    assert timeline.n_frames == 8
    assert timeline.dt == pytest.approx(1.25)


def test_timeline_properties() -> None:
    timeline = Timeline(t_I=-2.0, t_S=0.0, t_E=10.0, n_frames=100)

    assert timeline.dt == pytest.approx(0.1)
    assert timeline.warmup_duration == pytest.approx(2.0)
    assert timeline.generation_duration == pytest.approx(10.0)
    assert timeline.frame_to_time(0) == pytest.approx(0.0)
    assert timeline.frame_to_time(99) == pytest.approx(9.9)


def test_timeline_round_trip_default_config() -> None:
    timeline = Timeline(t_I=-2.0, t_S=0.0, t_E=10.0, n_frames=100)

    for k in range(100):
        assert timeline.time_to_frame(timeline.frame_to_time(k)) == k


def test_timeline_round_trip_nonzero_t_start() -> None:
    timeline = Timeline(t_I=-3.0, t_S=2.0, t_E=8.0, n_frames=12)

    for k in range(12):
        assert timeline.time_to_frame(timeline.frame_to_time(k)) == k


def test_timeline_time_to_frame_clamps_before_start_and_after_end() -> None:
    timeline = Timeline(t_I=-2.0, t_S=1.5, t_E=6.5, n_frames=10)

    assert timeline.time_to_frame(-10.0) == 0
    assert timeline.time_to_frame(1.4) == 0
    assert timeline.time_to_frame(6.5) == 9
    assert timeline.time_to_frame(100.0) == 9


def test_timeline_frame_to_time_rejects_out_of_range_index() -> None:
    timeline = Timeline(t_I=-2.0, t_S=0.0, t_E=10.0, n_frames=8)

    with pytest.raises(IndexError, match="frame index out of range"):
        timeline.frame_to_time(-1)

    with pytest.raises(IndexError, match="frame index out of range"):
        timeline.frame_to_time(8)


def test_timeline_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError, match="t_I < t_S < t_E"):
        Timeline(t_I=0.0, t_S=0.0, t_E=1.0, n_frames=1)

    with pytest.raises(ValueError, match="n_frames must be greater than 0"):
        Timeline(t_I=-1.0, t_S=1.0, t_E=2.0, n_frames=0)
