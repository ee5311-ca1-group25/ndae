from pathlib import Path
import json

import pytest
import torch
from PIL import Image

from ndae.config import DataConfig, load_config
from ndae.data.exemplar import ExemplarDataset
from ndae.data.timeline import Timeline


def write_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


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


def test_exemplar_dataset_from_config_loads_manifest_selected_frames() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "base.yaml")
    dataset = ExemplarDataset.from_config(config.data, base_dir=project_root)

    manifest = json.loads(
        (
            project_root
            / "data_local"
            / "svbrdf_mini"
            / "clay_solidifying"
            / "_manifest.json"
        ).read_text(encoding="utf-8")
    )
    expected_paths = tuple(
        (
            project_root / "data_local" / "svbrdf_mini" / Path(relative_path)
        ).resolve()
        for relative_path in manifest["selected_files"]
    )

    assert len(dataset) == 8
    assert dataset.source_paths == expected_paths


def test_exemplar_dataset_returns_expected_shape_dtype_and_range() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "base.yaml")
    dataset = ExemplarDataset.from_config(config.data, base_dir=project_root)

    assert dataset.frames.shape == (8, 3, 256, 256)
    assert dataset.frames.dtype == torch.float32
    assert dataset.frames.min().item() >= 0.0
    assert dataset.frames.max().item() <= 1.0
    assert dataset.image_size == (256, 256)


def test_exemplar_dataset_preserves_manifest_order(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "root" / "example"
    write_image(exemplar_dir / "frame_b.png", size=(24, 16), color=(10, 20, 30))
    write_image(exemplar_dir / "frame_a.png", size=(24, 16), color=(40, 50, 60))
    write_image(exemplar_dir / "frame_c.png", size=(24, 16), color=(70, 80, 90))

    (exemplar_dir / "_manifest.json").write_text(
        json.dumps(
            {
                "selected_files": [
                    "example/frame_c.png",
                    "example/frame_a.png",
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = ExemplarDataset(tmp_path / "root", "example", n_frames=2, image_size=12)

    assert tuple(path.name for path in dataset.source_paths) == ("frame_c.png", "frame_a.png")


def test_exemplar_dataset_without_manifest_falls_back_to_sorted_filenames(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "root" / "example"
    write_image(exemplar_dir / "frame_b.png", size=(24, 16), color=(10, 20, 30))
    write_image(exemplar_dir / "frame_a.png", size=(24, 16), color=(40, 50, 60))
    write_image(exemplar_dir / "frame_c.png", size=(24, 16), color=(70, 80, 90))

    dataset = ExemplarDataset(tmp_path / "root", "example", n_frames=3, image_size=12)

    assert tuple(path.name for path in dataset.source_paths) == (
        "frame_a.png",
        "frame_b.png",
        "frame_c.png",
    )


def test_exemplar_dataset_uniformly_samples_when_more_images_than_requested(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "root" / "example"
    for index in range(5):
        write_image(
            exemplar_dir / f"frame_{index:04d}.png",
            size=(18, 12),
            color=(index, index, index),
        )

    dataset = ExemplarDataset(tmp_path / "root", "example", n_frames=3, image_size=10)

    assert tuple(path.name for path in dataset.source_paths) == (
        "frame_0000.png",
        "frame_0002.png",
        "frame_0004.png",
    )


def test_exemplar_dataset_rejects_insufficient_available_frames(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "root" / "example"
    write_image(exemplar_dir / "frame_0000.png", size=(16, 16), color=(1, 2, 3))

    with pytest.raises(ValueError, match="requested 2, found 1"):
        ExemplarDataset(tmp_path / "root", "example", n_frames=2, image_size=8)


def test_exemplar_dataset_supports_negative_index_and_rejects_out_of_range_index(
    tmp_path: Path,
) -> None:
    exemplar_dir = tmp_path / "root" / "example"
    for index in range(3):
        write_image(
            exemplar_dir / f"frame_{index:04d}.png",
            size=(20, 14),
            color=(index * 20, index * 20, index * 20),
        )

    dataset = ExemplarDataset(tmp_path / "root", "example", n_frames=3, image_size=10)

    assert torch.equal(dataset[-1], dataset.frames[-1])

    with pytest.raises(IndexError, match="frame index out of range"):
        _ = dataset[len(dataset)]


def test_exemplar_dataset_from_config_resolves_relative_root_with_base_dir(tmp_path: Path) -> None:
    exemplar_dir = tmp_path / "data_root" / "example"
    write_image(exemplar_dir / "frame_0000.png", size=(16, 24), color=(10, 20, 30))
    write_image(exemplar_dir / "frame_0001.png", size=(16, 24), color=(40, 50, 60))

    config = DataConfig(
        root="data_root",
        exemplar="example",
        image_size=12,
        crop_size=8,
        n_frames=2,
    )

    dataset = ExemplarDataset.from_config(config, base_dir=tmp_path)

    assert len(dataset) == 2
    assert dataset.frames.shape == (2, 3, 12, 12)
