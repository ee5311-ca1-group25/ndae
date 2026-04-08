from pathlib import Path
import json

import pytest
import torch
from PIL import Image

from ndae.config import DataConfig, load_config
from ndae.data.exemplar import ExemplarDataset
from ndae.data.sampling import (
    apply_crop_spec,
    apply_take_spec,
    random_crop,
    random_take,
    sample_frame_indices,
    sample_random_crop_spec,
    sample_random_take_spec,
    stratified_uniform,
)
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


def test_exemplar_dataset_single_frame_path_uses_first_frame(tmp_path: Path) -> None:
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
                    "example/frame_b.png",
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = ExemplarDataset(tmp_path / "root", "example", n_frames=1, image_size=12)

    assert tuple(path.name for path in dataset.source_paths) == ("frame_c.png",)


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


def test_random_crop_shape_and_determinism() -> None:
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)

    crop1 = random_crop(image, 4, 4, generator=g1)
    crop2 = random_crop(image, 4, 4, generator=g2)

    assert crop1.shape == (3, 4, 4)
    assert torch.equal(crop1, crop2)


def test_random_crop_preserves_spatial_continuity() -> None:
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    crop = random_crop(image, 4, 4, generator=torch.Generator().manual_seed(0))

    found = False
    for top in range(5):
        for left in range(5):
            if torch.equal(crop, image[:, top : top + 4, left : left + 4]):
                found = True
                break
        if found:
            break

    assert found


def test_random_crop_rejects_invalid_shape_or_size() -> None:
    image = torch.zeros(3, 8, 8)

    with pytest.raises(ValueError, match="3D tensor"):
        random_crop(torch.zeros(8, 8), 4, 4)

    with pytest.raises(ValueError, match="greater than 0"):
        random_crop(image, 0, 4)

    with pytest.raises(ValueError, match="less than or equal to image size"):
        random_crop(image, 9, 4)


def test_random_take_shape_and_determinism() -> None:
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)

    result1 = random_take(image, 4, 4, generator=g1)
    result2 = random_take(image, 4, 4, generator=g2)

    assert result1.shape == (3, 4, 4)
    assert torch.equal(result1, result2)


def test_random_take_preserves_values() -> None:
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    result = random_take(image, 4, 4, generator=torch.Generator().manual_seed(7))

    for channel in range(3):
        source_values = set(image[channel].flatten().tolist())
        for value in result[channel].flatten().tolist():
            assert value in source_values


def test_random_take_destroys_spatial_structure() -> None:
    image = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(3, 16, 16)
    result = random_take(image, 8, 8, generator=torch.Generator().manual_seed(42))

    for top in range(9):
        for left in range(9):
            assert not torch.equal(result, image[:, top : top + 8, left : left + 8])


def test_random_take_rejects_invalid_shape_or_sample_size() -> None:
    image = torch.zeros(3, 8, 8)

    with pytest.raises(ValueError, match="3D tensor"):
        random_take(torch.zeros(8, 8), 4, 4)

    with pytest.raises(ValueError, match="greater than 0"):
        random_take(image, 0, 4)

    with pytest.raises(ValueError, match="less than or equal to H \\* W"):
        random_take(image, 9, 9)


def test_sample_random_crop_spec_and_apply_are_deterministic() -> None:
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)

    spec1 = sample_random_crop_spec(image, 4, 4, generator=g1)
    spec2 = sample_random_crop_spec(image, 4, 4, generator=g2)

    assert spec1 == spec2
    assert torch.equal(apply_crop_spec(image, spec1), apply_crop_spec(image, spec2))


def test_sample_random_take_spec_and_apply_are_deterministic() -> None:
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    g1 = torch.Generator().manual_seed(9)
    g2 = torch.Generator().manual_seed(9)

    spec1 = sample_random_take_spec(image, 4, 4, generator=g1)
    spec2 = sample_random_take_spec(image, 4, 4, generator=g2)

    assert spec1.kind == "take"
    assert spec2.kind == "take"
    assert torch.equal(spec1.indices, spec2.indices)
    assert torch.equal(apply_take_spec(image, spec1), apply_take_spec(image, spec2))


def test_stratified_uniform_returns_in_range_ordered_samples() -> None:
    samples = stratified_uniform(
        5,
        0.0,
        10.0,
        generator=torch.Generator().manual_seed(123),
    )

    assert samples.shape == (5,)
    assert samples.dtype == torch.float32
    assert torch.all(samples[1:] > samples[:-1])

    length = 2.0
    for index, sample in enumerate(samples.tolist()):
        assert index * length <= sample < (index + 1) * length


def test_stratified_uniform_is_deterministic_with_seed() -> None:
    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)

    s1 = stratified_uniform(4, -1.0, 3.0, generator=g1)
    s2 = stratified_uniform(4, -1.0, 3.0, generator=g2)

    assert torch.equal(s1, s2)


def test_stratified_uniform_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="n must be greater than 0"):
        stratified_uniform(0, 0.0, 1.0)

    with pytest.raises(ValueError, match="minval must be less than maxval"):
        stratified_uniform(4, 1.0, 1.0)

    with pytest.raises(ValueError, match="minval must be less than maxval"):
        stratified_uniform(4, 2.0, 1.0)


def test_sample_frame_indices_returns_zero_for_refresh_step() -> None:
    assert sample_frame_indices(100, 6, 0) == 0


def test_sample_frame_indices_samples_within_expected_stratum() -> None:
    n_frames = 100
    refresh_rate = 6

    for step_in_cycle in range(1, refresh_rate):
        index = sample_frame_indices(
            n_frames,
            refresh_rate,
            step_in_cycle,
            generator=torch.Generator().manual_seed(step_in_cycle),
        )
        segment = n_frames / (refresh_rate - 1)
        lower = int(segment * (step_in_cycle - 1))
        upper = min(int(segment * step_in_cycle), n_frames - 1)

        assert lower <= index <= upper


def test_sample_frame_indices_is_deterministic_with_seed() -> None:
    g1 = torch.Generator().manual_seed(2026)
    g2 = torch.Generator().manual_seed(2026)

    index1 = sample_frame_indices(100, 6, 3, generator=g1)
    index2 = sample_frame_indices(100, 6, 3, generator=g2)

    assert index1 == index2


def test_sample_frame_indices_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="n_frames must be greater than 0"):
        sample_frame_indices(0, 6, 0)

    with pytest.raises(ValueError, match="refresh_rate must be greater than 1"):
        sample_frame_indices(100, 1, 0)

    with pytest.raises(ValueError, match="step_in_cycle must satisfy"):
        sample_frame_indices(100, 6, 6)
