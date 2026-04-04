"""Validation helpers for NDAE configuration."""

from __future__ import annotations

from pathlib import Path

from .errors import ConfigError
from .schema import NDAEConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_config(config: NDAEConfig, *, base_dir: str | Path | None = None) -> None:
    """Validate structural and semantic constraints for an NDAE config."""
    ensure_non_empty_string(config.experiment.name, "experiment.name")
    ensure_non_empty_string(config.experiment.output_root, "experiment.output_root")
    ensure_int(config.experiment.seed, "experiment.seed")

    ensure_non_empty_string(config.data.root, "data.root")
    ensure_non_empty_string(config.data.exemplar, "data.exemplar")
    ensure_positive_int(config.data.image_size, "data.image_size")
    ensure_positive_int(config.data.crop_size, "data.crop_size")
    ensure_positive_int(config.data.n_frames, "data.n_frames")
    if config.data.crop_size > config.data.image_size:
        raise ConfigError("data.crop_size must be less than or equal to data.image_size")
    validate_dataset_layout(config, base_dir=base_dir)

    ensure_positive_int(config.model.dim, "model.dim")
    ensure_non_negative_int(config.model.n_aug_channels, "model.n_aug_channels")
    ensure_non_empty_string(config.model.solver, "model.solver")

    ensure_positive_int(config.train.batch_size, "train.batch_size")
    ensure_positive_float(config.train.lr, "train.lr")
    ensure_bool(config.train.dry_run, "train.dry_run")


def ensure_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string")


def ensure_int(value: int, field_name: str) -> None:
    if type(value) is not int:
        raise ConfigError(f"{field_name} must be an integer")


def ensure_positive_int(value: int, field_name: str) -> None:
    ensure_int(value, field_name)
    if value <= 0:
        raise ConfigError(f"{field_name} must be greater than 0")


def ensure_non_negative_int(value: int, field_name: str) -> None:
    ensure_int(value, field_name)
    if value < 0:
        raise ConfigError(f"{field_name} must be greater than or equal to 0")


def ensure_positive_float(value: float, field_name: str) -> None:
    if type(value) not in {int, float}:
        raise ConfigError(f"{field_name} must be a float")
    if float(value) <= 0.0:
        raise ConfigError(f"{field_name} must be greater than 0")


def ensure_bool(value: bool, field_name: str) -> None:
    if type(value) is not bool:
        raise ConfigError(f"{field_name} must be a boolean")


def validate_dataset_layout(config: NDAEConfig, *, base_dir: str | Path | None) -> None:
    """Validate dataset root/exemplar presence and image-count constraints."""
    root_dir = resolve_data_root(config.data.root, base_dir=base_dir)
    if not root_dir.exists():
        raise ConfigError(f"data.root does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise ConfigError(f"data.root must be a directory: {root_dir}")

    exemplar_dir = root_dir / config.data.exemplar
    if not exemplar_dir.exists():
        raise ConfigError(
            f"data.exemplar directory does not exist under data.root: {exemplar_dir}"
        )
    if not exemplar_dir.is_dir():
        raise ConfigError(f"data.exemplar must resolve to a directory: {exemplar_dir}")

    image_count = count_image_files(exemplar_dir)
    if image_count == 0:
        raise ConfigError(f"data.exemplar contains no image files: {exemplar_dir}")
    if config.data.n_frames > image_count:
        raise ConfigError(
            "data.n_frames exceeds available images in data.exemplar: "
            f"requested {config.data.n_frames}, found {image_count}"
        )


def resolve_data_root(root: str, *, base_dir: str | Path | None) -> Path:
    """Resolve a data root against an optional base directory."""
    root_path = Path(root)
    if root_path.is_absolute():
        return root_path
    anchor = Path(base_dir) if base_dir is not None else Path.cwd()
    return (anchor / root_path).resolve()


def count_image_files(directory: Path) -> int:
    """Count image files directly under the given exemplar directory."""
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
