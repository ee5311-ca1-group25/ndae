"""Validation helpers for NDAE configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ndae.rendering import RENDERER_REGISTRY, select_renderer

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
    ensure_float(config.data.t_I, "data.t_I")
    ensure_float(config.data.t_S, "data.t_S")
    ensure_float(config.data.t_E, "data.t_E")
    if not (config.data.t_I < config.data.t_S < config.data.t_E):
        raise ConfigError("data timeline must satisfy t_I < t_S < t_E")
    if config.data.crop_size > config.data.image_size:
        raise ConfigError("data.crop_size must be less than or equal to data.image_size")
    validate_dataset_layout(config, base_dir=base_dir)

    ensure_positive_int(config.model.dim, "model.dim")
    ensure_non_empty_string(config.model.solver, "model.solver")
    validate_rendering_config(config)

    ensure_positive_int(config.train.batch_size, "train.batch_size")
    ensure_positive_float(config.train.lr, "train.lr")
    ensure_bool(config.train.dry_run, "train.dry_run")
    ensure_positive_int(config.train.n_iter, "train.n_iter")
    ensure_non_negative_int(config.train.n_init_iter, "train.n_init_iter")
    if config.train.n_init_iter > config.train.n_iter:
        raise ConfigError("train.n_init_iter must be less than or equal to train.n_iter")
    ensure_positive_int(config.train.log_every, "train.log_every")
    ensure_positive_int(config.train.checkpoint_every, "train.checkpoint_every")
    ensure_positive_int(config.train.sample_every, "train.sample_every")
    ensure_positive_int(config.train.sample_size, "train.sample_size")
    if config.train.resume_from is not None:
        ensure_non_empty_string(config.train.resume_from, "train.resume_from")


def validate_rendering_config(config: NDAEConfig) -> None:
    renderer_type = config.rendering.renderer_type
    ensure_non_empty_string(renderer_type, "rendering.renderer_type")
    if renderer_type not in RENDERER_REGISTRY:
        supported = ", ".join(RENDERER_REGISTRY)
        raise ConfigError(f"rendering.renderer_type must be one of: {supported}")

    expected_channels = select_renderer(renderer_type).n_brdf_channels
    if config.rendering.n_brdf_channels != expected_channels:
        raise ConfigError(
            "rendering.n_brdf_channels must match the selected renderer_type: "
            f"expected {expected_channels}, got {config.rendering.n_brdf_channels}"
        )

    ensure_positive_int(config.rendering.n_normal_channels, "rendering.n_normal_channels")
    ensure_non_negative_int(config.rendering.n_aug_channels, "rendering.n_aug_channels")
    ensure_positive_float(config.rendering.camera_fov, "rendering.camera_fov")
    ensure_positive_float(config.rendering.camera_distance, "rendering.camera_distance")
    ensure_float(config.rendering.light_intensity, "rendering.light_intensity")
    ensure_positive_float(config.rendering.height_scale, "rendering.height_scale")
    ensure_positive_float(config.rendering.gamma, "rendering.gamma")

    position = config.rendering.light_xy_position
    if not isinstance(position, tuple) or len(position) != 2:
        raise ConfigError("rendering.light_xy_position must be a tuple of two finite floats")

    x, y = position
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ConfigError("rendering.light_xy_position must be a tuple of two finite floats")
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        raise ConfigError("rendering.light_xy_position must be a tuple of two finite floats")


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


def ensure_float(value: float, field_name: str) -> None:
    if type(value) not in {int, float}:
        raise ConfigError(f"{field_name} must be a float")


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

    available_images = resolve_available_images(exemplar_dir, exemplar=config.data.exemplar)
    image_count = len(available_images)
    if image_count == 0:
        raise ConfigError(f"data.exemplar contains no usable image files: {exemplar_dir}")
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


def resolve_available_images(exemplar_dir: Path, *, exemplar: str) -> list[Path]:
    """Resolve available exemplar images, preferring _manifest.json when present."""
    manifest_path = exemplar_dir / "_manifest.json"
    if manifest_path.exists():
        return load_manifest_images(manifest_path, exemplar_dir=exemplar_dir, exemplar=exemplar)

    return sorted(
        path
        for path in exemplar_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_manifest_images(
    manifest_path: Path,
    *,
    exemplar_dir: Path,
    exemplar: str,
) -> list[Path]:
    """Load the manifest-declared image set for an exemplar directory."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid manifest JSON: {manifest_path}") from exc

    if not isinstance(payload, dict):
        raise ConfigError(f"Manifest must be a JSON object: {manifest_path}")

    selected_files = payload.get("selected_files")
    if not isinstance(selected_files, list):
        raise ConfigError(f"Manifest selected_files must be a list: {manifest_path}")

    resolved: list[Path] = []
    for entry in selected_files:
        if not isinstance(entry, str) or not entry.strip():
            raise ConfigError(f"Manifest selected_files entries must be non-empty strings: {manifest_path}")

        relative_path = Path(entry)
        if relative_path.parts and relative_path.parts[0] == exemplar:
            relative_path = Path(*relative_path.parts[1:])
        elif len(relative_path.parts) > 1:
            raise ConfigError(
                "Manifest selected_files entry must point inside the configured exemplar: "
                f"{entry}"
            )

        candidate = exemplar_dir / relative_path
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ConfigError(f"Manifest entry is not a supported image file: {entry}")
        if not candidate.exists():
            raise ConfigError(f"Manifest entry does not exist on disk: {candidate}")
        if not candidate.is_file():
            raise ConfigError(f"Manifest entry is not a file: {candidate}")
        resolved.append(candidate)

    return resolved
