"""Exemplar image loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ndae.config.schema import DataConfig
from ndae.config.validation import resolve_available_images, resolve_data_root


class ExemplarDataset:
    """Load an exemplar sequence into memory as a `[N, 3, H, W]` tensor."""

    def __init__(
        self,
        root: str | Path,
        exemplar: str,
        *,
        n_frames: int = 100,
        image_size: int = 256,
    ) -> None:
        if n_frames <= 0:
            raise ValueError("n_frames must be greater than 0")
        if image_size <= 0:
            raise ValueError("image_size must be greater than 0")

        root_path = Path(root)
        exemplar_dir = root_path / exemplar
        available_paths = resolve_available_images(exemplar_dir, exemplar=exemplar)
        selected_paths = self._select_frame_paths(available_paths, n_frames)
        frames = [self._load_frame(path, image_size=image_size) for path in selected_paths]

        self.frames = torch.stack(frames, dim=0)
        self.n_frames = len(selected_paths)
        self.image_size = (image_size, image_size)
        self.source_paths = tuple(selected_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        if not -self.n_frames <= index < self.n_frames:
            raise IndexError(f"frame index out of range: {index}")
        return self.frames[index]

    def __len__(self) -> int:
        return self.n_frames

    @classmethod
    def from_config(
        cls,
        data_config: DataConfig,
        *,
        base_dir: str | Path | None = None,
    ) -> ExemplarDataset:
        root_path = resolve_data_root(data_config.root, base_dir=base_dir)
        return cls(
            root=root_path,
            exemplar=data_config.exemplar,
            n_frames=data_config.n_frames,
            image_size=data_config.image_size,
        )

    @staticmethod
    def _select_frame_paths(paths: list[Path], n_frames: int) -> list[Path]:
        available = len(paths)
        if available < n_frames:
            raise ValueError(
                "n_frames exceeds available exemplar images: "
                f"requested {n_frames}, found {available}"
            )
        if n_frames == 1:
            return [paths[0]]
        if available == n_frames:
            return list(paths)

        interval = (available - 1) / (n_frames - 1)
        selected_indices = [0]
        selected_indices.extend(round(i * interval) for i in range(1, n_frames - 1))
        selected_indices.append(available - 1)
        return [paths[index] for index in selected_indices]

    @staticmethod
    def _load_frame(path: Path, *, image_size: int) -> torch.Tensor:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            processed = ExemplarDataset._preprocess_image(rgb_image, image_size=image_size)

        array = np.asarray(processed, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    @staticmethod
    def _preprocess_image(image: Image.Image, *, image_size: int) -> Image.Image:
        new_dimension = min(image.width, image.height)
        left = (image.width - new_dimension) / 2
        top = (image.height - new_dimension) / 2
        right = (image.width + new_dimension) / 2
        bottom = (image.height + new_dimension) / 2

        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((image_size, image_size))
