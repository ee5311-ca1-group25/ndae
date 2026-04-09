#!/usr/bin/env python3
"""Generate an NDAE exemplar manifest from an existing local image directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_svbrdf_mini import (
    DATASET_DOI_URL,
    DATASET_DOWNLOAD_URL,
    write_manifest,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate _manifest.json for a local exemplar directory.",
    )
    parser.add_argument(
        "exemplar_dir",
        help="Directory that contains one exemplar's image files.",
    )
    parser.add_argument(
        "--exemplar",
        default=None,
        help="Optional exemplar name override. Defaults to the directory name.",
    )
    parser.add_argument(
        "--page-url",
        default=DATASET_DOI_URL,
        help="Source dataset page URL recorded into the manifest.",
    )
    parser.add_argument(
        "--download-url",
        default=DATASET_DOWNLOAD_URL,
        help="Source download URL recorded into the manifest.",
    )
    return parser


def resolve_selected_files(exemplar_dir: Path, exemplar: str) -> list[str]:
    selected_files = sorted(
        f"{exemplar}/{path.name}"
        for path in exemplar_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not selected_files:
        raise ValueError(f"No image files found under {exemplar_dir}")
    return selected_files


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    exemplar_dir = Path(args.exemplar_dir).expanduser().resolve()
    if not exemplar_dir.exists():
        parser.error(f"Exemplar directory does not exist: {exemplar_dir}")
    if not exemplar_dir.is_dir():
        parser.error(f"Exemplar path must be a directory: {exemplar_dir}")

    exemplar = args.exemplar or exemplar_dir.name
    selected_files = resolve_selected_files(exemplar_dir, exemplar)
    write_manifest(
        exemplar,
        selected_files,
        exemplar_dir,
        args.page_url,
        args.download_url,
    )
    print(f"selected_files: {len(selected_files)}")
    print(f"manifest: {exemplar_dir / '_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
