import importlib
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_NAMES = [
    "ndae",
    "ndae.config",
    "ndae.cli",
    "ndae.data",
    "ndae.models",
    "ndae.rendering",
    "ndae.losses",
    "ndae.training",
    "ndae.evaluation",
    "ndae.utils",
]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_package_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)


def test_train_cli_stub_returns_success(capsys: pytest.CaptureFixture[str]) -> None:
    from ndae.cli.train import run_train_cli

    assert run_train_cli(["--config", "configs/base.yaml", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "NDAE Lecture 1 Dry Run" in output
    assert "Dry run completed." in output


def test_main_entrypoint_smoke() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "main.py", "--config", "configs/base.yaml", "--dry-run"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "NDAE Lecture 1 Dry Run" in result.stdout
    assert "Dry run completed." in result.stdout
