from pathlib import Path

from ndae.cli.train import run_train_cli


def test_dry_run_creates_workspace_and_resolved_config(tmp_path: Path, capsys) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "base.yaml"

    exit_code = run_train_cli(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0

    workspace = tmp_path / "lecture1_smoke"
    resolved_config = workspace / "config.resolved.yaml"

    assert workspace.is_dir()
    assert resolved_config.is_file()

    output = capsys.readouterr().out
    assert "workspace:" in output
    assert str(workspace) in output
    assert "data.exemplar: clay_solidifying" in output
    assert "Dry run completed." in output

    resolved_text = resolved_config.read_text(encoding="utf-8")
    assert "lecture1_smoke" in resolved_text
    assert "exemplar: clay_solidifying" in resolved_text
    assert f"output_root: '{tmp_path}'" not in resolved_text
    assert f"output_root: {tmp_path}" in resolved_text
