import json
import subprocess
import sys
from pathlib import Path


def test_plot_metrics_script_writes_png(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    output_path = tmp_path / "plots" / "loss_curve.png"
    payloads = [
        {
            "global_step": 1,
            "loss_total": 1.25,
            "loss_init": 1.00,
            "loss_local": 0.00,
            "loss_overflow": 0.25,
        },
        {
            "global_step": 2,
            "loss_total": 0.90,
            "loss_init": 0.00,
            "loss_local": 0.70,
            "loss_overflow": 0.20,
        },
        {
            "global_step": 3,
            "loss_total": 0.75,
            "loss_init": 0.00,
            "loss_local": 0.55,
            "loss_overflow": 0.20,
        },
    ]
    metrics_path.write_text(
        "\n".join(json.dumps(payload) for payload in payloads) + "\n",
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_metrics.py",
            str(metrics_path),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert "Rendered loss plot." in result.stdout
    assert str(metrics_path) in result.stdout
    assert str(output_path) in result.stdout
