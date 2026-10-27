"""Exercise the actual CLI with a saved offline GLiNER checkpoint."""

import json
from pathlib import Path
from typing import Any

from test_port import model  # noqa: F401  # shared offline fixture

# Imported pytest fixture is injected by name below.
from benchmark import main


def test_complete_cpu_benchmark_and_reports(model: Any, tmp_path: Path) -> None:  # noqa: F811
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint))
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps(
            {
                "id": "offline",
                "text": "文書 조회",
                "tasks": {"route": ["search", "sql"]},
                "expected": {"route": "search"},
            }
        )
    )
    output = tmp_path / "results"
    code = main(
        [
            "--backend",
            "cpu",
            "--dtype",
            "float32",
            "--model",
            str(checkpoint),
            "--data",
            str(data),
            "--output",
            str(output),
            "--bucket",
            "128",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--cpu-threads",
            "1",
        ]
    )
    assert code == 0
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["candidate"]["quality"]["cpu_agreement"] == 1.0
    assert report["candidate"]["marker_logit_error"]["max_abs"] < 1e-6
    assert report["diagnostics"]["model_forward"]["gather_cpu_fallback"]["status"] == "unknown"
    assert (output / "model-forward.host-trace.json").exists()
    assert report["candidate"]["end_to_end"]["aggregate"]["requests"] == 2


def test_failure_is_not_a_successful_measurement(tmp_path: Path) -> None:
    data = tmp_path / "bad.jsonl"
    data.write_text(
        json.dumps(
            {"text": "x", "tasks": {"route": ["a", "b"]}, "expected": {"route": "not-a-label"}}
        )
    )
    output = tmp_path / "results"
    assert main(["--backend", "cpu", "--data", str(data), "--output", str(output)]) == 1
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"


def test_one_command_shell_workflow(model: Any, tmp_path: Path) -> None:  # noqa: F811
    import os
    import subprocess
    import sys

    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint))
    data = tmp_path / "request.jsonl"
    data.write_text(json.dumps({"text": "文書 조회", "tasks": {"route": ["search", "sql"]}}))
    output = tmp_path / "workflow-results"
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        BACKEND="cpu",
        DTYPE="float32",
        SKIP_INSTALL="1",
        SERVING_DEMO="1",
        MODEL=str(checkpoint),
        BUCKET="128",
        CPU_THREADS="1",
        WARMUP="1",
        REPEATS="1",
    )
    result = subprocess.run(
        ["bash", "run_all.sh", str(data), str(output)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "벤치마크 완료" in (output / "summary.txt").read_text()
    assert "정답 없음" in (output / "summary.txt").read_text()
    assert (output / "sdk-constraints.txt").exists()
    assert (output / "environment.txt").exists()
    assert json.loads((output / "http-demo/demo-report.json").read_text())["status"] == "completed"
