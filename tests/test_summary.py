"""The one-command workflow must distinguish failure from measured results."""

import json
from pathlib import Path


def test_summary_keeps_unknown_fallback_and_unlabeled_accuracy(tmp_path: Path) -> None:
    from summarize import render_summary

    report = {
        "status": "completed",
        "candidate": {
            "quality": {"accuracy": None, "cpu_agreement": 0.9},
            "end_to_end": {"aggregate": {"p50_ms": 10, "p95_ms": 20, "requests_per_second": 50}},
            "speedup_vs_cpu_port": 2.0,
        },
        "diagnostics": {"status": "not_requested"},
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    summary, status = render_summary(tmp_path)
    assert status == 0
    assert "정답 없음" in summary
    assert "unknown" in summary
    assert "2.000" in summary


def test_failure_does_not_render_speedup(tmp_path: Path) -> None:
    from summarize import render_summary

    (tmp_path / "failure.json").write_text(json.dumps({"error": "compile failed"}))
    summary, status = render_summary(tmp_path)
    assert status == 1
    assert "compile failed" in summary
    assert "속도비" not in summary
