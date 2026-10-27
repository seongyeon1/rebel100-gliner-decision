"""Print a standalone benchmark summary without requiring torch or SDK imports."""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple


def percentage(value: Optional[float]) -> str:
    """Represent missing gold accuracy without confusing it with CPU agreement."""
    return "정답 없음" if value is None else f"{value:.2%}"


def render_summary(root: Path) -> Tuple[str, int]:
    """Summarize measurements only when the benchmark finished successfully."""
    failure = root / "failure.json"
    if failure.exists():
        error = json.loads(failure.read_text()).get("error", "Unknown failure")
        return f"벤치마크 실패\n결과 경로: {root.resolve()}\n\n{error}\n", 1
    path = root / "report.json"
    if not path.exists():
        return f"측정 결과 없음: {path}\n설치/실행 로그를 확인하세요.\n", 1
    report = json.loads(path.read_text())
    if report.get("status") != "completed":
        return f"벤치마크 미완료: {report.get('status')}\n{report.get('error', '')}\n", 1
    lines = ["벤치마크 완료", f"결과 경로: {root.resolve()}", ""]
    for name in ("cpu_original", "cpu_port", "candidate"):
        result = report.get(name)
        if not result:
            continue
        quality = result["quality"]
        timing = result["end_to_end"]["aggregate"]
        lines.extend(
            [
                f"[{name}]",
                f"정답 정확도: {percentage(quality.get('accuracy'))}",
                f"CPU 일치율: {percentage(quality.get('cpu_agreement'))}",
                f"p50/p95: {timing['p50_ms']:.3f} / {timing['p95_ms']:.3f} ms",
                f"순차 처리량: {timing['requests_per_second']:.3f} req/s",
                "",
            ]
        )
    candidate = report["candidate"]
    lines.append(f"CPU 포팅 대비 속도비: {candidate['speedup_vs_cpu_port']:.3f}x")
    if "marker_logit_error" in candidate:
        lines.append(f"라벨 logit 오차: {candidate['marker_logit_error']}")
    diagnostics = report.get("diagnostics", {})
    lines.append(f"진단 상태: {diagnostics.get('status', 'unknown')}")
    first = report.get("first_forward_gather_fallback", {})
    lines.append(f"첫 forward gather CPU 폴백: {first.get('status', 'unknown')}")
    for name in ("model_forward", "gather_eager_probe"):
        item = diagnostics.get(name, {})
        fallback = item.get("gather_cpu_fallback", {})
        lines.append(f"{name} gather CPU 폴백: {fallback.get('status', 'unknown')}")
        for evidence in fallback.get("evidence", []):
            lines.append(f"  {evidence}")
        if item.get("explain_verdict") is not None:
            lines.append(
                "  explain: " + json.dumps(item["explain_verdict"], ensure_ascii=False, default=str)
            )
    if diagnostics.get("error"):
        lines.append("진단 오류: " + str(diagnostics["error"]))
    lines.extend(
        [
            "",
            "unknown은 NPU 실행 확인을 뜻하지 않습니다.",
            "eager probe 결과는 컴파일된 모델 내부의 실행 위치를 증명하지 않습니다.",
        ]
    )
    return "\n".join(lines) + "\n", 0


def main() -> int:
    """Write summary.txt and display the same text on the terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    text, status = render_summary(args.directory)
    args.directory.mkdir(parents=True, exist_ok=True)
    (args.directory / "summary.txt").write_text(text)
    print(text, end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
