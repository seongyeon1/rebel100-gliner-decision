"""Missing diagnostics must not be reported as absence of CPU fallback."""

from diagnostics import fallback_status


def test_no_evidence_is_unknown() -> None:
    assert fallback_status("aten::gather CPU profiler event")["status"] == "unknown"


def test_explicit_fallback_is_confirmed() -> None:
    result = fallback_status("INFO aten::gather: falling back to CPU (unsupported)")
    assert result["status"] == "confirmed"
    assert result["evidence"]
