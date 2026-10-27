"""Persist fallback evidence; CPU profiler events alone never prove fallback."""

import os
import re
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List

import torch


def fallback_status(text: str) -> Dict[str, Any]:
    """Confirm only explicit operator-specific CPU fallback messages."""
    evidence: List[str] = []
    for line in text.splitlines():
        if re.search(r"\bgather\b|aten::gather", line, re.I) and re.search(
            r"falling back to cpu|fallback to cpu|cpu fallback|pinned to cpu", line, re.I
        ):
            evidence.append(line)
    return {
        "status": "confirmed" if evidence else "unknown",
        "evidence": evidence,
        "note": "No matching message is not proof of device execution. Inspect raw explain, "
        "stderr and RBLN compiler profiler; compiled fused regions may be opaque.",
    }


@contextmanager
def capture_stderr(path: Path) -> Iterator[None]:
    """Capture native SDK stderr as well as Python stderr on the local process."""
    sys.stderr.flush()
    old_fd = os.dup(2)
    try:
        with path.open("w") as stream:
            os.dup2(stream.fileno(), 2)
            yield
            sys.stderr.flush()
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)


def synchronize(device: str) -> None:
    """Wait for pending work before ending device measurements."""
    if device.startswith("rbln"):
        with torch.rbln.device(torch.device(device)):
            torch.rbln.synchronize()


def diagnose(fn: Callable[[], Any], output: Path, name: str, device: str) -> Dict[str, Any]:
    """Run separately from timing, keeping raw evidence and host dispatcher trace."""
    output.mkdir(parents=True, exist_ok=True)
    stderr = output / f"{name}.stderr.log"
    explain_factory = getattr(getattr(torch, "rbln", None), "explain", None)
    result: Dict[str, Any] = {"device": device, "explain_available": explain_factory is not None}
    with capture_stderr(stderr):
        ctx = explain_factory() if explain_factory else nullcontext()
        with ctx as probe:
            value = fn()
            synchronize(device)
        del value
        if probe is not None:
            result["explain_report"] = probe.report()
            result["explain_verdict"] = probe.verdict()
            result["explain_dump"] = probe.dump()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
        ) as profile:
            value = fn()
            synchronize(device)
        del value
    profile.export_chrome_trace(str(output / f"{name}.host-trace.json"))
    (output / f"{name}.host-operators.txt").write_text(
        profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=80)
    )
    result["gather_cpu_fallback"] = fallback_status(stderr.read_text(errors="replace"))
    result["host_trace_warning"] = (
        "aten::gather host events are dispatcher events, not device-placement proof."
    )
    return result


def gather_probe(output: Path, bucket: int, device: str, dtype: torch.dtype) -> Dict[str, Any]:
    """Probe the relative-attention gather shape in eager mode, not the compiled model.

    This isolates SDK eager placement. It does not establish compiled placement.
    """
    source = torch.randn(12, bucket, 512, dtype=dtype).to(device)
    indices = (
        torch.arange(bucket)
        .clamp(max=511)[None, None, :]
        .expand(12, bucket, bucket)
        .contiguous()
        .to(device)
    )

    def run() -> torch.Tensor:
        return torch.gather(source, -1, indices)

    run()
    synchronize(device)
    result = diagnose(run, output, "gather-eager-probe", device)
    result["scope"] = "eager gather only; do not extrapolate to compiled graph"
    result["source_shape"] = list(source.shape)
    result["index_shape"] = list(indices.shape)
    return result
