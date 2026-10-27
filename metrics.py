"""Ground-truth accuracy, CPU agreement, and serial request latency metrics."""

import math
import statistics
from typing import Any, Dict, List, Sequence


def canonical(value: Any) -> Any:
    """Ignore ordering for multi-label predictions, preserving scalar labels."""
    return tuple(sorted(value)) if isinstance(value, list) else value


def summarize_predictions(
    rows: Sequence[Dict[str, Any]],
    reference: Sequence[Dict[str, Any]],
    candidate: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Measure decision-level exact match; unlabeled accuracy stays null."""
    if not (len(rows) == len(reference) == len(candidate)):
        raise ValueError("Prediction and dataset lengths differ")
    agree = correct = total = labeled = 0
    by_task: Dict[str, List[bool]] = {}
    for row, ref, pred in zip(rows, reference, candidate, strict=False):
        if set(ref) != set(pred):
            raise ValueError("Candidate task keys differ from CPU reference")
        for task in ref:
            total += 1
            agree += canonical(ref[task]) == canonical(pred[task])
        for task, gold in row.get("expected", {}).items():
            if task not in pred:
                raise ValueError(f"Gold task {task} missing from predictions")
            ok = canonical(gold) == canonical(pred[task])
            by_task.setdefault(task, []).append(ok)
            correct += ok
            labeled += 1
    return {
        "accuracy": correct / labeled if labeled else None,
        "accuracy_definition": "per-task exact match against expected labels",
        "labeled_decisions": labeled,
        "cpu_agreement": agree / total if total else None,
        "decisions": total,
        "accuracy_by_task": {k: sum(v) / len(v) for k, v in by_task.items()},
    }


def latency_summary(samples_ms: Sequence[float]) -> Dict[str, Any]:
    """Summarize sequential batch-one timings without inventing concurrency."""
    if not samples_ms or any(x <= 0 or not math.isfinite(x) for x in samples_ms):
        raise ValueError("Latency samples must be finite and positive")
    values = sorted(samples_ms)

    def percentile(q: float) -> float:
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "requests": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "requests_per_second": 1000 * len(values) / sum(values),
    }
