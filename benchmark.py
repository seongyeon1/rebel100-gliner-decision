"""Benchmark original CPU GLiNER, CPU port and Rebel100 classification forward."""

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Must be set before importing the SDK. Native fallback logs are retained below.
os.environ.setdefault("TORCH_RBLN_LOG_LEVEL", "INFO")

import torch

from diagnostics import capture_stderr, diagnose, fallback_status, gather_probe, synchronize
from metrics import canonical, latency_summary, summarize_predictions
from port import (
    MODEL_ID,
    MODEL_REVISION,
    ClassificationOnly,
    load_model,
    predict,
    prepare,
)


def write_json(path: Path, value: Any) -> None:
    """Persist raw results, including SDK objects represented as strings."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Read requests; reject bad gold labels instead of silently dropping them."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Dataset is empty")
    for i, row in enumerate(rows):
        if not isinstance(row.get("text"), str) or not row.get("tasks"):
            raise ValueError(f"Row {i}: text and nonempty tasks required")
        for task, gold in row.get("expected", {}).items():
            config = row["tasks"].get(task)
            if config is None:
                raise ValueError(f"Row {i}: unknown gold task {task}")
            labels = config["labels"] if isinstance(config, dict) else config
            values = gold if isinstance(gold, list) else [gold]
            if not values or any(value not in labels for value in values):
                raise ValueError(f"Row {i}: invalid expected label for {task}")
            multi = isinstance(config, dict) and config.get("multi_label", False)
            if isinstance(gold, list) != bool(multi):
                raise ValueError(f"Row {i}: expected must be a list exactly for multi-label tasks")
    return rows


def timed(fn: Callable[[], Any], device: str) -> Tuple[Any, float]:
    """Time a completed request, excluding previous pending device work."""
    synchronize(device)
    start = time.perf_counter_ns()
    value = fn()
    synchronize(device)
    return value, (time.perf_counter_ns() - start) / 1e6


def measure(
    functions: List[Callable[[], Any]], device: str, warmup: int, repeats: int
) -> Dict[str, Any]:
    """Warm every request; report batch-one serial steady-state timings."""
    for fn in functions:
        for _ in range(warmup):
            fn()
            synchronize(device)
    all_ms: List[float] = []
    per_request: List[List[float]] = [[] for _ in functions]
    for _ in range(repeats):
        for i, fn in enumerate(functions):
            _, elapsed = timed(fn, device)
            all_ms.append(elapsed)
            per_request[i].append(elapsed)
    return {
        "aggregate": latency_summary(all_ms),
        "per_request": [latency_summary(values) for values in per_request],
        "raw_ms": per_request,
    }


def public_predictions(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize original API confidence payloads to labels."""
    return {
        task: [item["label"] for item in value] if isinstance(value, list) else value["label"]
        for task, value in result.items()
    }


def run(args: argparse.Namespace, output: Path) -> Dict[str, Any]:
    """Run CPU stages first, then move the shared encoder/head to the target."""
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    rows = load_rows(args.data)
    report: Dict[str, Any] = {
        "status": "running",
        "arguments": vars(args).copy(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
        "model": {"id": args.model, "revision": args.revision},
        "timing": {
            "batch_size": 1,
            "steady_state_only": True,
            "order": "cpu_original, cpu_port, candidate; no concurrency",
        },
    }
    for package in ["gliner2", "transformers", "torch-rbln", "rebel-compiler"]:
        try:
            report["environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["environment"][package] = None
    write_json(output / "report.json", report)
    print("Loading pinned checkpoint...", flush=True)
    model = load_model(args.model, args.revision)
    temperature = float(model.boundary_settings.classification_temperature)
    prepared = [prepare(model, row, args.bucket) for row in rows]
    report["tokens_per_request"] = [item.tokens for item in prepared]
    cpu_port = ClassificationOnly(model)

    def original(row: Dict[str, Any]) -> Dict[str, Any]:
        return model.classify_text(row["text"], row["tasks"], include_confidence=True)

    def cpu_e2e(row: Dict[str, Any]) -> Any:
        return predict(cpu_port, prepare(model, row, args.bucket), "cpu", temperature)

    original_fns = [lambda row=row: original(row) for row in rows]
    cpu_fns = [lambda row=row: cpu_e2e(row) for row in rows]
    original_results = [fn() for fn in original_fns]
    reference = [public_predictions(value) for value in original_results]
    cpu_results = [fn() for fn in cpu_fns]
    cpu_predictions = [value[0] for value in cpu_results]
    report["cpu_original"] = {
        "quality": summarize_predictions(rows, reference, reference),
        "end_to_end": measure(original_fns, "cpu", args.warmup, args.repeats),
    }
    report["cpu_port"] = {
        "quality": summarize_predictions(rows, reference, cpu_predictions),
        "end_to_end": measure(cpu_fns, "cpu", args.warmup, args.repeats),
    }
    # Probability comparison catches marker errors even when argmax is unchanged.
    parity_errors = []
    for row_index, (public, result) in enumerate(zip(original_results, cpu_results, strict=True)):
        for task, value in public.items():
            chosen = value if isinstance(value, list) else [value]
            for item in chosen:
                error = abs(result[1][task][item["label"]] - item["confidence"])
                parity_errors.append(error)
                if error > args.cpu_parity_atol:
                    raise RuntimeError(
                        f"CPU port probability parity failed at row {row_index}, task {task}: {error}"
                    )
    report["cpu_port"]["max_selected_probability_error"] = max(parity_errors, default=0)
    if any(
        canonical(ref[k]) != canonical(pred[k])
        for ref, pred in zip(reference, cpu_predictions, strict=True)
        for k in ref
    ):
        raise RuntimeError(
            "CPU port labels differ from original API; refusing an invalid NPU comparison"
        )
    write_json(output / "report.json", report)
    print("CPU parity passed. Preparing candidate...", flush=True)
    device = "cpu" if args.backend == "cpu" else f"rbln:{args.device}"
    dtype = getattr(torch, args.dtype)
    if args.backend != "cpu":
        import rebel  # noqa: F401  # registers the compiler backend
        import torch_rbln  # noqa: F401  # registers RBLN device and diagnostics

        if not torch.rbln.is_available():
            raise RuntimeError("RBLN device unavailable; inspect rbln-smi and SDK version pairing")
        report["environment"]["device"] = torch.rbln.get_device_name(args.device)
    module: Any = cpu_port.to(dtype=dtype, device=device).eval()
    compile_seconds = 0.0
    if args.backend == "rbln-compile":
        # Strict fullgraph: no silent eager continuation after compiler failure.
        module = torch.compile(
            module,
            backend="rbln",
            fullgraph=True,
            dynamic=False,
            options={"mode": "strict", "cache_dir": str(output / "compile-cache")},
        )
    t0 = time.perf_counter()
    with capture_stderr(output / "first-forward.stderr.log"):
        predict(module, prepared[0], device, temperature)
        synchronize(device)
    compile_seconds = time.perf_counter() - t0
    report["first_forward_gather_fallback"] = fallback_status(
        (output / "first-forward.stderr.log").read_text(errors="replace")
    )

    def candidate_e2e(row: Dict[str, Any]) -> Any:
        return predict(module, prepare(model, row, args.bucket), device, temperature)

    candidate_fns = [lambda row=row: candidate_e2e(row) for row in rows]
    results = [fn() for fn in candidate_fns]
    candidate_predictions = [value[0] for value in results]
    report["candidate"] = {
        "backend": args.backend,
        "dtype": args.dtype,
        "first_forward_including_compile_seconds": compile_seconds,
        "quality": summarize_predictions(rows, reference, candidate_predictions),
        "end_to_end": measure(candidate_fns, device, args.warmup, args.repeats),
    }
    # Same padded shapes/CPU postprocessing on CPU port vs candidate, excluding preprocessing.
    report["candidate"]["prepared_request"] = measure(
        [lambda item=item: predict(module, item, device, temperature) for item in prepared],
        device,
        args.warmup,
        args.repeats,
    )
    errors = []
    prediction_rows = []
    for row, prep, cpu, target in zip(rows, prepared, cpu_results, results, strict=True):
        marker_positions = [pos for _, positions, _ in prep.groups for pos in positions]
        errors.extend((cpu[2][0, marker_positions] - target[2][0, marker_positions]).abs().tolist())
        prediction_rows.append(
            {
                "id": row.get("id"),
                "expected": row.get("expected"),
                "cpu_original": reference[len(prediction_rows)],
                "cpu_port": cpu[0],
                "candidate": target[0],
                "cpu_probabilities": cpu[1],
                "candidate_probabilities": target[1],
            }
        )
    report["candidate"]["marker_logit_error"] = {
        "max_abs": max(errors),
        "mean_abs": statistics.mean(errors),
    }
    cpu_mean = report["cpu_port"]["end_to_end"]["aggregate"]["mean_ms"]
    public_mean = report["cpu_original"]["end_to_end"]["aggregate"]["mean_ms"]
    target_mean = report["candidate"]["end_to_end"]["aggregate"]["mean_ms"]
    report["candidate"]["speedup_vs_cpu_port"] = cpu_mean / target_mean
    report["candidate"]["speedup_vs_cpu_original"] = public_mean / target_mean
    write_json(output / "predictions.json", prediction_rows)
    report["diagnostics"] = {"status": "not_requested", "gather_cpu_fallback": "unknown"}
    if not args.no_diagnostics:
        print("Collecting diagnostics separately from latency timings...", flush=True)
        try:
            report["diagnostics"] = {
                "status": "collected",
                "model_forward": diagnose(
                    lambda: predict(module, prepared[0], device, temperature),
                    output,
                    "model-forward",
                    device,
                ),
            }
            if device != "cpu":
                report["diagnostics"]["gather_eager_probe"] = gather_probe(
                    output, args.bucket, device, dtype
                )
        except Exception:
            report["diagnostics"]["error"] = traceback.format_exc()
            report["diagnostics"]["status"] = "partial_or_failed"
    report["status"] = "completed"
    write_json(output / "report.json", report)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    """Write failures to disk and exit nonzero instead of emitting fake results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("examples.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--backend", choices=["cpu", "rbln-compile", "rbln-eager"], default="rbln-compile"
    )
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bucket", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cpu-parity-atol", type=float, default=1e-5)
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    if min(args.bucket, args.cpu_threads, args.repeats) < 1 or args.warmup < 1:
        parser.error("bucket, threads, repeats and warmup must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        parser.error("output contains report.json; use a fresh output directory")
    try:
        with torch.inference_mode():
            report = run(args, args.output)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "candidate": report["candidate"]["quality"],
                    "speedup_vs_cpu_port": report["candidate"]["speedup_vs_cpu_port"],
                    "report": str(args.output / "report.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception:
        error = traceback.format_exc()
        report_path = args.output / "report.json"
        if report_path.exists():
            partial = json.loads(report_path.read_text())
            partial["status"] = "failed"
            partial["error"] = error
            write_json(report_path, partial)
        write_json(
            args.output / "failure.json",
            {
                "status": "failed",
                "error": error,
                "note": "No successful NPU measurement is implied; partial report may contain CPU results.",
            },
        )
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
