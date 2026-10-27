"""Regression checks for honest accuracy and latency reporting."""

import importlib.util
import unittest


class MetricsTests(unittest.TestCase):
    def test_module_exists(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("metrics"), "metrics not implemented")

    def test_unlabeled_accuracy_is_unknown(self) -> None:
        from metrics import summarize_predictions

        rows = [{"tasks": {"route": ["search", "sql"]}}]
        predictions = [{"route": "search"}]
        report = summarize_predictions(rows, predictions, predictions)
        self.assertIsNone(report["accuracy"])
        self.assertEqual(report["cpu_agreement"], 1.0)
        self.assertEqual(report["labeled_decisions"], 0)

    def test_gold_not_cpu_defines_accuracy(self) -> None:
        from metrics import summarize_predictions

        rows = [{"expected": {"route": "sql", "tags": ["a", "b"]}}]
        cpu = [{"route": "search", "tags": ["a", "b"]}]
        candidate = [{"route": "sql", "tags": ["b", "a"]}]
        result = summarize_predictions(rows, cpu, candidate)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["cpu_agreement"], 0.5)

    def test_missing_task_rejected(self) -> None:
        from metrics import summarize_predictions

        with self.assertRaises(ValueError):
            summarize_predictions([{}], [{"route": "a"}], [{}])

    def test_latency_throughput(self) -> None:
        from metrics import latency_summary

        result = latency_summary([10.0, 20.0, 30.0])
        self.assertEqual(result["p50_ms"], 20.0)
        self.assertAlmostEqual(result["requests_per_second"], 50.0)
