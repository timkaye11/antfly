#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchmark_gemma4_cuda_batching import (
    ROW_TWO_COUNTER,
    SCHEDULER_COUNTERS,
    evaluate_acceptance,
    parse_prometheus_counters,
    require_scheduler_counters,
)


def scheduler_delta(row_two_steps: int = 0) -> dict[str, int]:
    counters = {name: 0 for name in SCHEDULER_COUNTERS}
    counters[ROW_TWO_COUNTER] = row_two_steps
    return counters


def measurement(tok_s: float, p95_ms: float, fingerprint: str, row_two_steps: int = 0) -> dict:
    return {
        "aggregate_tok_s": {"median": tok_s},
        "request_latency_ms": {"p95": p95_ms},
        "fingerprints": [fingerprint],
        "scheduler_counter_delta": scheduler_delta(row_two_steps),
    }


class PrometheusCounterTests(unittest.TestCase):
    def test_parser_keeps_unlabelled_counters_only(self) -> None:
        text = """
# HELP requests_total Request count
# TYPE requests_total counter
requests_total 12
# TYPE latency gauge
latency 4
# TYPE labelled_total counter
labelled_total{route="generate"} 7
# TYPE fractional_total counter
fractional_total 1.5
"""
        self.assertEqual(
            {"requests_total": 12, "fractional_total": 1.5},
            parse_prometheus_counters(text),
        )

    def test_required_scheduler_counters_reject_missing_samples(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing counters"):
            require_scheduler_counters("# TYPE unrelated_total counter\nunrelated_total 1\n")

    def test_required_scheduler_counters_accept_complete_exposition(self) -> None:
        text = "\n".join(
            line
            for name in SCHEDULER_COUNTERS
            for line in (f"# TYPE {name} counter", f"{name} 3")
        )
        self.assertEqual({name: 3 for name in SCHEDULER_COUNTERS}, require_scheduler_counters(text))


class AcceptanceTests(unittest.TestCase):
    def test_c2_row_batch_and_throughput_pass_while_c4_is_diagnostic(self) -> None:
        baseline = {"measurements": {"1": measurement(100.0, 100.0, "same")}}
        batched = {
            "measurements": {
                "1": measurement(99.0, 103.0, "same"),
                "2": measurement(160.0, 180.0, "same", row_two_steps=12),
                "4": measurement(1.0, 10_000.0, "different"),
            }
        }

        acceptance = evaluate_acceptance(baseline, batched, 1.5, 1.05)

        self.assertTrue(acceptance["passed"])
        self.assertEqual(12, acceptance["c2_step_batch_size_2_total"])
        self.assertFalse(acceptance["concurrency_4_plus_diagnostics"]["4"]["exact_response_fingerprints"])

    def test_missing_row_two_steps_fails(self) -> None:
        baseline = {"measurements": {"1": measurement(100.0, 100.0, "same")}}
        batched = {
            "measurements": {
                "1": measurement(100.0, 100.0, "same"),
                "2": measurement(180.0, 180.0, "same", row_two_steps=0),
            }
        }

        acceptance = evaluate_acceptance(baseline, batched, 1.5, 1.05)

        self.assertFalse(acceptance["passed"])
        self.assertFalse(acceptance["c2_row_two_steps_positive"])

    def test_c2_speedup_and_exact_fingerprint_are_gating(self) -> None:
        baseline = {"measurements": {"1": measurement(100.0, 100.0, "same")}}
        batched = {
            "measurements": {
                "1": measurement(100.0, 100.0, "same"),
                "2": measurement(140.0, 180.0, "different", row_two_steps=8),
            }
        }

        acceptance = evaluate_acceptance(baseline, batched, 1.5, 1.05)

        self.assertFalse(acceptance["passed"])
        self.assertFalse(acceptance["c2_exact_response_fingerprints"])
        self.assertLess(acceptance["c2_aggregate_speedup"], 1.5)


if __name__ == "__main__":
    unittest.main()
