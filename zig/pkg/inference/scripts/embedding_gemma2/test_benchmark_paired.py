# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_paired
import qualify_large_batch_endpoint


class BenchmarkTests(unittest.TestCase):
    def test_percentiles_are_stable(self) -> None:
        values = [0.001, 0.002, 0.003, 0.004, 0.005]
        self.assertEqual(0.003, benchmark_paired.percentile(values, 0.5))
        self.assertEqual(0.004, benchmark_paired.percentile(values, 0.95))

    def test_bootstrap_reports_throughput_as_inverse_latency(self) -> None:
        low, high = benchmark_paired.bootstrap_ratio([1.0] * 10, [2.0] * 10, 100, 7)
        self.assertEqual(2.0, low)
        self.assertEqual(2.0, high)

    def test_heterogeneous_corpus_preserves_batch_and_maximum_length(self) -> None:
        rows = benchmark_paired.corpus_inputs("heterogeneous_padding", 8, 128)
        self.assertEqual(8, len(rows))
        lengths = [len(row.split()) for row in rows]
        self.assertEqual(128, max(lengths))
        self.assertGreater(len(set(lengths)), 1)

    def test_code_corpus_is_deterministic(self) -> None:
        left = benchmark_paired.corpus_inputs("source_code", 2, 128)
        right = benchmark_paired.corpus_inputs("source_code", 2, 128)
        self.assertEqual(left, right)
        self.assertIn("cosine", left[0])

    def test_competitive_gate_requires_both_throughput_and_tail_latency(self) -> None:
        self.assertTrue(benchmark_paired.evaluate_gates(0.90, 1.10))
        self.assertFalse(benchmark_paired.evaluate_gates(0.8999, 1.0))
        self.assertFalse(benchmark_paired.evaluate_gates(1.0, 1.1001))

    def test_large_batch_vector_contract_rejects_missing_and_nonfinite_rows(self) -> None:
        valid = {"data": [{"index": 0, "embedding": [0.0] * 768}]}
        self.assertEqual(1, len(qualify_large_batch_endpoint.checked_vectors(valid, 1)))
        with self.assertRaisesRegex(RuntimeError, "expected 2 embedding rows"):
            qualify_large_batch_endpoint.checked_vectors(valid, 2)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            qualify_large_batch_endpoint.checked_vectors(
                {"data": [{"index": 0, "embedding": [float("nan")] * 768}]}, 1
            )


if __name__ == "__main__":
    unittest.main()
