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

import workloads


class WorkloadTests(unittest.TestCase):
    def test_text_benchmark_matrix_is_complete(self) -> None:
        text = [cell for cell in workloads.cells() if cell["lane"] == "paired_http"]
        self.assertEqual(3 * 5 * 3, len(text))
        triples = {(cell["batch_size"], cell["target_expanded_tokens"], cell["corpus"]) for cell in text}
        self.assertEqual(45, len(triples))

    def test_quality_matrix_covers_modalities_tasks_and_mrl(self) -> None:
        cells = workloads.cells()
        ids = {cell["id"] for cell in cells}
        self.assertTrue({"image-32x32", "image-1536x256", "image-256x1536"} <= ids)
        self.assertTrue({"audio-0p1s-16000hz", "audio-31s-16000hz", "audio-1s-8000hz", "audio-10s-48000hz"} <= ids)
        self.assertTrue({"mixed-text-image-audio", "mixed-audio-text-image", "mixed-image-image-text", "mixed-audio-audio-text"} <= ids)
        self.assertEqual(set(workloads.PROMPTS), {cell["task_type"] for cell in cells if cell["id"].startswith("task-")})
        self.assertEqual(set(workloads.TRAINED_DIMENSIONS), {cell["dimensions"] for cell in cells if cell["id"].startswith("mrl-")})

    def test_negative_contract_matrix_is_fail_closed(self) -> None:
        negatives = workloads.negative_cases()
        self.assertEqual(len(negatives), len({case["id"] for case in negatives}))
        self.assertEqual(workloads.MAX_TOKENS + 1, next(case["target_expanded_tokens"] for case in negatives if case["id"] == "expanded-8193"))
        self.assertTrue(all(case["expect"] in {"reject", "subsequent_success"} for case in negatives))


if __name__ == "__main__":
    unittest.main()
