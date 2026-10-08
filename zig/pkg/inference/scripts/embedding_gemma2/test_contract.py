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

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contract


class ContractTests(unittest.TestCase):
    def test_literal_prompt_spacing_and_pipes(self) -> None:
        self.assertEqual(
            "task: search result | query: ants",
            contract.render_text("ants", "RETRIEVAL_QUERY"),
        )
        self.assertEqual(
            "title: none | text: ants",
            contract.render_text("ants", "RETRIEVAL_DOCUMENT"),
        )
        self.assertEqual(
            "task: code retrieval | query: cosine",
            contract.render_text("cosine", "CODE_RETRIEVAL"),
        )

    def test_unknown_task_fails_closed(self) -> None:
        with self.assertRaises(contract.ContractError):
            contract.render_text("ants", "CUSTOM")

    def test_truncation_renormalizes_arbitrary_valid_prefix(self) -> None:
        vector = [3.0, 4.0] + [1.0] * 6
        reduced = contract.truncate_and_normalize(vector, 2)
        self.assertEqual([0.6, 0.8], reduced)
        self.assertAlmostEqual(1.0, math.sqrt(sum(x * x for x in reduced)))

    def test_model_pin_and_dimensions(self) -> None:
        self.assertEqual(8_192, contract.MAX_TOKENS)
        self.assertEqual((128, 256, 512, 768), contract.TRAINED_DIMENSIONS)
        self.assertEqual(1_488_915_288, contract.WEIGHT_BYTES)
        self.assertEqual(64, len(contract.WEIGHT_SHA256))


if __name__ == "__main__":
    unittest.main()
