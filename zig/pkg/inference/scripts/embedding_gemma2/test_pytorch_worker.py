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
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytorch_worker
import select_pytorch_baseline


class PytorchWorkerContractTests(unittest.TestCase):
    def test_supported_configuration_matrix_is_explicit(self) -> None:
        self.assertEqual(("eager", "sdpa", "flex_attention"), pytorch_worker.ATTENTION_MODES)
        self.assertEqual(("none", "default", "max-autotune"), pytorch_worker.COMPILE_MODES)

    def test_selector_percentiles_are_deterministic(self) -> None:
        values = [0.5, 0.1, 0.4, 0.2, 0.3]
        self.assertEqual(0.3, select_pytorch_baseline.percentile(values, 0.5))
        self.assertEqual(0.4, select_pytorch_baseline.percentile(values, 0.95))

    def test_selector_worker_readiness_is_bounded(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "print('ready', flush=True)"],
            text=True,
            stdout=subprocess.PIPE,
        )
        try:
            self.assertEqual("ready", select_pytorch_baseline.readiness_line(process, 2).strip())
        finally:
            process.wait(timeout=2)
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
