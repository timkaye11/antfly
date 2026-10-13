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
import io
import subprocess
import sys
import unittest
from unittest import mock
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytorch_worker
import select_pytorch_baseline


class PytorchWorkerContractTests(unittest.TestCase):
    def test_cuda_oom_batch_split_preserves_order_and_reuses_limit(self) -> None:
        class Oom(RuntimeError):
            pass

        attempted, released, caps = [], [], {}

        def run(items):
            attempted.append(list(items))
            if len(items) > 2:
                raise Oom('device memory')
            return [[item] for item in items]

        items = list(range(7))
        result = pytorch_worker.resource_bounded_batch(items, run, Oom,
                                                       lambda: released.append(True), caps, 'long')
        self.assertEqual([[item] for item in items], result)
        self.assertEqual(1, caps['long'])
        self.assertEqual(2, len(released))
        attempted.clear()
        self.assertEqual(result, pytorch_worker.resource_bounded_batch(items, run, Oom,
                                                                       lambda: released.append(True), caps, 'long'))
        self.assertTrue(all(len(batch) == 1 for batch in attempted))
        self.assertEqual(2, len(released))

    def test_cuda_oom_single_row_and_unrelated_errors_propagate(self) -> None:
        class Oom(RuntimeError):
            pass

        released, caps = [], {}

        def oom(_items):
            raise Oom('single row')

        with self.assertRaisesRegex(Oom, 'single row'):
            pytorch_worker.resource_bounded_batch([0], oom, Oom,
                                                  lambda: released.append(True), caps, 'long')
        self.assertEqual({}, caps)
        self.assertEqual([], released)

        def invalid(_items):
            raise ValueError('out of memory is not a typed CUDA error')

        with self.assertRaises(ValueError):
            pytorch_worker.resource_bounded_batch([0, 1], invalid, Oom,
                                                  lambda: released.append(True), caps, 'long')
        self.assertEqual({}, caps)

    def test_compiler_wrapped_cuda_oom_is_recognized(self) -> None:
        class Oom(RuntimeError):
            pass

        def run(items):
            if len(items) > 1:
                try:
                    raise Oom('device')
                except Oom as error:
                    raise RuntimeError('compiled forward') from error
            return items

        self.assertEqual([1, 2], pytorch_worker.resource_bounded_batch(
            [1, 2], run, Oom, lambda: None, {}, 'wrapped'))

    def test_reference_oom_is_conclusive_but_timeout_is_not(self) -> None:
        self.assertEqual(('failed_hardware', True),
                         select_pytorch_baseline.failure_status('CUDA out of memory'))
        self.assertEqual(('inconclusive_timeout', None),
                         select_pytorch_baseline.failure_status('CUDA out of memory', timed_out=True))
        self.assertEqual(('failed_runtime', True),
                         select_pytorch_baseline.failure_status('HTTP 400: invalid input'))

    def test_reference_http_failure_retains_typed_error_body(self) -> None:
        error = urllib.error.HTTPError('http://localhost', 400, 'Bad Request', {},
                                       io.BytesIO(b'{"error_type":"OutOfMemoryError","error":"device memory"}'))
        with mock.patch.object(select_pytorch_baseline.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 400: .*OutOfMemoryError'):
                select_pytorch_baseline.post('http://localhost', {}, 1)

    def test_single_row_wrapped_oom_keeps_hardware_error_class(self) -> None:
        class Oom(RuntimeError):
            pass

        wrapped = RuntimeError('compiled forward failed')
        wrapped.__cause__ = Oom('device memory')
        self.assertEqual({'error': 'compiled forward failed', 'error_type': 'OutOfMemoryError'},
                         pytorch_worker.error_payload(wrapped, Oom))
        self.assertEqual('ValueError', pytorch_worker.error_payload(ValueError('invalid input'), Oom)['error_type'])

    def test_group_titles_are_rendered_once_and_only_for_documents(self) -> None:
        group = {'title': 'T', 'content': [{'type': 'text', 'text': 'body'}, {'type': 'text', 'text': ' tail'}]}
        text, images, audio = pytorch_worker.Worker._prepare_group(group, 'RETRIEVAL_DOCUMENT')
        self.assertEqual(text, 'title: T | text: body tail')
        self.assertEqual((images, audio), ([], []))
        with self.assertRaisesRegex(ValueError, 'invalid document title'):
            pytorch_worker.Worker._prepare_group(group, 'CODE_RETRIEVAL')
        group['title'] = ''
        self.assertEqual(pytorch_worker.Worker._prepare_group(group, 'RETRIEVAL_DOCUMENT')[0], 'title:  | text: body tail')

    def test_supported_configuration_matrix_is_explicit(self) -> None:
        self.assertEqual(("eager", "sdpa", "flex_attention"), pytorch_worker.ATTENTION_MODES)
        self.assertEqual(("none", "default", "max-autotune"), pytorch_worker.COMPILE_MODES)

    def test_selector_percentiles_are_deterministic(self) -> None:
        values = [0.5, 0.1, 0.4, 0.2, 0.3]
        self.assertEqual(0.3, select_pytorch_baseline.percentile(values, 0.5))
        self.assertEqual(0.4, select_pytorch_baseline.percentile(values, 0.95))

    def test_timeout_cannot_promote_a_slower_completed_reference(self) -> None:
        rows = [{"attention": attention, "compile": mode, "status": "qualified"}
                for attention in select_pytorch_baseline.DEFAULT_ATTENTION_MODES for mode in pytorch_worker.COMPILE_MODES]
        self.assertTrue(select_pytorch_baseline.selection_complete(rows))
        rows[1]['status'] = 'inconclusive_timeout'
        self.assertFalse(select_pytorch_baseline.selection_complete(rows))
        rows[1]['status'] = 'unsupported'
        self.assertTrue(select_pytorch_baseline.selection_complete(rows))
        self.assertFalse(select_pytorch_baseline.selection_complete(rows[:-1]))

    def test_unknown_runtime_failures_leave_baseline_selection_inconclusive(self) -> None:
        rows = [{"attention": attention, "compile": mode, "status": "qualified"}
                for attention in select_pytorch_baseline.DEFAULT_ATTENTION_MODES for mode in pytorch_worker.COMPILE_MODES]
        for status in ("failed_runtime", "configuration_begin", None):
            with self.subTest(status=status):
                rows[1]['status'] = status
                self.assertFalse(select_pytorch_baseline.selection_complete(rows))
        for status in ("failed_hardware", "failed_parity", "unsupported"):
            with self.subTest(status=status):
                rows[1]['status'] = status
                self.assertTrue(select_pytorch_baseline.selection_complete(rows))

    def test_production_selection_requires_all_six_core_configurations(self) -> None:
        self.assertEqual(select_pytorch_baseline.DEFAULT_ATTENTION_MODES, ('eager', 'sdpa'))
        rows = [{'attention': attention, 'compile': mode, 'status': 'qualified'}
                for attention in select_pytorch_baseline.DEFAULT_ATTENTION_MODES
                for mode in pytorch_worker.COMPILE_MODES]
        self.assertTrue(select_pytorch_baseline.selection_complete(rows))
        self.assertFalse(select_pytorch_baseline.selection_complete(rows[:3]))
        self.assertFalse(select_pytorch_baseline.selection_complete(rows, include_flex=True))

    def test_optional_flex_diagnostics_require_explicit_full_matrix(self) -> None:
        rows = [{'attention': attention, 'compile': mode, 'status': 'qualified'}
                for attention in pytorch_worker.ATTENTION_MODES for mode in pytorch_worker.COMPILE_MODES]
        self.assertFalse(select_pytorch_baseline.selection_complete(rows))
        self.assertTrue(select_pytorch_baseline.selection_complete(rows, include_flex=True))
        rows[-1]['status'] = 'failed_runtime'
        self.assertFalse(select_pytorch_baseline.selection_complete(rows, include_flex=True))

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

    def test_partial_readiness_line_cannot_bypass_deadline(self) -> None:
        process = subprocess.Popen(
            [sys.executable, '-c', "import time; print('partial', end='', flush=True); time.sleep(5)"],
            text=True, stdout=subprocess.PIPE,
        )
        try:
            with self.assertRaises(TimeoutError):
                select_pytorch_baseline.readiness_line(process, .1)
        finally:
            process.terminate()
            process.wait(timeout=2)
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
