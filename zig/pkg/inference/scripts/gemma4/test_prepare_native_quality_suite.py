# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import prepare_native_quality_suite as prepare


class ModelSelectionTests(unittest.TestCase):
    def test_default_e4b_suite_is_unchanged(self):
        self.assertEqual(prepare.build_cases(100), prepare.build_cases(100, prepare.MODEL))

    def test_both_request_arms_use_the_selected_model(self):
        for model in prepare.MODELS:
            cases = prepare.build_cases(100, model)
            self.assertEqual(64, len(cases))
            self.assertEqual(set(prepare.CATEGORIES), {c['category'] for c in cases})
            for case in cases:
                self.assertEqual(model, case['body']['model'])
                self.assertEqual(model, case['checkpoint_body']['model'])

    def test_cli_materializes_e2b_bounded_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'suite.jsonl'
            subprocess.run([sys.executable, str(Path(prepare.__file__)), '--model', prepare.MODELS[0],
                            '--lines', '100', '--quality-profile', 'bounded', '--output', str(output)],
                           check=True, capture_output=True)
            cases = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(16, len(cases))
            self.assertTrue(all(c['body']['model'] == c['checkpoint_body']['model'] == prepare.MODELS[0]
                                for c in cases))
            manifest = json.loads(output.with_suffix('.manifest.json').read_text())
            self.assertEqual(16, manifest['cases'])

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare.build_cases(100, 'unqualified/model')


if __name__ == '__main__':
    unittest.main()
