# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from prepare_finetune_performance import train_source_rows


class PerformanceSourceTest(unittest.TestCase):
    def test_unicode_separators_inside_json_strings_do_not_split_records(self):
        records = [
            {"query": "first\u2028second\u2029third", "split": "train"},
            {"query": "another training query"},
            {"query": "held out", "split": "test"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n")
            self.assertEqual(train_source_rows(source), records[:2])


if __name__ == "__main__":
    unittest.main()
