# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from materialize_official_retrieval_sources import qrel, verify_source, write_unique_rows


class OfficialSourceTests(unittest.TestCase):
    def test_pinned_git_and_lfs_bytes_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            data = b"query-id\tcorpus-id\tscore\nq\td\t1\n"
            path.write_bytes(data)
            git = SimpleNamespace(lfs=None, blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(), rfilename="test.tsv")
            lfs = SimpleNamespace(lfs={"sha256": hashlib.sha256(data).hexdigest()}, rfilename="corpus.parquet")
            self.assertEqual(verify_source(path, git), verify_source(path, lfs))
            path.write_bytes(data + b"extra")
            for metadata in (git, lfs):
                with self.assertRaisesRegex(ValueError, "differs from pinned"):
                    verify_source(path, metadata)

    def test_disk_inventory_detects_duplicate_and_missing_positive_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(write_unique_rows(root / "good.jsonl", [{"id": "a"}, {"id": "b"}], positive_documents={"b"}), 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                write_unique_rows(root / "duplicate.jsonl", [{"id": "a"}, {"id": "a"}])
            with self.assertRaisesRegex(ValueError, "absent"):
                write_unique_rows(root / "missing.jsonl", [{"id": "a"}], positive_documents={"b"})
            self.assertFalse(list(root.glob("*.sqlite")))

    def test_qrels_preserve_ids_and_reject_nonfinite_or_negative_grades(self):
        self.assertEqual(qrel({"query-id": 0, "corpus-id": "001", "score": "2"}), {"query_id": "0", "corpus_id": "001", "score": 2.0})
        for score in (float("nan"), float("inf"), -1, True):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                qrel({"query-id": "q", "corpus-id": "d", "score": score})


if __name__ == "__main__":
    unittest.main()
