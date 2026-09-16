from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gemma4_mlx_source as source


class SourceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        self.root.mkdir()
        self.revision = "a" * 40
        self.archive = self.base / "source.tar.gz"
        self.payload = b"VALUE = 42\n"
        (self.root / "model.py").write_bytes(self.payload)
        self.write_archive()
        self.pin = patch.dict(
            source.MLX_LM_ARCHIVE_SHA256,
            {self.revision: hashlib.sha256(self.archive.read_bytes()).hexdigest()},
            clear=True,
        )
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def write_archive(self, name="model.py"):
        prefix = "mlx-lm-" + self.revision
        with tarfile.open(self.archive, "w:gz") as out:
            entry = tarfile.TarInfo(prefix)
            entry.type = tarfile.DIRTYPE
            out.addfile(entry)
            entry = tarfile.TarInfo(prefix + "/" + name)
            entry.size = len(self.payload)
            out.addfile(entry, io.BytesIO(self.payload))

    def attest(self):
        return source.attest_mlx_lm_archive(self.root, self.archive, self.revision)

    def test_exact_source_is_bound_to_pinned_archive(self):
        result = self.attest()
        self.assertEqual(1, result["source_files"])
        self.assertEqual(self.revision, result["revision"])
        self.assertEqual(
            hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            result["archive_sha256"],
        )

    def test_changed_source_and_generated_bytecode_are_rejected(self):
        (self.root / "model.py").write_bytes(b"VALUE = 0\n")
        with self.assertRaisesRegex(source.SourceArchiveError, "file changed"):
            self.attest()
        (self.root / "model.py").write_bytes(self.payload)
        cache = self.root / "__pycache__"
        cache.mkdir()
        (cache / "model.pyc").write_bytes(b"stale")
        with self.assertRaisesRegex(source.SourceArchiveError, "inventory differs"):
            self.attest()

    def test_unpinned_revision_or_changed_archive_is_rejected(self):
        with self.assertRaisesRegex(source.SourceArchiveError, "no pinned"):
            source.attest_mlx_lm_archive(self.root, self.archive, "b" * 40)
        self.archive.write_bytes(self.archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(source.SourceArchiveError, "SHA-256 differs"):
            self.attest()

    def test_source_symlink_is_rejected(self):
        original = self.root / "model.py"
        original.rename(self.base / "elsewhere.py")
        original.symlink_to(self.base / "elsewhere.py")
        with self.assertRaisesRegex(source.SourceArchiveError, "symlink"):
            self.attest()

    def test_traversal_is_rejected_even_with_matching_archive_digest(self):
        self.write_archive("../escape.py")
        source.MLX_LM_ARCHIVE_SHA256[self.revision] = hashlib.sha256(
            self.archive.read_bytes()
        ).hexdigest()
        with self.assertRaisesRegex(source.SourceArchiveError, "escapes"):
            self.attest()


if __name__ == "__main__":
    unittest.main()
