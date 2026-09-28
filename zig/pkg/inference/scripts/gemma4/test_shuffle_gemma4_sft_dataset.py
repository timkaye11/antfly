from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import shuffle_gemma4_sft_dataset as shuffle


def records(count: int = 300) -> list[bytes]:
    return [
        json.dumps(
            {
                "schema": "gemma_chat/v1",
                "id": f"source-{i}",
                "messages": [
                    {"role": "assistant", "content": "yes" if i < count // 2 else "no"}
                ],
            },
            separators=(",", ":"),
        ).encode()
        for i in range(count)
    ]


class SftDatasetOrderTests(unittest.TestCase):
    def test_seeded_order_preserves_records_and_breaks_class_cluster(self) -> None:
        original = records()
        output, ids = shuffle.ordered_records(b"\n".join(original), 42)
        self.assertEqual(Counter(original), Counter(output.splitlines()))
        self.assertEqual(300, len(set(ids)))
        self.assertEqual(
            {"yes", "no"},
            {
                json.loads(row)["messages"][0]["content"]
                for row in output.splitlines()[-32:]
            },
        )

    def test_input_order_and_labels_do_not_control_permutation(self) -> None:
        original = records()
        first, ids = shuffle.ordered_records(b"\n".join(original), 42)
        self.assertEqual(
            (first, ids), shuffle.ordered_records(b"\n".join(reversed(original)), 42)
        )
        altered = [
            row.replace(b'"yes"', b'"unknown"').replace(b'"no"', b'"unknown"')
            for row in original
        ]
        self.assertEqual(ids, shuffle.ordered_records(b"\n".join(altered), 42)[1])
        self.assertNotEqual(ids, shuffle.ordered_records(b"\n".join(original), 17)[1])

    def test_record_bytes_and_unicode_separators_are_preserved(self) -> None:
        row = (
            b' {"schema": "gemma_chat/v1", "id": "a", "text": "'
            + "\u2028".encode()
            + b'"}\r'
        )
        self.assertEqual(row + b"\n", shuffle.ordered_records(row + b"\n", 0)[0])

    def test_rejects_ambiguous_identity_and_invalid_input(self) -> None:
        good = records(1)[0]
        for payload in (
            b"",
            b"\n",
            good + b"\n\n",
            good + b"\n" + good,
            b"[]",
            b'{"schema":"gemma_chat/v1"}',
            b'{"schema":"gemma_chat/v1","id":1}',
            b'{"schema":"gemma_chat/v1","id":" "}',
            b'{"schema":"other","id":"a"}',
            b"\xff",
        ):
            with (
                self.subTest(payload=payload),
                self.assertRaises(shuffle.DatasetOrderError),
            ):
                shuffle.ordered_records(payload, 42)
        for seed in (-1, 2**64, True):
            with self.subTest(seed=seed), self.assertRaises(shuffle.DatasetOrderError):
                shuffle.ordered_records(good, seed)

    def test_publication_binds_hashes_and_refuses_existing_or_alias_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_bytes(b"\n".join(records(8)))
            output = root / "ordered"
            manifest = shuffle.materialize(source, output, 42)
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                manifest["source_sha256"],
            )
            snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
            self.assertEqual(
                hashlib.sha256(snapshot["train.jsonl"]).hexdigest(),
                manifest["dataset_sha256"],
            )
            alias = root / "alias"
            alias.symlink_to(output, target_is_directory=True)
            for destination in (output, alias, root):
                with self.assertRaises(FileExistsError):
                    shuffle.materialize(source, destination, 17)
            self.assertEqual(
                snapshot, {p.name: p.read_bytes() for p in output.iterdir()}
            )

    def test_invalid_input_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text('{"id":"missing-schema"}\n')
            with self.assertRaises(shuffle.DatasetOrderError):
                shuffle.materialize(source, root / "output", 42)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
