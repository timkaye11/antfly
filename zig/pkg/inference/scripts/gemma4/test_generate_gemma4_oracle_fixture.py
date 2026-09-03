from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from generate_gemma4_oracle_fixture import DEFAULT_SPEC, generate_rows, main


class Gemma4OracleFixtureTest(unittest.TestCase):
    def test_generation_is_deterministic_group_disjoint_and_covers_contract_cases(self) -> None:
        counts = {"train": 16, "eval": 8, "test": 8}
        first_data, first_manifest = generate_rows(DEFAULT_SPEC, 20260810, counts)
        second_data, second_manifest = generate_rows(DEFAULT_SPEC, 20260810, counts)
        self.assertEqual(first_data, second_data)
        self.assertEqual(first_manifest, second_manifest)

        rows = [json.loads(line) for line in first_data.decode("utf-8").splitlines()]
        self.assertEqual(sum(counts.values()), len(rows))
        names = {row["id"].rsplit("-", 1)[-1] for row in rows}
        self.assertEqual(
            {
                "plain",
                "system",
                "multiturn",
                "tool_call",
                "parallel_tool_calls",
                "unicode",
                "escaped_json",
                "boundary_payload",
            },
            names,
        )
        groups = {
            split: {row["metadata"]["group_id"] for row in rows if row["split"] == split}
            for split in counts
        }
        self.assertFalse(groups["train"] & groups["eval"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["eval"] & groups["test"])
        self.assertTrue(any(len(message["content"]) > 500 for row in rows for message in row["messages"]))
        self.assertTrue(any(len(message.get("tool_calls", [])) == 2 for row in rows for message in row["messages"]))
        self.assertTrue(any("👩🏽‍💻" in message["content"] for row in rows for message in row["messages"]))

    def test_cli_check_detects_drift_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fixture.jsonl"
            manifest = Path(tmp) / "fixture.manifest.json"
            common = [
                "--output", str(output),
                "--manifest", str(manifest),
                "--train-count", "8",
                "--eval-count", "8",
                "--test-count", "8",
            ]
            captured = io.StringIO()
            with redirect_stdout(captured), redirect_stderr(captured):
                self.assertEqual(0, main(common))
                self.assertEqual(0, main([*common, "--check"]))
                output.write_bytes(output.read_bytes() + b"\n")
                self.assertEqual(2, main([*common, "--check"]))
            self.assertIn("generated fixture drifted", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
