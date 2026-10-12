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

"""Exercise invalid evidence and the benchmark's failure reporting path."""

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark
from evidence import (
    clean_swap_delta,
    parse_swap_snapshot,
    validate_receipt,
    validate_repeat,
)
from qualify_lifecycle import write_with_deadline


def receipt(family="27b"):
    value = {
        "architecture": "qwen35" if family == "27b" else "qwen4exp",
        "backend": "metal",
        "finish_reason": "length",
        "context_size": 4096,
        "prompt_token_ids": [1, 2, 3],
        "generated_token_ids": [4, 5, 6, 7],
        "generated_tokens": 4,
        "decode_steps": 3,
        "decode_token_ms": [10, 10, 10],
        "load_ms": 10,
        "prefill_ms": 20,
        "decode_ms": 31,
        "first_token_ready_ms": 30,
        "admitted_bytes": 1024,
        "process_memory": {"available": True, "peak_footprint_bytes": 2048},
        "hash_verified": True,
        "prompt_lookup": 0,
        "position": 6,
        "prompt_tokens": 3,
        "sidecars_hash_verified": True,
    }
    return value


class EvidenceTests(unittest.TestCase):
    def test_lifecycle_write_deadline_handles_a_stalled_pipe(self):
        read_fd, write_fd = os.pipe()
        with (
            os.fdopen(read_fd, "rb", buffering=0) as reader,
            os.fdopen(write_fd, "wb", buffering=0) as writer,
        ):
            process = SimpleNamespace(stdin=writer)
            write_with_deadline(process, b"hello", time.monotonic() + 1)
            self.assertEqual(reader.read(5), b"hello")
            with self.assertRaisesRegex(TimeoutError, "session request write"):
                write_with_deadline(
                    process, b"x" * (1024 * 1024), time.monotonic() + 0.05
                )
            self.assertTrue(os.get_blocking(write_fd))
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        with os.fdopen(write_fd, "wb", buffering=0) as writer:
            write_with_deadline(
                SimpleNamespace(stdin=writer), b"closed", time.monotonic() + 1
            )
            self.assertTrue(os.get_blocking(write_fd))

    def test_flash_runtime_identity_matches_ordered_artifact_manifest(self):
        inference = Path(__file__).resolve().parents[2]
        manifest = json.loads(
            (inference / "scripts/qwen4exp/artifact.json").read_text()
        )
        source = (inference / "src/models/qwen4exp.zig").read_text()
        pins = [
            (int(size), digest)
            for size, digest in re.findall(
                r'\.bytes = (\d+), \.sha256 = "([a-f0-9]{64})"', source
            )
        ]
        expected = [
            (item["bytes"], item["sha256"])
            for item in manifest["files"]
            if item["path"].endswith(".gguf")
        ]
        self.assertEqual(len(expected), 4)
        self.assertEqual(pins, expected)
        self.assertIn(f'artifact_revision = "{manifest["model_revision"]}"', source)

    def test_valid_receipts_and_eos_accounting(self):
        for family in ("27b", "flash"):
            data = receipt(family)
            result = validate_receipt(data, family, 4096, 4)
            self.assertAlmostEqual(
                result["effective_decode_tokens_per_second"], 3000 / 31
            )
            self.assertEqual(result["artifact_hash_verified_by_runtime"], True)
            data.update(
                finish_reason="stop", generated_token_ids=[4, 5, 6], generated_tokens=3
            )
            validate_receipt(data, family, 4096, 4)

    def test_malformed_or_incomplete_evidence_is_rejected(self):
        mutations = [
            {"architecture": "qwen4exp"},
            {"backend": "cpu"},
            {"finish_reason": "Cancelled"},
            {"context_size": 4097},
            {"hash_verified": False},
            {"prompt_lookup": 1},
            {"generated_token_ids": [4, 5]},
            {"generated_token_ids": [4, 5, 6, 248320]},
            {"generated_token_ids": [4, 5, 6, True]},
            {"prompt_token_ids": []},
            {"generated_tokens": True},
            {"decode_steps": 4},
            {"decode_token_ms": [10]},
            {"decode_ms": float("nan")},
            {"prefill_ms": float("inf")},
            {"load_ms": -1},
            {"first_token_ready_ms": 60},
            {"decode_ms": 10},
            {"failure_stage": "decode"},
            {"process_memory": None},
            {"process_memory": {"available": False}},
            {
                "process_memory": {
                    "available": True,
                    "peak_footprint_bytes": 5 * 1024**3,
                }
            },
            {"admitted_bytes": 5 * 1024**3},
            {"position": 5},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                data = receipt()
                data.update(mutation)
                with self.assertRaises(ValueError):
                    validate_receipt(data, "27b", 4096, 4)
        for missing in receipt():
            if missing in ("prompt_tokens", "sidecars_hash_verified"):
                continue
            with self.subTest(missing=missing):
                data = receipt()
                del data[missing]
                with self.assertRaises(ValueError):
                    validate_receipt(data, "27b", 4096, 4)

    def test_swapins_alone_and_counter_resets_reject_timing(self):
        before = parse_swap_snapshot("Swapins: 10.\nSwapouts: 20.\n")
        self.assertEqual(
            clean_swap_delta(before, before), {"swapins": 0, "swapouts": 0}
        )
        for after in (
            {"swapins": 11, "swapouts": 20},
            {"swapins": 10, "swapouts": 21},
            {"swapins": 0, "swapouts": 0},
        ):
            with self.assertRaises(ValueError):
                clean_swap_delta(before, after)
        with self.assertRaises(ValueError):
            parse_swap_snapshot("Swapouts: 20.\n")

    def test_repeated_greedy_output_must_match(self):
        first = receipt()
        validate_repeat(first, copy.deepcopy(first))
        for key, value in (
            ("prompt_token_ids", [2, 3]),
            ("generated_token_ids", [9, 5, 6, 7]),
            ("finish_reason", "stop"),
        ):
            second = copy.deepcopy(first)
            second[key] = value
            with self.assertRaises(ValueError):
                validate_repeat(first, second)

    def run_harness(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "runner"
            binary.write_bytes(b"fixture executable identity")
            output = root / "results"
            arguments = [
                "benchmark.py",
                "--binary",
                str(binary),
                "--model",
                "fixture.gguf",
                "--family",
                "27b",
                "--output",
                str(output),
            ]
            calls = 0

            def execute(command, **kwargs):
                nonlocal calls
                calls += 1
                if mode == "timeout":
                    raise subprocess.TimeoutExpired(command, 2400)
                data = receipt()
                if mode == "nonfinite":
                    data["decode_ms"] = float("nan")
                if mode == "mismatch" and calls == 4:
                    data["generated_token_ids"][0] = 9
                path = Path(command[command.index("--json-timing") + 1])
                if mode != "missing":
                    path.write_text(
                        "not JSON" if mode == "malformed" else json.dumps(data)
                    )
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.object(sys, "argv", arguments),
                patch.object(benchmark.platform, "platform", return_value="fixture"),
                patch.object(benchmark.subprocess, "run", side_effect=execute),
                patch.object(
                    benchmark,
                    "swap_snapshot",
                    return_value={"swapins": 0, "swapouts": 0},
                ),
            ):
                if mode == "success":
                    benchmark.main()
                else:
                    with self.assertRaises(
                        (ValueError, OSError, subprocess.TimeoutExpired)
                    ):
                        benchmark.main()
            return json.loads((output / "summary.json").read_text())

    def test_harness_requires_receipt_and_preserves_failure_record(self):
        for mode in ("missing", "malformed", "nonfinite", "timeout", "mismatch"):
            with self.subTest(mode=mode):
                result = self.run_harness(mode)
                self.assertFalse(result["benchmark_checks_passed"])
                self.assertFalse(result["cases"][-1]["checks_passed"])
                self.assertIn("error", result["cases"][-1])

    def test_harness_only_passes_after_all_cases_and_repeats(self):
        result = self.run_harness("success")
        self.assertTrue(result["benchmark_checks_passed"])
        self.assertEqual(len(result["cases"]), 6)
        self.assertTrue(all(row["checks_passed"] for row in result["cases"]))


if __name__ == "__main__":
    unittest.main()
