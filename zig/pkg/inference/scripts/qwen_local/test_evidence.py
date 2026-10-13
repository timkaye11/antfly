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
import hashlib
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
import machine_profile
from chat import command, process_budget
from evidence import (
    clean_swap_delta,
    parse_swap_snapshot,
    validate_receipt,
    validate_repeat,
)
from qualify_lifecycle import write_with_deadline
from verify_install import check_files


def receipt(family="27b", derived=False):
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
    if derived:
        value.update(process_cap_bytes=4096 * 1024**2, process_cap_source="derived")
    return value


def flash_cache(value, source="explicit", slots=11, stride=3 * 1024**2):
    value.update(
        admitted_bytes=3 * 1024**3,
        process_memory={"available": True, "peak_footprint_bytes": 2 * 1024**3},
        expert_slot_counts=[slots] * 48,
        expert_strides=[stride] * 48,
        expert_cache_bytes=slots * stride * 48,
        expert_cache_source=source,
    )
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

    def test_context_limit_mirrors_the_runtime_machine_profile(self):
        inference = Path(__file__).resolve().parents[2]
        source = (inference / "src/runtime/machine_profile.zig").read_text()
        match = re.search(r"pub const max_context: usize = (\d+);", source)
        self.assertIsNotNone(match)
        self.assertEqual(int(match[1]), machine_profile.MAX_CONTEXT)
        # The harness carries no cap arithmetic of its own to drift from it.
        for name in ("chat.py", "benchmark.py", "evidence.py", "machine_profile.py"):
            text = (Path(__file__).parent / name).read_text()
            for literal in ("10240", "9216"):
                self.assertNotIn(literal, text, f"{name} hard-codes a process cap")

    def test_explicit_budgets_pass_through_unclamped_and_unset_ones_are_derived(self):
        physical = 48 * 1024**3
        self.assertIsNone(machine_profile.explicit_mb("cap", None, physical))
        self.assertEqual(machine_profile.explicit_mb("cap", 36864, physical), 36864)
        for bad in (0, -1, 49153, 1.5, True):
            with self.assertRaises(ValueError):
                machine_profile.explicit_mb("cap", bad, physical)
        self.assertEqual(machine_profile.budget_arguments("flash"), [])
        self.assertEqual(
            machine_profile.budget_arguments("flash", 30720, 36864, "stamp"),
            [
                "--combined-budget-mb",
                "36864",
                "--moe-cache-mb",
                "30720",
                "--verify",
                "stamp",
            ],
        )
        with self.assertRaises(ValueError):
            machine_profile.budget_arguments("27b", cache_mb=1536)
        with self.assertRaises(ValueError):
            machine_profile.budget_arguments("27b", verify="never")
        self.assertIsNone(process_budget("27b"))
        derived = command("runner", "model.gguf", "flash")
        for flag in ("--combined-budget-mb", "--moe-cache-mb", "--verify"):
            self.assertNotIn(flag, derived)
        explicit = command(
            "runner", "model.gguf", "flash", cache_mb=1536, combined_mb=1, verify="full"
        )
        self.assertEqual(explicit[explicit.index("--moe-cache-mb") + 1], "1536")
        self.assertEqual(explicit[explicit.index("--verify") + 1], "full")
        self.assertNotIn("--moe-cache-mb", command("runner", "m", "27b", cache_mb=1536))

    def test_receipts_report_the_enforced_cap_and_never_a_substituted_one(self):
        for family in ("27b", "flash"):
            derived = receipt(family, derived=True)
            result = validate_receipt(derived, family, None, 4)
            self.assertEqual(result["process_cap_bytes"], 4096 * 1024**2)
            # Without an explicit request the runner must say what it derived.
            with self.assertRaises(ValueError):
                validate_receipt(receipt(family), family, None, 4)
            mislabeled = receipt(family, derived=True)
            mislabeled["process_cap_source"] = "explicit"
            with self.assertRaises(ValueError):
                validate_receipt(mislabeled, family, None, 4)
            # An explicit request must come back unchanged: no silent clamp.
            clamped = receipt(family)
            clamped.update(
                process_cap_bytes=2048 * 1024**2, process_cap_source="explicit"
            )
            with self.assertRaises(ValueError):
                validate_receipt(clamped, family, 4096, 4)
            with self.assertRaises(ValueError):
                validate_receipt(derived, family, 4096, 4)
            honoured = receipt(family)
            honoured.update(
                process_cap_bytes=4096 * 1024**2, process_cap_source="explicit"
            )
            validate_receipt(honoured, family, 4096, 4)
        legacy = receipt("27b")
        legacy["cap_bytes"] = 4096 * 1024**2
        validate_receipt(legacy, "27b", 4096, 4)
        legacy["cap_bytes"] = 2048 * 1024**2
        with self.assertRaises(ValueError):
            validate_receipt(legacy, "27b", 4096, 4)

    def test_flash_cache_is_reported_as_allocated_not_as_requested(self):
        budget = 4096
        explicit = flash_cache(receipt("flash"))
        result = validate_receipt(explicit, "flash", budget, 4, cache_mb=1600)
        self.assertEqual(result["expert_cache_bytes"], 11 * 3 * 1024**2 * 48)
        # More than requested, a stale byte count, or a truncated inventory.
        with self.assertRaises(ValueError):
            validate_receipt(explicit, "flash", budget, 4, cache_mb=1024)
        for mutation in (
            {"expert_cache_bytes": 1536 * 1024**2},
            {"expert_slot_counts": [11] * 47},
            {"expert_slot_counts": [9] * 48},
            {"expert_cache_source": "derived"},
            {"expert_cache_bytes": None},
        ):
            with self.subTest(mutation=mutation):
                data = flash_cache(receipt("flash"))
                data.update(mutation)
                with self.assertRaises(ValueError):
                    validate_receipt(data, "flash", budget, 4, cache_mb=1600)
        # More than 255 slots per layer is ordinary on a large host.
        large = flash_cache(receipt("flash"), source="derived", slots=199, stride=1024)
        validate_receipt(large, "flash", budget, 4)
        with self.assertRaises(ValueError):
            validate_receipt(flash_cache(receipt("flash")), "flash", budget, 4)
        # A requested cache needs the inventory even from an older runner.
        with self.assertRaises(ValueError):
            validate_receipt(receipt("flash"), "flash", budget, 4, cache_mb=1600)

    def test_stamp_verification_is_accepted_only_where_the_caller_allows_it(self):
        hashed = receipt()
        hashed["verification_mode"] = "full"
        stamped = receipt()
        stamped.update(
            hash_verified=False, stamp_verified=True, verification_mode="stamp"
        )
        fallback = receipt()
        fallback.update(stamp_verified=False, verification_mode="stamp")
        for data in (hashed, stamped, fallback):
            result = validate_receipt(data, "27b", 4096, 4, verify="stamp")
            self.assertEqual(
                result["verification"], "stamp" if data is stamped else "hash"
            )
        validate_receipt(hashed, "27b", 4096, 4, verify="full")
        # Qualification never accepts a stamp, nor a pass run in stamp mode.
        for data in (stamped, fallback):
            with self.assertRaises(ValueError):
                validate_receipt(data, "27b", 4096, 4, verify="full")
        for mutation in (
            {"hash_verified": False, "stamp_verified": False},
            {"hash_verified": True, "stamp_verified": True},
            {
                "hash_verified": False,
                "stamp_verified": True,
                "verification_mode": "full",
            },
        ):
            with self.subTest(mutation=mutation):
                data = receipt()
                data.update(mutation)
                with self.assertRaises(ValueError):
                    validate_receipt(data, "27b", 4096, 4, verify="stamp")
        with self.assertRaises(ValueError):
            validate_receipt(receipt(), "27b", 4096, 4, verify="never")

    def test_context_is_a_parameter_of_the_evidence_not_a_literal(self):
        data = receipt()
        data["context_size"] = 2048
        validate_receipt(data, "27b", 4096, 4, context=2048)
        with self.assertRaises(ValueError):
            validate_receipt(data, "27b", 4096, 4)
        data["prompt_token_ids"] = [1] * 2045
        data["position"] = 2048
        with self.assertRaises(ValueError):
            validate_receipt(data, "27b", 4096, 4, context=2048)

    def test_install_check_binds_layout_size_and_digest_to_the_manifest(self):
        def digest(data):
            return hashlib.sha256(data).hexdigest()

        files = {
            "UD-Q4_K_XL/model-00001-of-00002.gguf": b"index",
            "UD-Q4_K_XL/model-00002-of-00002.gguf": b"weights",
            "tokenizer.json": b"{}",
            "README.md": b"notes",
        }
        manifest = {
            "files": [
                {"path": path, "bytes": len(data), "sha256": digest(data)}
                for path, data in files.items()
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "UD-Q4_K_XL").mkdir()
            for path, data in files.items():
                (root / path).write_bytes(data)
            self.assertTrue(all(row["ok"] for row in check_files(root, manifest)))
            # Provenance files may be absent; anything the runner reads may not.
            (root / "README.md").unlink()
            self.assertTrue(all(row["ok"] for row in check_files(root, manifest)))
            (root / "tokenizer.json").unlink()
            rows = {row["path"]: row for row in check_files(root, manifest)}
            self.assertEqual(rows["tokenizer.json"].get("error"), "missing")
            (root / "tokenizer.json").write_bytes(b"{}")
            # A shard copied flat, beside the sidecars, is not the pinned layout.
            shard = root / "UD-Q4_K_XL/model-00002-of-00002.gguf"
            shard.rename(root / shard.name)
            rows = {row["path"]: row for row in check_files(root, manifest)}
            self.assertFalse(rows["UD-Q4_K_XL/model-00002-of-00002.gguf"]["ok"])
            (root / shard.name).rename(shard)
            shard.write_bytes(b"weightz")
            rows = {row["path"]: row for row in check_files(root, manifest)}
            self.assertEqual(
                rows[str(shard.relative_to(root))]["error"], "sha256 mismatch"
            )
            # Sizes alone cannot see a same-size change; hashing is the check.
            self.assertTrue(
                all(row["ok"] for row in check_files(root, manifest, hash_files=False))
            )
            shard.write_bytes(b"short")
            rows = {row["path"]: row for row in check_files(root, manifest, False)}
            self.assertEqual(
                rows[str(shard.relative_to(root))]["error"], "size mismatch"
            )

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
                data = receipt(derived=True)
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
                patch.object(
                    benchmark, "physical_memory_bytes", return_value=16 * 1024**3
                ),
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
        # Nothing is requested unless the operator asks: the runner derives it.
        self.assertEqual(
            result["requested"],
            {
                "combined_budget_mb": None,
                "cache_mb": None,
                "context": machine_profile.MAX_CONTEXT,
                "verify": "stamp",
            },
        )
        for row in result["cases"]:
            self.assertNotIn("--combined-budget-mb", row["command"])
            self.assertEqual(
                row["command"][row["command"].index("--verify") + 1], "stamp"
            )
            self.assertEqual(row["metrics"]["process_cap_bytes"], 4096 * 1024**2)


if __name__ == "__main__":
    unittest.main()
