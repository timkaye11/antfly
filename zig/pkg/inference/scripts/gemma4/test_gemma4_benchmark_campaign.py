from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gemma4_oracle_contract import ContractError, prefixed_sha256
from run_gemma4_lora_benchmark_campaign import (
    MANIFEST_SCHEMA_VERSION,
    ORCHESTRATOR_RELATIVE_PATH,
    PLAN_SCHEMA_VERSION,
    SCRIPT_PATH as CAMPAIGN_ORCHESTRATOR_PATH,
    execute_campaign,
    load_plan,
    verify_complete_campaign_manifest,
)


SCRIPT_DIR = Path(__file__).resolve().parent


class CampaignFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.output = root / "campaign"
        self.plan_path = root / "plan.json"
        self.clock_values = iter((100, 110, 120, 130, 140, 150, 160, 170, 180, 190))
        self.tokens = iter(("c" * 32, "0" * 16, "1" * 16, "2" * 16, "3" * 16))
        self.launches: list[list[str]] = []
        self.plan_path.write_text(
            json.dumps(
                {
                    "schema_version": PLAN_SCHEMA_VERSION,
                    "repetitions": 2,
                    "cells": [
                        {
                            "cell_id": "e2b-peft-qv-s128-ga1",
                            "commands": {
                                "antfly-zig-metal": [
                                    sys.executable,
                                    str(SCRIPT_DIR / "run_antfly_gemma4_lora_benchmark.py"),
                                    "--fixture",
                                    "antfly",
                                ],
                                "mlx-lm": [
                                    sys.executable,
                                    str(SCRIPT_DIR / "run_gemma4_lora_mlx_benchmark.py"),
                                    "--fixture",
                                    "mlx",
                                ],
                            },
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def option(argv: list[str], name: str) -> str:
        return argv[argv.index(name) + 1]

    def run_process(self, raw_argv) -> int:
        argv = list(raw_argv)
        self.launches.append(argv)
        sequence_index = int(self.option(argv, "--sequence-index"))
        output = Path(self.option(argv, "--output"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "campaign_id": self.option(argv, "--campaign-id"),
                    "run_id": self.option(argv, "--run-id"),
                    "framework": (
                        "antfly-zig-metal"
                        if Path(argv[1]).name == "run_antfly_gemma4_lora_benchmark.py"
                        else "mlx-lm"
                    ),
                    "repetition": int(self.option(argv, "--repetition")),
                    "sequence_index": sequence_index,
                    "process": {"pid": 1000 + sequence_index, "started_unix_ns": 111 + sequence_index * 20},
                    "case": {"fixture": "e2b-peft-qv-s128-ga1"},
                    "implementation": {
                        "producer_source": {
                            "files": [
                                {
                                    "relative_path": ORCHESTRATOR_RELATIVE_PATH,
                                    "source_sha256": prefixed_sha256(CAMPAIGN_ORCHESTRATOR_PATH),
                                }
                            ]
                        }
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    def execute(self) -> Path:
        return execute_campaign(
            self.plan_path,
            self.output,
            run_process=self.run_process,
            clock=lambda: next(self.clock_values),
            token=lambda _size: next(self.tokens),
        )

    def samples(self) -> list[tuple[Path, dict]]:
        result = []
        for path in sorted((self.output / "samples").glob("*.json")):
            result.append((path, json.loads(path.read_text(encoding="utf-8"))))
        return result


class Gemma4BenchmarkCampaignTest(unittest.TestCase):
    def test_manifest_schema_tracks_the_closed_ledger_shape(self) -> None:
        schema = json.loads(
            (SCRIPT_DIR / "gemma4_benchmark_campaign.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(MANIFEST_SCHEMA_VERSION, schema["properties"]["schema_version"]["const"])
        self.assertEqual(
            {
                "schema_version", "status", "campaign_id", "created_unix_ns",
                "completed_unix_ns", "plan", "orchestrator", "run_count", "runs",
            },
            set(schema["required"]),
        )
        self.assertEqual(
            {
                "cell_id", "framework", "repetition", "sequence_index", "run_id",
                "argv", "argv_sha256", "started_unix_ns", "completed_unix_ns", "sample",
            },
            set(schema["properties"]["runs"]["items"]["required"]),
        )

    def test_orchestrator_publishes_closed_serial_alternating_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            manifest_path = fixture.execute()
            manifest = verify_complete_campaign_manifest(manifest_path, fixture.samples())

            self.assertEqual(MANIFEST_SCHEMA_VERSION, manifest["schema_version"])
            self.assertEqual("complete", manifest["status"])
            self.assertEqual(4, manifest["run_count"])
            self.assertEqual(
                ["antfly-zig-metal", "mlx-lm", "mlx-lm", "antfly-zig-metal"],
                [run["framework"] for run in manifest["runs"]],
            )
            self.assertTrue(
                all(
                    manifest["runs"][index]["completed_unix_ns"]
                    <= manifest["runs"][index + 1]["started_unix_ns"]
                    for index in range(3)
                )
            )
            with self.assertRaisesRegex(ContractError, "refusing to replace"):
                fixture.execute()

    def test_relabelled_sample_is_rejected_even_when_artifact_hash_is_updated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            manifest_path = fixture.execute()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sample_path = fixture.output / manifest["runs"][0]["sample"]["relative_path"]
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            sample["sequence_index"] = 3
            sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            data = sample_path.read_bytes()
            manifest["runs"][0]["sample"]["sha256"] = "sha256:" + hashlib.sha256(data).hexdigest()
            manifest["runs"][0]["sample"]["size_bytes"] = len(data)
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "sequence_index differs from launch evidence"):
                verify_complete_campaign_manifest(manifest_path, fixture.samples())

    def test_relabelled_framework_is_rejected_against_runner_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            manifest_path = fixture.execute()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run = manifest["runs"][0]
            sample_path = fixture.output / run["sample"]["relative_path"]
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            run["framework"] = "mlx-lm"
            sample["framework"] = "mlx-lm"
            sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            data = sample_path.read_bytes()
            run["sample"]["sha256"] = "sha256:" + hashlib.sha256(data).hexdigest()
            run["sample"]["size_bytes"] = len(data)
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "relabeled against runner argv"):
                verify_complete_campaign_manifest(manifest_path, fixture.samples())

    def test_grouped_or_reordered_framework_runs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            manifest_path = fixture.execute()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runs"] = [manifest["runs"][0], manifest["runs"][2], manifest["runs"][1], manifest["runs"][3]]
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "immutable launch order"):
                verify_complete_campaign_manifest(manifest_path, fixture.samples())

    def test_overlapping_run_intervals_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            manifest_path = fixture.execute()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runs"][1]["started_unix_ns"] = manifest["runs"][0]["completed_unix_ns"] - 1
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "overlapping"):
                verify_complete_campaign_manifest(manifest_path, fixture.samples())

    def test_missing_comparison_sample_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            manifest_path = fixture.execute()
            with self.assertRaisesRegex(ContractError, "membership"):
                verify_complete_campaign_manifest(manifest_path, fixture.samples()[:-1])

    def test_replaced_sample_is_rejected_by_size_or_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            manifest_path = fixture.execute()
            samples = fixture.samples()
            samples[0][0].write_bytes(samples[0][0].read_bytes() + b" ")
            with self.assertRaisesRegex(ContractError, "replaced or modified"):
                verify_complete_campaign_manifest(manifest_path, fixture.samples())

    def test_plan_cannot_self_assert_campaign_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            plan = json.loads(fixture.plan_path.read_text(encoding="utf-8"))
            plan["cells"][0]["commands"]["mlx-lm"].extend(("--sequence-index", "7"))
            fixture.plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "reserved to the orchestrator"):
                fixture.execute()

    def test_plan_preserves_virtual_environment_python_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            launcher = fixture.root / "venv-python"
            launcher.symlink_to(sys.executable)
            plan = json.loads(fixture.plan_path.read_text(encoding="utf-8"))
            for command in plan["cells"][0]["commands"].values():
                command[0] = str(launcher)
            fixture.plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
            normalized, _digest = load_plan(fixture.plan_path)
            self.assertEqual(
                str(launcher.absolute()),
                normalized["cells"][0]["commands"]["mlx-lm"][0],
            )

    def test_failed_child_never_publishes_complete_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            with self.assertRaisesRegex(ContractError, "COMPLETE was not published"):
                execute_campaign(
                    fixture.plan_path,
                    fixture.output,
                    run_process=lambda _argv: 17,
                    clock=lambda: next(fixture.clock_values),
                    token=lambda _size: next(fixture.tokens),
                )
            self.assertFalse((fixture.output / "COMPLETE.json").exists())

    def test_failed_directory_fsync_removes_complete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            calls = 0

            def fail_directory_fsync(_descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("fixture directory fsync failure")

            with mock.patch(
                "run_gemma4_lora_benchmark_campaign.os.fsync",
                side_effect=fail_directory_fsync,
            ):
                with self.assertRaisesRegex(OSError, "directory fsync failure"):
                    fixture.execute()
            self.assertFalse((fixture.output / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
