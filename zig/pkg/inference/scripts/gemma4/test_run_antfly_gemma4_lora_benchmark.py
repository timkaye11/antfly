from __future__ import annotations

import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_antfly_gemma4_lora_benchmark as runner
from bench_gemma4_lora_mlx_zig import benchmark_workload_sha256, validate_sample
from gemma4_oracle_contract import (
    BENCHMARK_PRODUCER_RELATIVE_PATHS,
    BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION,
    ContractError,
    LOCK_PATH,
    canonical_benchmark_producer_source_sha256,
    load_lock,
)


FAKE_RUNNER = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import sys
import time

if sys.argv[1:] == ["inference", "version"]:
    print("antfly inference v1.2.3-test")
    print("backends: native=true onnx=true onnx_runtime=false metal=true cuda=false")
    raise SystemExit(0)

if os.environ.get("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR") != "1":
    raise SystemExit(71)
if os.environ.get("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR") != "0":
    raise SystemExit(72)
if os.environ.get("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK") != "0":
    raise SystemExit(73)
if os.environ.get("TERMITE_DEBUG_DEVICE_GRAD_NORM") != "0":
    raise SystemExit(74)
required_environment = os.environ.get("ANTFLY_TEST_REQUIRE_ENVIRONMENT")
if required_environment:
    required_name, required_value = required_environment.split("=", 1)
    if os.environ.get(required_name) != required_value:
        raise SystemExit(75)
request_path = pathlib.Path(sys.argv[sys.argv.index("--benchmark-request") + 1])
telemetry_path = pathlib.Path(sys.argv[sys.argv.index("--benchmark-telemetry-out") + 1])
request = json.loads(request_path.read_text())
request_sha = "sha256:" + hashlib.sha256(request_path.read_bytes()).hexdigest()
protocol = request["protocol"]
bindings = request["bindings"]
signal_fd = int(os.environ[request["measurement_control"]["signal_fd_environment"]])
ack_fd = int(os.environ[request["measurement_control"]["ack_fd_environment"]])
steps = []
count = protocol["cold_optimizer_steps"] + protocol["first_steady_steps"] + protocol["warmup_steps"] + protocol["measured_steps"]
for index in range(count):
    phase = "cold" if index == 0 else "first" if index == 1 else "warmup" if index < 5 else "measured"
    phase_evidence = {
        "graph_build_ns": 10, "runtime_input_ns": 10, "train_step_ns": 100,
        "compile_ns": 500_000 if index == 0 else 0, "autodiff_ns": 10,
        "execute_ns": 20, "extract_ns": 10, "optimizer_update_ns": 10,
        "device_optimizer_ns": 5, "total_ns": 100, "metal_frame_wait_ns": 10,
        "metal_frame_gpu_ns": 20, "graph_executor_plan_build_ns": 20 if index == 0 else 0,
        "graph_executor_buffer_plan_build_ns": 10 if index == 0 else 0,
    }
    command_plan_evidence = {
        "graph_executor_partitions": bindings["grad_accum"],
        "graph_executor_command_dispatches": 9 * bindings["grad_accum"],
        "graph_executor_planned_dispatches": 8 * bindings["grad_accum"],
        "graph_executor_runtime_region_dispatches": bindings["grad_accum"],
        "graph_executor_runtime_region_active_regions": bindings["grad_accum"],
        "graph_executor_runtime_region_covered_nodes": 2 * bindings["grad_accum"],
        "graph_executor_runtime_region_elided_nodes": bindings["grad_accum"],
        "graph_executor_runtime_region_plan_compiles": 1 if index == 0 else 0,
        "graph_executor_runtime_region_plan_reuses": bindings["grad_accum"] - (1 if index == 0 else 0),
        "graph_executor_plan_cache_hits": bindings["grad_accum"] - (1 if index == 0 else 0),
        "graph_executor_plan_cache_misses": 1 if index == 0 else 0,
        "metal_lora_backward_regions": bindings["grad_accum"],
        "metal_low_rank_lora_backward_regions": 0, "metal_rank_adapter_backward_regions": 0,
        "metal_ffn_gelu_backward_regions": 0, "metal_head_mlp_forward_regions": 0,
        "metal_head_mlp_backward_regions": 0,
        "metal_gemma4_bf16_gate_up_fused_calls": bindings["grad_accum"],
        "metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls": 2 * bindings["grad_accum"],
        "metal_linear_cce_forward_calls": bindings["grad_accum"],
        "metal_linear_cce_backward_calls": bindings["grad_accum"],
        "metal_linear_cce_forward_state_hits": bindings["grad_accum"],
        "metal_linear_cce_forward_state_misses": 0,
        "metal_linear_cce_peak_scratch_bytes": 65536,
        "metal_command_dot_general_dispatches": 2 * bindings["grad_accum"],
        "metal_command_head_dot_dispatches": 0,
        "metal_command_transpose_dispatches": bindings["grad_accum"],
        "metal_command_gather_dispatches": bindings["grad_accum"],
        "metal_command_reduce_dispatches": bindings["grad_accum"],
        "metal_command_elementwise_dispatches": 2 * bindings["grad_accum"],
        "metal_command_activation_dispatches": bindings["grad_accum"],
        "metal_command_activation_backward_dispatches": 0,
        "metal_command_other_dispatches": bindings["grad_accum"],
        "metal_last_frame_compute_encoders": bindings["grad_accum"],
        "metal_last_frame_blit_encoders": 0, "metal_last_frame_planned_scopes": bindings["grad_accum"],
        "metal_last_frame_planned_barriers": 0,
        "metal_last_frame_planned_command_ops": 9 * bindings["grad_accum"],
    }
    steps.append({
        "index": index,
        "phase": phase,
        "duration_ns": 1_000_000 + index,
        "input_tokens": bindings["sequence_length"] * bindings["grad_accum"] * bindings["microbatch"],
        "supervised_tokens": bindings["supervised_tokens"],
        "optimizer_stepped": True,
        "explicit_device_sync": True,
        "strict_metal_evidence": {
            "optimizer_backend": "metal",
            "metal_optimizer_steps": 1,
            "graph_executor_steps": bindings["grad_accum"],
            "graph_executor_fallback_steps": 0,
            "native_partitions": 0,
            "unsupported_ops": 0,
            "interpreter_fallbacks": 0,
            "runtime_region_fallbacks": 0,
            "true_host_outputs": 0,
            "host_gradient_tensors": 0,
        },
        "phase_evidence": phase_evidence,
        "command_plan_evidence": command_plan_evidence,
    })
    if index == 4:
        os.write(signal_fd, b"B")
        if os.read(ack_fd, 1) != b"b":
            raise SystemExit(75)
        time.sleep(0.03)
    if index == count - 1:
        os.write(signal_fd, b"A")
        if os.read(ack_fd, 1) != b"a":
            raise SystemExit(76)
(telemetry_path.parent / "post-measured-work-started").write_text("save-and-post-eval", encoding="utf-8")
time.sleep(0.05)
payload = {
    "schema_version": "antfly_gemma4_lora_benchmark_telemetry/v4",
    "producer": {
        "pid": os.getpid(),
        "backend": "metal",
        "strict_metal_execution": True,
        "version": request["implementation"]["version"],
        "metal_device": request["implementation"]["metal_device"],
        "executable_sha256": request["implementation"]["executable_sha256"],
        "source_revision": request["implementation"]["source_revision"],
        "request_sha256": request_sha,
        "command_sha256": os.environ["ANTFLY_GEMMA4_BENCHMARK_COMMAND_SHA256"],
    },
    "bindings": bindings,
    "protocol": protocol,
    "runtime": request["runtime"],
    "measurement_control": request["measurement_control"],
    "timings": {
        "load_ns": 2_000_000,
        "cold_step_was_first_graph_execution": True,
        "cold_compile_and_step_ns": steps[0]["duration_ns"],
        "cold_compile_ns": 500_000,
        "first_steady_step_ns": steps[1]["duration_ns"],
        "warmup_step_ns": [step["duration_ns"] for step in steps[2:5]],
        "measured_step_ns": [step["duration_ns"] for step in steps[5:]],
        "optimizer_steps": steps,
    },
    "memory": {
        "peak_bytes": 987654321,
        "source": "darwin-proc-pid-rusage-v4-lifetime-max-phys-footprint",
    },
}
telemetry_path.write_text(json.dumps(payload), encoding="utf-8")
'''

FAKE_INCOMPATIBLE_RUNNER = r'''#!/bin/sh
if [ "$1" = "inference" ] && [ "$2" = "version" ]; then
  echo "antfly inference v1.2.3-test"
  exit 0
fi
exit 2
'''


def phase_evidence(index: int) -> dict[str, int]:
    return {
        field: (
            1 if index == 0 and field in (
                "compile_ns",
                "graph_executor_plan_build_ns",
                "graph_executor_buffer_plan_build_ns",
            ) else 0
        )
        for field in runner.PHASE_EVIDENCE_FIELDS
    }


def command_plan_evidence(index: int, grad_accum: int) -> dict[str, int]:
    evidence = {field: 0 for field in runner.COMMAND_PLAN_EVIDENCE_FIELDS}
    evidence.update({
        "graph_executor_partitions": grad_accum,
        "graph_executor_command_dispatches": 3 * grad_accum,
        "graph_executor_planned_dispatches": 2 * grad_accum,
        "graph_executor_plan_cache_hits": grad_accum - (1 if index == 0 else 0),
        "graph_executor_plan_cache_misses": 1 if index == 0 else 0,
        "metal_command_dot_general_dispatches": grad_accum,
        "metal_command_elementwise_dispatches": grad_accum,
        "metal_command_other_dispatches": grad_accum,
    })
    return evidence


def benchmark_producer_source(source_revision: str = "b" * 40) -> dict:
    files = [
        {
            "relative_path": relative_path,
            "source_sha256": "sha256:" + f"{index + 1:064x}",
        }
        for index, relative_path in enumerate(BENCHMARK_PRODUCER_RELATIVE_PATHS)
    ]
    relative_path = runner.ANTFLY_RUNNER_RELATIVE_PATH
    source_sha256 = next(
        item["source_sha256"] for item in files if item["relative_path"] == relative_path
    )
    source_tree = "c" * 40
    return {
        "schema_version": BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION,
        "relative_path": relative_path,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "source_clean": True,
        "source_sha256": source_sha256,
        "files": files,
        "manifest_sha256": canonical_benchmark_producer_source_sha256(
            relative_path=relative_path,
            source_revision=source_revision,
            source_tree=source_tree,
            files=files,
        ),
    }


def benchmark_diagnostic_producer_source(source_revision: str = "b" * 40) -> dict:
    release_source = benchmark_producer_source(source_revision)
    manifest_input = {
        "relative_path": release_source["relative_path"],
        "source_revision": release_source["source_revision"],
        "source_tree": release_source["source_tree"],
        "source_clean": False,
        "working_tree_status_sha256": "sha256:" + "d" * 64,
        "files": release_source["files"],
    }
    return {
        "schema_version": runner.DIAGNOSTIC_SOURCE_SCHEMA_VERSION,
        **manifest_input,
        "dirty_entry_count": 1,
        "source_sha256": release_source["source_sha256"],
        "manifest_sha256": runner._diagnostic_source_manifest_sha256(manifest_input),
    }


class AntflyGemma4BenchmarkRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = unittest.mock.patch.multiple(
            runner,
            source_identity=mock.DEFAULT,
            diagnostic_producer_source=mock.DEFAULT,
            attest_benchmark_producer_source=mock.DEFAULT,
            verify_model_directory=mock.DEFAULT,
            prepared_binding=mock.DEFAULT,
            load_prepared_example=mock.DEFAULT,
            verify_prepared_source_dataset=mock.DEFAULT,
            adapter_binding=mock.DEFAULT,
            _host_identity=mock.DEFAULT,
            darwin_phys_footprint_bytes=mock.DEFAULT,
            capture_darwin_system_memory_snapshot=mock.DEFAULT,
        )
        self.patches = self.stack.start()
        self.addCleanup(self.stack.stop)
        self.patches["capture_darwin_system_memory_snapshot"].side_effect = lambda *, before_measured: SimpleNamespace(
            page_size=4096,
            pageins=10,
            pageouts=20,
            swapins=30,
            swapouts=40,
            pressure_available_percent=80.0 if before_measured else 79.0,
        )

    def fixture(
        self,
        root: Path,
        *,
        compatible: bool = True,
        diagnostic_only: bool = False,
        diagnostic_env: tuple[str, ...] = (),
    ) -> tuple[Namespace, dict]:
        lock = load_lock(LOCK_PATH)
        source = root / "source"
        source.mkdir()
        model = root / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"model")
        adapter = root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        train = root / "train.json"
        train.write_text("{}", encoding="utf-8")
        eval_path = root / "eval.json"
        eval_path.write_text("{}", encoding="utf-8")
        executable = root / "antfly"
        executable.write_text(FAKE_RUNNER if compatible else FAKE_INCOMPATIBLE_RUNNER, encoding="utf-8")
        executable.chmod(0o755)

        def process_footprint(_pid: int) -> int:
            if any(root.rglob("post-measured-work-started")):
                raise AssertionError("process footprint sampled after the measured optimizer window")
            return 987654321

        self.patches["darwin_phys_footprint_bytes"].side_effect = process_footprint

        revision = "b" * 40
        self.patches["source_identity"].return_value = (source.resolve(), revision)
        self.patches["attest_benchmark_producer_source"].return_value = benchmark_producer_source(
            revision
        )
        self.patches["diagnostic_producer_source"].return_value = benchmark_diagnostic_producer_source(
            revision
        )
        model_revision = lock["models"]["gemma-4-E2B-it"]["revision"]
        self.patches["verify_model_directory"].return_value = {
            "model_key": "gemma-4-E2B-it",
            "revision": model_revision,
            "directory": str(model.resolve()),
            "local_artifact_sha256": "sha256:" + "d" * 64,
        }
        padded_ids = [1] * 128
        padded_labels = [-100] * 96 + [1] * 32
        workload = benchmark_workload_sha256([padded_ids], [padded_labels], [[1] * 128])
        train_info = train.stat()
        self.patches["prepared_binding"].return_value = runner.PreparedBinding(
            case={
                "schema_version": "gemma4_prepared/v6",
                "artifact_sha256": "sha256:" + "1" * 64,
                "example_index": 0,
                "source_dataset_sha256": "2" * 64,
                "source_record_sha256": "3" * 64,
                "rendered_chat_sha256": "4" * 64,
                "workload_sha256": workload,
            },
            workload_sha256=workload,
            input_tokens=128,
            supervised_tokens=32,
            artifact_sha256="sha256:" + "1" * 64,
            base_model_sha256="5" * 64,
            tokenizer_sha256="6" * 64,
            chat_template_sha256="7" * 64,
            snapshot=((train.name, train_info.st_dev, train_info.st_ino, train_info.st_size, train_info.st_mtime_ns),),
        )
        self.patches["load_prepared_example"].return_value = ({
            "base_model_sha256": "5" * 64,
            "tokenizer_sha256": "6" * 64,
            "chat_template_sha256": "7" * 64,
            "source_dataset_sha256": "8" * 64,
        }, {})
        canonical_modules = tuple(sorted(
            [f"model.layers.{index}.self_attn.q_proj" for index in range(35)]
            + [f"model.layers.{index}.self_attn.v_proj" for index in range(15)]
        ))
        self.patches["adapter_binding"].return_value = runner.AdapterBinding(
            semantic_sha256="sha256:" + "9" * 64,
            tensor_count=2 * len(canonical_modules),
            target_inventory_sha256=runner.canonical_target_inventory_sha256(canonical_modules),
            canonical_modules=canonical_modules,
            rank=16,
            alpha=32.0,
            target_preset="peft-qv",
            base_model_sha256="5" * 64,
            tokenizer_sha256="6" * 64,
            chat_template_sha256="7" * 64,
            snapshot=runner._tree_snapshot(adapter),
        )
        self.patches["_host_identity"].return_value = {
            "platform": "Darwin",
            "machine": "arm64",
            "chip": "Apple M4 Max",
            "memory_bytes": 64 * 1024**3,
            "os_version": "15.6",
            "os_build": "24G84",
            "metal_device": "Apple M4 Max",
        }
        argv = [
            "--antfly", str(executable),
            "--source-root", str(source),
            "--model-key", "gemma-4-E2B-it",
            "--model-dir", str(model),
            "--adapter-dir", str(adapter),
            "--train-prepared", str(train),
            "--eval-prepared", str(eval_path),
            "--example-index", "0",
            "--target-preset", "peft-qv",
            "--sequence-length", "128",
            "--grad-accum", "1",
            "--campaign-id", "campaign-1",
            "--run-id", "antfly-0",
            "--repetition", "0",
            "--sequence-index", "0",
            "--metal-device", "Apple M4 Max",
            "--timeout-seconds", "10",
            "--output", str(root / "sample.json"),
        ]
        if diagnostic_only:
            argv.append("--diagnostic-only")
        for entry in diagnostic_env:
            argv.extend(("--diagnostic-env", entry))
        args = runner.parser().parse_args(argv)
        return args, lock

    def test_fresh_subprocess_telemetry_becomes_schema_valid_sample(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, lock = self.fixture(root)
            stale_environment = {
                "TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_NODE_IDS": "1,2",
                "TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR": "0",
            }
            with mock.patch.dict(os.environ, stale_environment, clear=False):
                payload = runner.run(args)

            sample = validate_sample(args.output, lock, LOCK_PATH)
            self.assertEqual("antfly-zig-metal", sample.payload["framework"])
            self.assertEqual("1.2.3-test", payload["implementation"]["antfly"]["version"])
            self.assertEqual(
                benchmark_producer_source(),
                payload["implementation"]["producer_source"],
            )
            self.patches["attest_benchmark_producer_source"].assert_called_once_with(
                runner.SCRIPT_PATH,
                expected_entrypoint=runner.ANTFLY_RUNNER_RELATIVE_PATH,
            )
            self.assertEqual(20, len(payload["metrics"]["step_seconds"]))
            self.assertEqual(128, payload["metrics"]["input_tokens"])
            self.assertEqual(32, payload["metrics"]["supervised_tokens"])
            self.assertEqual(987654321, payload["metrics"]["memory"]["process_peak_phys_footprint_bytes"])
            self.assertEqual(-1.0, payload["metrics"]["memory"]["system_deltas"]["pressure_available_percent_delta"])
            self.assertEqual(
                [mock.call(before_measured=True), mock.call(before_measured=False)],
                self.patches["capture_darwin_system_memory_snapshot"].call_args_list,
            )
            self.assertNotEqual(os.getpid(), payload["process"]["pid"])
            self.assertTrue(args.output.is_file())

    def test_missing_zig_telemetry_interface_fails_closed_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args, _lock = self.fixture(Path(name), compatible=False)
            with self.assertRaisesRegex(runner.BenchmarkInterfaceUnavailable, "Required Zig change") as raised:
                runner.run(args)
            self.assertIn("complete optimizer window", str(raised.exception))
            self.assertFalse(args.output.exists())

    def test_atomic_publication_never_replaces_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args, _lock = self.fixture(Path(name))
            args.output.write_text("known-good\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "refusing to replace"):
                runner.run(args)
            self.assertEqual("known-good\n", args.output.read_text(encoding="utf-8"))

    def test_diagnostic_mode_is_content_bound_but_never_release_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, lock = self.fixture(root, diagnostic_only=True)
            with mock.patch.object(runner, "enforce_system_memory_gates") as memory_gate:
                payload = runner.run(args)

            self.assertEqual(runner.DIAGNOSTIC_SAMPLE_SCHEMA_VERSION, payload["schema_version"])
            self.assertFalse(payload["diagnostic"]["release_eligible"])
            self.assertFalse(payload["diagnostic"]["release_gates_enforced"])
            self.assertEqual({}, payload["diagnostic"]["environment_overrides"])
            self.assertFalse(payload["implementation"]["antfly"]["source_clean"])
            self.assertEqual(
                runner.DIAGNOSTIC_SOURCE_SCHEMA_VERSION,
                payload["implementation"]["producer_source"]["schema_version"],
            )
            self.assertEqual(2, self.patches["diagnostic_producer_source"].call_count)
            self.patches["source_identity"].assert_not_called()
            self.patches["attest_benchmark_producer_source"].assert_not_called()
            memory_gate.assert_not_called()
            runner.validate_diagnostic_payload(payload)
            with self.assertRaises(ContractError):
                validate_sample(args.output, lock, LOCK_PATH)

    def test_diagnostic_mode_rejects_producer_mutation_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, _lock = self.fixture(root, diagnostic_only=True)
            before = benchmark_diagnostic_producer_source()
            after = dict(before)
            after["working_tree_status_sha256"] = "sha256:" + "e" * 64
            self.patches["diagnostic_producer_source"].side_effect = [before, after]

            with self.assertRaisesRegex(ContractError, "producer source changed"):
                runner.run(args)
            self.assertFalse(args.output.exists())

    def test_diagnostic_environment_override_is_scoped_propagated_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            override = "TERMITE_METAL_DISABLE_GEMMA4_BF16_MLP_FUSION=1"
            args, _lock = self.fixture(
                root,
                diagnostic_only=True,
                diagnostic_env=(override,),
            )
            with mock.patch.dict(
                os.environ,
                {"ANTFLY_TEST_REQUIRE_ENVIRONMENT": override},
                clear=False,
            ):
                payload = runner.run(args)

            self.assertEqual(
                {"TERMITE_METAL_DISABLE_GEMMA4_BF16_MLP_FUSION": "1"},
                payload["diagnostic"]["environment_overrides"],
            )
            runner.validate_diagnostic_payload(payload)

    def test_diagnostic_environment_override_is_rejected_for_release_runs(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args, _lock = self.fixture(
                Path(name),
                diagnostic_env=("TERMITE_METAL_DISABLE_GEMMA4_BF16_MLP_FUSION=1",),
            )
            with self.assertRaisesRegex(ContractError, "requires --diagnostic-only"):
                runner.run(args)
            self.assertFalse(args.output.exists())

    def test_diagnostic_environment_override_cannot_weaken_strict_metal_contract(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args, _lock = self.fixture(
                Path(name),
                diagnostic_only=True,
                diagnostic_env=("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=0",),
            )
            with self.assertRaisesRegex(ContractError, "cannot replace locked strict-Metal variable"):
                runner.run(args)
            self.assertFalse(args.output.exists())

    def test_diagnostic_environment_override_rejects_non_termite_and_duplicates(self) -> None:
        with self.assertRaisesRegex(ContractError, "only TERMITE_"):
            runner.diagnostic_environment_overrides(["PATH=/tmp"], diagnostic_only=True)
        with self.assertRaisesRegex(ContractError, "repeats TERMITE_METAL_TEST"):
            runner.diagnostic_environment_overrides(
                ["TERMITE_METAL_TEST=0", "TERMITE_METAL_TEST=1"],
                diagnostic_only=True,
            )


class AntflyAdapterBindingTest(unittest.TestCase):
    def test_adapter_binding_accepts_current_v3_internal_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for filename in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "antfly_finetune_manifest.json",
            ):
                (root / filename).write_bytes(b"fixture")
            module = "model.layers.0.self_attn.q_proj"
            artifact = SimpleNamespace(
                semantics={
                    "policy_source": "antfly-finetune-manifest/v3",
                    "r": 16,
                    "lora_alpha": 32.0,
                    "provenance": {
                        "base_model_sha256": "a" * 64,
                        "tokenizer_sha256": "b" * 64,
                        "chat_template_sha256": "c" * 64,
                    },
                },
                semantic_sha256="sha256:" + "d" * 64,
                tensors={(module, "lora_A"): object(), (module, "lora_B"): object()},
            )
            lock = {"performance_gate": {"rank": 16, "alpha": 32.0}}
            with mock.patch.object(runner, "inspect_initial_adapter", return_value=artifact):
                binding = runner.adapter_binding(root, lock, "model", "peft-qv", {})
            self.assertEqual((module,), binding.canonical_modules)
            self.assertEqual("sha256:" + "d" * 64, binding.semantic_sha256)


class AntflySourceIdentityTest(unittest.TestCase):
    @staticmethod
    def git_result(*, stdout: str = "", returncode: int = 0) -> SimpleNamespace:
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    def test_accepts_clean_checkout_root_containing_exact_wrapper_path(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            wrapper = root / runner.ANTFLY_RUNNER_RELATIVE_PATH

            def git(command: list[str], **_kwargs: object) -> SimpleNamespace:
                if command[-2:] == ["rev-parse", "--show-toplevel"]:
                    return self.git_result(stdout=str(root) + "\n")
                if command[-2:] == ["rev-parse", "HEAD"]:
                    return self.git_result(stdout="b" * 40 + "\n")
                if command[-3:] == ["status", "--porcelain=v1", "--untracked-files=all"]:
                    return self.git_result()
                raise AssertionError(f"unexpected git command: {command}")

            with (
                mock.patch.object(runner, "SCRIPT_PATH", wrapper),
                mock.patch.object(runner.subprocess, "run", side_effect=git) as run_git,
            ):
                actual_root, revision = runner.source_identity(root)

            self.assertEqual(root, actual_root)
            self.assertEqual("b" * 40, revision)
            self.assertEqual(3, run_git.call_count)

    def test_rejects_source_root_from_another_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            container = Path(name).resolve()
            wrapper_root = container / "wrapper-checkout"
            supplied_root = container / "other-checkout"
            wrapper_root.mkdir()
            supplied_root.mkdir()
            wrapper = wrapper_root / runner.ANTFLY_RUNNER_RELATIVE_PATH

            with (
                mock.patch.object(runner, "SCRIPT_PATH", wrapper),
                mock.patch.object(
                    runner.subprocess,
                    "run",
                    return_value=self.git_result(stdout=str(supplied_root) + "\n"),
                ),
            ):
                with self.assertRaisesRegex(ContractError, "outside --source-root"):
                    runner.source_identity(supplied_root)

    def test_rejects_dirty_checkout_including_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            wrapper = root / runner.ANTFLY_RUNNER_RELATIVE_PATH

            def git(command: list[str], **_kwargs: object) -> SimpleNamespace:
                if command[-2:] == ["rev-parse", "--show-toplevel"]:
                    return self.git_result(stdout=str(root) + "\n")
                if command[-2:] == ["rev-parse", "HEAD"]:
                    return self.git_result(stdout="b" * 40 + "\n")
                if command[-3:] == ["status", "--porcelain=v1", "--untracked-files=all"]:
                    return self.git_result(stdout="?? untracked-file\n")
                raise AssertionError(f"unexpected git command: {command}")

            with (
                mock.patch.object(runner, "SCRIPT_PATH", wrapper),
                mock.patch.object(runner.subprocess, "run", side_effect=git),
            ):
                with self.assertRaisesRegex(ContractError, "including untracked files"):
                    runner.source_identity(root)

    def test_diagnostic_identity_records_dirty_checkout_without_claiming_cleanliness(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            for relative_path in BENCHMARK_PRODUCER_RELATIVE_PATHS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative_path, encoding="utf-8")
            wrapper = root / runner.ANTFLY_RUNNER_RELATIVE_PATH
            dirty_status = b" M zig/pkg/inference/source.zig\n"

            def git(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
                arguments = command[3:]
                if arguments == ("rev-parse", "--show-toplevel"):
                    return SimpleNamespace(stdout=str(root) + "\n", stderr="", returncode=0)
                if arguments == ("rev-parse", "HEAD"):
                    return SimpleNamespace(stdout="b" * 40 + "\n", stderr="", returncode=0)
                if arguments == ("rev-parse", "HEAD^{tree}"):
                    return SimpleNamespace(stdout="c" * 40 + "\n", stderr="", returncode=0)
                if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
                    self.assertFalse(kwargs["text"])
                    return SimpleNamespace(stdout=dirty_status, stderr=b"", returncode=0)
                raise AssertionError(f"unexpected git command: {command}")

            with (
                mock.patch.object(runner, "SCRIPT_PATH", wrapper),
                mock.patch.object(runner.subprocess, "run", side_effect=git),
            ):
                source = runner.diagnostic_producer_source(root)

            self.assertFalse(source["source_clean"])
            self.assertEqual(1, source["dirty_entry_count"])
            self.assertEqual("sha256:" + __import__("hashlib").sha256(dirty_status).hexdigest(), source["working_tree_status_sha256"])
            self.assertEqual(runner.DIAGNOSTIC_SOURCE_SCHEMA_VERSION, source["schema_version"])


class AntflyExecutableVersionTest(unittest.TestCase):
    def test_accepts_one_canonical_version_record_on_stderr(self) -> None:
        result = SimpleNamespace(
            returncode=0,
            stdout="",
            stderr=(
                "antfly inference vdev\n"
                "backends: native=true onnx=true onnx_runtime=false metal=true cuda=false\n"
            ),
        )
        with mock.patch.object(runner.subprocess, "run", return_value=result):
            self.assertEqual("dev", runner.executable_version(Path("/tmp/antfly")))

    def test_rejects_ambiguous_version_records_across_output_streams(self) -> None:
        result = SimpleNamespace(
            returncode=0,
            stdout="antfly inference vdev\n",
            stderr="antfly inference vother\n",
        )
        with mock.patch.object(runner.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ContractError, "missing or ambiguous"):
                runner.executable_version(Path("/tmp/antfly"))


class AntflyTelemetryValidationTest(unittest.TestCase):
    def telemetry(self) -> tuple[dict, dict]:
        protocol = {
            "fresh_process": True,
            "cold_optimizer_steps": 1,
            "cold_step_mutates_optimizer_state": True,
            "first_steady_steps": 1,
            "warmup_steps": 3,
            "measured_steps": 20,
            "explicit_device_sync": True,
            "sync_point": runner.SYNC_POINT,
            "timed_unit": runner.TIMED_UNIT,
        }
        bindings = {
            "sequence_length": 128,
            "grad_accum": 4,
            "microbatch": 1,
            "supervised_tokens": 12,
        }
        request = {
            "implementation": {
                "version": "1.2.3-test",
                "metal_device": "Apple M4 Max",
                "executable_sha256": "sha256:" + "a" * 64,
                "source_revision": "b" * 40,
            },
            "bindings": bindings,
            "protocol": protocol,
            "runtime": dict(load_lock(LOCK_PATH)["benchmark_contract"]["runtime"]),
            "measurement_control": dict(runner.MEASUREMENT_CONTROL),
        }
        steps = []
        for index in range(25):
            steps.append({
                "index": index,
                "phase": "cold" if index == 0 else "first" if index == 1 else "warmup" if index < 5 else "measured",
                "duration_ns": 10,
                "input_tokens": 512,
                "supervised_tokens": 12,
                "optimizer_stepped": True,
                "explicit_device_sync": True,
                "strict_metal_evidence": {
                    "optimizer_backend": "metal",
                    "metal_optimizer_steps": 1,
                    "graph_executor_steps": 4,
                    "graph_executor_fallback_steps": 0,
                    "native_partitions": 0,
                    "unsupported_ops": 0,
                    "interpreter_fallbacks": 0,
                    "runtime_region_fallbacks": 0,
                    "true_host_outputs": 0,
                    "host_gradient_tensors": 0,
                },
                "phase_evidence": phase_evidence(index),
                "command_plan_evidence": command_plan_evidence(index, 4),
            })
        telemetry = {
            "schema_version": runner.TELEMETRY_SCHEMA_VERSION,
            "producer": {
                "pid": 123,
                "backend": "metal",
                "strict_metal_execution": True,
                "version": "1.2.3-test",
                "metal_device": "Apple M4 Max",
                "executable_sha256": "sha256:" + "a" * 64,
                "source_revision": "b" * 40,
                "request_sha256": "sha256:" + "c" * 64,
                "command_sha256": "sha256:" + "d" * 64,
            },
            "bindings": bindings,
            "protocol": protocol,
            "runtime": request["runtime"],
            "measurement_control": request["measurement_control"],
            "timings": {
                "load_ns": 0,
                "cold_step_was_first_graph_execution": True,
                "cold_compile_and_step_ns": 10,
                "cold_compile_ns": 1,
                "first_steady_step_ns": 10,
                "warmup_step_ns": [10, 10, 10],
                "measured_step_ns": [10] * 20,
                "optimizer_steps": steps,
            },
            "memory": {"peak_bytes": 1, "source": runner.PEAK_MEMORY_SOURCE},
        }
        return request, telemetry

    def validate(self, root: Path, request: dict, telemetry: dict) -> dict:
        path = root / "telemetry.json"
        path.write_text(json.dumps(telemetry), encoding="utf-8")
        return runner._validate_telemetry(
            path,
            request=request,
            request_sha256="sha256:" + "c" * 64,
            command_sha256="sha256:" + "d" * 64,
            pid=123,
        )

    def test_rejects_unsynchronized_optimizer_window(self) -> None:
        request, telemetry = self.telemetry()
        telemetry["timings"]["optimizer_steps"][4]["explicit_device_sync"] = False
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ContractError, "not a synchronized complete optimizer step"):
                self.validate(Path(name), request, telemetry)

    def test_rejects_cold_window_after_prior_graph_execution(self) -> None:
        request, telemetry = self.telemetry()
        telemetry["timings"]["cold_step_was_first_graph_execution"] = False
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ContractError, "first graph execution"):
                self.validate(Path(name), request, telemetry)

    def test_rejects_product_version_mismatch_independently_of_source_revision(self) -> None:
        request, telemetry = self.telemetry()
        telemetry["producer"]["version"] = "different-product-version"
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ContractError, "version differs"):
                self.validate(Path(name), request, telemetry)

    def test_rejects_epoch_or_last_microbatch_token_totals(self) -> None:
        request, telemetry = self.telemetry()
        telemetry["timings"]["optimizer_steps"][4]["input_tokens"] = 128
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ContractError, "token totals differ"):
                self.validate(Path(name), request, telemetry)

    def test_rejects_non_authoritative_peak_memory_source(self) -> None:
        request, telemetry = self.telemetry()
        telemetry["memory"]["source"] = "sampled-ps-rss"
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ContractError, "lifetime maximum physical footprint"):
                self.validate(Path(name), request, telemetry)

    def test_rejects_any_strict_metal_fallback(self) -> None:
        request, telemetry = self.telemetry()
        telemetry["timings"]["optimizer_steps"][8]["strict_metal_evidence"]["true_host_outputs"] = 1
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ContractError, "forbidden strict-Metal fallback"):
                self.validate(Path(name), request, telemetry)

    def test_rejects_legacy_telemetry_without_phase_and_command_plan_evidence(self) -> None:
        request, telemetry = self.telemetry()
        telemetry["schema_version"] = "antfly_gemma4_lora_benchmark_telemetry/v2"
        for step in telemetry["timings"]["optimizer_steps"]:
            step.pop("phase_evidence")
            step.pop("command_plan_evidence")
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ContractError, "unsupported Antfly benchmark telemetry schema"):
                self.validate(Path(name), request, telemetry)

    def test_rejects_warm_command_plan_cache_miss(self) -> None:
        request, telemetry = self.telemetry()
        command = telemetry["timings"]["optimizer_steps"][5]["command_plan_evidence"]
        command["graph_executor_plan_cache_hits"] -= 1
        command["graph_executor_plan_cache_misses"] += 1
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ContractError, "unexpected cache miss count"):
                self.validate(Path(name), request, telemetry)

    def test_rejects_command_family_over_attribution(self) -> None:
        request, telemetry = self.telemetry()
        command = telemetry["timings"]["optimizer_steps"][5]["command_plan_evidence"]
        command["metal_command_other_dispatches"] = command["graph_executor_command_dispatches"]
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ContractError, "command attribution exceeds"):
                self.validate(Path(name), request, telemetry)

    def test_rejects_command_plan_without_gemma4_bf16_gate_up_attribution(self) -> None:
        request, telemetry = self.telemetry()
        telemetry["timings"]["optimizer_steps"][5]["command_plan_evidence"].pop(
            "metal_gemma4_bf16_gate_up_fused_calls"
        )
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ContractError, "metal_gemma4_bf16_gate_up_fused_calls"):
                self.validate(Path(name), request, telemetry)

    def test_rejects_command_plan_without_gemma4_bf16_gate_up_backward_input_sum_attribution(self) -> None:
        request, telemetry = self.telemetry()
        telemetry["timings"]["optimizer_steps"][5]["command_plan_evidence"].pop(
            "metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls"
        )
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(
                ContractError,
                "metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls",
            ):
                self.validate(Path(name), request, telemetry)


class AntflyPreparedWorkloadBindingTest(unittest.TestCase):
    def test_exact_selected_row_is_repeated_for_every_accumulation_microstep(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            prepared = root / "prepared.json"
            prepared.write_text("{}", encoding="utf-8")
            summary = {
                "max_seq_len": 128,
                "base_model_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "chat_template_sha256": "c" * 64,
            }

            def example(_path: Path, index: int) -> tuple[dict, dict]:
                return summary, {
                    "schema_version": "gemma4_prepared/v6",
                    "input_ids": [index + 1] * 128,
                    "labels": [-100] * 127 + [index + 1],
                    "source_dataset_sha256": "d" * 64,
                    "source_record_sha256": f"{index + 1:064x}",
                    "rendered_chat_sha256": f"{index + 5:064x}",
                }

            with (
                mock.patch.object(runner, "load_prepared_example", side_effect=example) as loader,
                mock.patch.object(runner, "verify_prepared_source_dataset") as source_check,
            ):
                binding = runner.prepared_binding(
                    prepared,
                    sequence_length=128,
                    grad_accum=4,
                    example_index=3,
                    source_dataset=None,
                )

            input_rows = [[4] * 128 for _index in range(4)]
            label_rows = [[-100] * 127 + [4] for _index in range(4)]
            mask_rows = [[1] * 128 for _index in range(4)]
            self.assertEqual(
                benchmark_workload_sha256(input_rows, label_rows, mask_rows),
                binding.workload_sha256,
            )
            self.assertEqual(512, binding.input_tokens)
            self.assertEqual(4, binding.supervised_tokens)
            loader.assert_called_once_with(prepared.resolve(), 3)
            source_check.assert_called_once_with(summary, None)

    def test_prepared_max_sequence_length_must_equal_cell(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            prepared = Path(name) / "prepared.json"
            prepared.write_text("{}", encoding="utf-8")
            summary = {
                "max_seq_len": 512,
                "base_model_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "chat_template_sha256": "c" * 64,
            }
            selected = {
                "schema_version": "gemma4_prepared/v6",
                "input_ids": [1, 2],
                "labels": [-100, 2],
                "source_dataset_sha256": "d" * 64,
                "source_record_sha256": "e" * 64,
                "rendered_chat_sha256": "f" * 64,
            }
            with (
                mock.patch.object(runner, "load_prepared_example", return_value=(summary, selected)),
                mock.patch.object(runner, "verify_prepared_source_dataset"),
            ):
                with self.assertRaisesRegex(ContractError, "must equal the benchmark cell"):
                    runner.prepared_binding(
                        prepared,
                        sequence_length=128,
                        grad_accum=1,
                        example_index=0,
                        source_dataset=None,
                    )

    def test_adapter_directory_digest_rejects_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "file").write_text("content", encoding="utf-8")
            try:
                (root / "alias").symlink_to(root / "file")
            except OSError:
                self.skipTest("symbolic links are unavailable")
            with self.assertRaisesRegex(ContractError, "symbolic link"):
                runner._tree_snapshot(root)


if __name__ == "__main__":
    unittest.main()
