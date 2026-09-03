from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from bench_gemma4_lora_mlx_zig import (
    MLX_RUNNER_RELATIVE_PATH,
    canonical_initial_adapter_sha256,
    canonical_moment_inventory_sha256,
    canonical_tensor_inventory_sha256,
    validate_sample,
)
from gemma4_oracle_contract import (
    BENCHMARK_PRODUCER_RELATIVE_PATHS,
    BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION,
    ContractError,
    LOCK_PATH,
    canonical_benchmark_producer_source_sha256,
    canonical_mlx_native_artifact_inventory_sha256,
    load_lock,
)
from run_gemma4_lora_mlx_benchmark import (
    CONTROL_PROTOCOL_VERSION,
    DIAGNOSTIC_SAMPLE_SCHEMA_VERSION,
    DIAGNOSTIC_SOURCE_SCHEMA_VERSION,
    OFFLINE_ENVIRONMENT,
    PREPARED_CHAT_TEMPLATE_IDENTITY,
    AdapterArtifact,
    AdapterTensor,
    DarwinProcessMemorySampler,
    DarwinSystemMemorySnapshot,
    JsonControlChannel,
    Preflight,
    ProcessMemoryMeasurement,
    PrecisionEvidenceRecorder,
    WorkerPhaseReporter,
    _coordinator_result,
    atomic_publish_json,
    build_diagnostic_payload,
    build_precision_evidence,
    build_sample_payload,
    build_workload,
    canonicalize_loaded_dyld_image_path,
    capture_file_identities,
    darwin_system_memory_deltas,
    force_offline_environment,
    inspect_initial_adapter,
    expected_unused_shared_kv_parameter_names,
    measure_steps,
    parse_memory_pressure_available_percent,
    parse_vm_stat,
    precision_evidence_output_path,
    precision_evidence_reference,
    require_files_unchanged,
    require_f32_gradient_inventory,
    require_f32_optimizer_state,
    require_prepared_model_binding,
    runner_source_attestation,
    target_module_names,
    validate_mlx_gemma4_checkpoint_coverage,
    validate_diagnostic_payload,
    verify_mlx_native_build_before_import,
    verify_mlx_environment,
    verify_mlx_native_runtime,
    zig_model_provenance,
)


class FakeModel:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def named_modules(self):
        return [(name, object()) for name in self.names]


class Gemma4MlxRunnerTest(unittest.TestCase):
    @staticmethod
    def producer_source() -> dict:
        files = [
            {
                "relative_path": relative_path,
                "source_sha256": "sha256:"
                + hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
            }
            for relative_path in BENCHMARK_PRODUCER_RELATIVE_PATHS
        ]
        source_revision = "b" * 40
        source_tree = "d" * 40
        entrypoint_sha256 = next(
            item["source_sha256"]
            for item in files
            if item["relative_path"] == MLX_RUNNER_RELATIVE_PATH
        )
        return {
            "schema_version": BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION,
            "relative_path": MLX_RUNNER_RELATIVE_PATH,
            "source_revision": source_revision,
            "source_tree": source_tree,
            "source_clean": True,
            "source_sha256": entrypoint_sha256,
            "files": files,
            "manifest_sha256": canonical_benchmark_producer_source_sha256(
                relative_path=MLX_RUNNER_RELATIVE_PATH,
                source_revision=source_revision,
                source_tree=source_tree,
                files=files,
            ),
        }

    @staticmethod
    def write_safetensors(path: Path, tensors: list[tuple[str, tuple[int, int], list[float]]]) -> None:
        offset = 0
        header: dict[str, dict] = {}
        payload = bytearray()
        for name, shape, values in tensors:
            encoded = struct.pack(f"<{len(values)}f", *values)
            header[name] = {
                "dtype": "F32",
                "shape": list(shape),
                "data_offsets": [offset, offset + len(encoded)],
            }
            offset += len(encoded)
            payload.extend(encoded)
        raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)

    @staticmethod
    def precision_observations(payload: dict) -> dict:
        modules = payload["case"]["target_inventory"]["canonical_modules"]
        lora_tensors = sorted(
            (
                {"name": f"{module}.{suffix}", "dtype": "float32", "shape": [16, 16]}
                for module in modules
                for suffix in ("lora_a", "lora_b")
            ),
            key=lambda tensor: tensor["name"],
        )
        moments = sorted(
            (
                {
                    "name": f"{tensor['name']}.{role}",
                    "parameter_name": tensor["name"],
                    "role": role,
                    "dtype": "float32",
                    "shape": tensor["shape"],
                }
                for tensor in lora_tensors
                for role in ("m", "v")
            ),
            key=lambda moment: moment["name"],
        )
        recorder = PrecisionEvidenceRecorder()
        base_tensors = [
            {"name": "model.embed_tokens.weight", "dtype": "bfloat16", "shape": [2, 2]}
        ]
        recorder.record(
            "base_model_storage",
            {
                "evidence_kind": "materialized-parameter-inventory",
                "dtype": "bfloat16",
                "tensor_count": 1,
                "inventory_sha256": canonical_tensor_inventory_sha256(base_tensors),
                "tensors": base_tensors,
            },
        )
        recorder.record(
            "lora_parameter_storage",
            {
                "evidence_kind": "materialized-trainable-parameter-inventory",
                "dtype": "float32",
                "tensor_count": len(lora_tensors),
                "inventory_sha256": canonical_tensor_inventory_sha256(lora_tensors),
                "tensors": lora_tensors,
            },
        )
        for stage in ("raw", "accumulated", "clipped"):
            recorder.record_gradient(stage, lora_tensors)
        recorder.record(
            "optimizer_moment_storage",
            {
                "evidence_kind": "materialized-post-cold-adamw-moment-inventory",
                "dtype": "float32",
                "parameter_count": len(lora_tensors),
                "moment_tensor_count": len(moments),
                "inventory_sha256": canonical_moment_inventory_sha256(moments),
                "moments": moments,
            },
        )
        recorder.record(
            "loss",
            {
                "evidence_kind": "evaluated-training-loss-graph",
                "loss_tensor_dtype": "float32",
                "reduction_input_dtype": "float32",
            },
        )
        return recorder.finalize()

    def test_import_does_not_require_mlx(self) -> None:
        source = Path(__file__).with_name("run_gemma4_lora_mlx_benchmark.py").read_text(
            encoding="utf-8"
        )
        prefix = source.split("def verify_mlx_environment", 1)[0]
        self.assertNotIn("import mlx\n", prefix)
        self.assertNotIn("import mlx.", prefix)

    def test_offline_environment_overrides_ambient_values(self) -> None:
        previous = {name: os.environ.get(name) for name in OFFLINE_ENVIRONMENT}
        previous_dyld = os.environ.get("DYLD_LIBRARY_PATH")
        previous_dont_write_bytecode = sys.dont_write_bytecode
        try:
            for name in OFFLINE_ENVIRONMENT:
                os.environ[name] = "0"
            os.environ["DYLD_LIBRARY_PATH"] = "/untrusted"
            force_offline_environment()
            self.assertEqual(OFFLINE_ENVIRONMENT, {name: os.environ[name] for name in OFFLINE_ENVIRONMENT})
            self.assertNotIn("DYLD_LIBRARY_PATH", os.environ)
            self.assertTrue(sys.dont_write_bytecode)
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
            if previous_dyld is None:
                os.environ.pop("DYLD_LIBRARY_PATH", None)
            else:
                os.environ["DYLD_LIBRARY_PATH"] = previous_dyld
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_workload_binds_all_gradient_accumulation_rows_and_causal_tokens(self) -> None:
        workload = build_workload(
            {"input_ids": [7, 8, 9, 10], "labels": [-100, -100, 9, 10]},
            sequence_length=4,
            grad_accum=3,
        )
        self.assertEqual(12, workload.input_tokens)
        self.assertEqual(6, workload.supervised_tokens)
        self.assertEqual((1, 1, 1, 1), workload.attention_mask)
        self.assertRegex(workload.digest, r"^sha256:[0-9a-f]{64}$")
        changed = build_workload(
            {"input_ids": [7, 8, 9, 10], "labels": [-100, -100, -100, 10]},
            sequence_length=4,
            grad_accum=3,
        )
        self.assertNotEqual(workload.digest, changed.digest)

    def test_workload_rejects_padding_truncation_and_empty_causal_supervision(self) -> None:
        with self.assertRaisesRegex(ContractError, "may not pad or truncate"):
            build_workload(
                {"input_ids": [1, 2], "labels": [-100, 2]},
                sequence_length=4,
                grad_accum=1,
            )
        with self.assertRaisesRegex(ContractError, "no causal supervised"):
            build_workload(
                {"input_ids": [1, 2], "labels": [2, -100]},
                sequence_length=2,
                grad_accum=1,
            )

    def test_measurement_executes_compile_first_warmup_and_exact_measured_steps(self) -> None:
        calls = {"execute": 0, "sync": 0, "clock": 0}
        boundaries: list[tuple[str, int]] = []

        def synchronize() -> None:
            calls["sync"] += 1

        def execute() -> float:
            calls["execute"] += 1
            return 1.0

        def clock() -> float:
            value = calls["clock"] * 0.25
            calls["clock"] += 1
            return value

        measured = measure_steps(
            synchronize,
            execute,
            warmup_steps=3,
            measured_steps=20,
            clock=clock,
            result_is_finite=lambda value: value == 1.0,
            after_cold=lambda _value: boundaries.append(("cold-audit", calls["execute"])),
            before_measured=lambda: boundaries.append(("start", calls["execute"])),
            after_measured=lambda: boundaries.append(("end", calls["execute"])),
        )
        self.assertEqual(25, calls["execute"])
        self.assertEqual(50, calls["sync"])
        self.assertEqual(0.25, measured.cold_compile_and_step_seconds)
        self.assertEqual(0.25, measured.first_steady_step_seconds)
        self.assertEqual((0.25,) * 20, measured.step_seconds)
        self.assertEqual(
            [("cold-audit", 1), ("start", 5), ("end", 25)],
            boundaries,
        )

    def test_measurement_rejects_nonfinite_loss(self) -> None:
        tick = iter((0.0, 1.0))
        with self.assertRaisesRegex(ContractError, "non-finite loss"):
            measure_steps(
                lambda: None,
                lambda: float("nan"),
                warmup_steps=0,
                measured_steps=1,
                clock=lambda: next(tick),
                result_is_finite=lambda value: value == value,
            )

    def test_optimizer_state_audit_requires_f32_moments(self) -> None:
        class Array:
            def __init__(self, dtype: object, shape: tuple[int, ...] = (2, 2)) -> None:
                self.dtype = dtype
                self.shape = shape
                self.size = math.prod(shape) if shape else 1

        class Model:
            def trainable_parameters(self):
                return [("a", Array("f32")), ("b", Array("f32"))]

        class Optimizer:
            state = [
                ("step", Array("u64", ())),
                ("learning_rate", Array("f32", ())),
                ("a.m", Array("f32")),
                ("a.v", Array("f32")),
                ("b.m", Array("f32")),
                ("b.v", Array("f32")),
            ]

        mx = argparse.Namespace(float32="f32", uint64="u64")
        observation = require_f32_optimizer_state(
            Model(),
            Optimizer(),
            mx,
            tree_flatten_fn=lambda value: value,
        )
        self.assertEqual(2, observation["parameter_count"])
        self.assertEqual(4, observation["moment_tensor_count"])
        self.assertEqual(
            canonical_moment_inventory_sha256(observation["moments"]),
            observation["inventory_sha256"],
        )
        Optimizer.state[2] = ("a.m", Array("bf16"))
        with self.assertRaisesRegex(ContractError, "a.m is not float32"):
            require_f32_optimizer_state(
                Model(),
                Optimizer(),
                mx,
                tree_flatten_fn=lambda value: value,
            )

    def test_optimizer_state_audit_rejects_missing_extra_misnamed_and_wrong_shape(self) -> None:
        class Array:
            def __init__(self, dtype: object, shape: tuple[int, ...] = (2, 2)) -> None:
                self.dtype = dtype
                self.shape = shape
                self.size = math.prod(shape) if shape else 1

        class Model:
            def trainable_parameters(self):
                return [("a", Array("f32"))]

        valid = [
            ("step", Array("u64", ())),
            ("learning_rate", Array("f32", ())),
            ("a.m", Array("f32")),
            ("a.v", Array("f32")),
        ]
        mx = argparse.Namespace(float32="f32", uint64="u64")
        mutations = (
            ("missing", valid[:-1], "inventory drift"),
            ("extra", [*valid, ("extra", Array("f32"))], "inventory drift"),
            ("misnamed", [*valid[:2], ("a.m", Array("f32")), ("a.velocity", Array("f32"))], "inventory drift"),
            ("shape", [*valid[:2], ("a.m", Array("f32", (1, 4))), ("a.v", Array("f32"))], "shape differs"),
            ("learning-rate", [("step", Array("u64", ())), ("learning_rate", Array("bf16", ())), *valid[2:]], "learning-rate state"),
        )
        for name, state, message in mutations:
            with self.subTest(name=name):
                optimizer = argparse.Namespace(state=state)
                with self.assertRaisesRegex(ContractError, message):
                    require_f32_optimizer_state(
                        Model(), optimizer, mx, tree_flatten_fn=lambda value: value,
                    )

    def test_gradient_inventory_audit_rejects_name_dtype_and_shape_drift(self) -> None:
        class Array:
            def __init__(self, dtype: object, shape: tuple[int, ...] = (2, 2)) -> None:
                self.dtype = dtype
                self.shape = shape

        mx = argparse.Namespace(float32="f32")
        expected = [{"name": "a", "dtype": "float32", "shape": [2, 2]}]
        self.assertEqual(
            expected,
            require_f32_gradient_inventory(
                [("a", Array("f32"))],
                expected,
                mx,
                where="test gradients",
                tree_flatten_fn=lambda value: value,
            ),
        )
        for name, gradients, message in (
            ("name", [("b", Array("f32"))], "inventory differs"),
            ("dtype", [("a", Array("bf16"))], "not float32"),
            ("shape", [("a", Array("f32", (1, 4)))], "inventory differs"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ContractError, message):
                    require_f32_gradient_inventory(
                        gradients,
                        expected,
                        mx,
                        where="test gradients",
                        tree_flatten_fn=lambda value: value,
                    )

    def test_precision_recorder_rejects_incomplete_and_changing_observations(self) -> None:
        recorder = PrecisionEvidenceRecorder()
        tensor = [{"name": "a", "dtype": "float32", "shape": [1]}]
        recorder.record_gradient("raw", tensor)
        with self.assertRaisesRegex(ContractError, "precision evidence is incomplete"):
            recorder.finalize()
        with self.assertRaisesRegex(ContractError, "changed within one run"):
            recorder.record_gradient(
                "raw", [{"name": "a", "dtype": "float32", "shape": [2]}]
            )

    def test_runner_source_attestation_requires_tracked_clean_exact_source(self) -> None:
        from unittest.mock import patch

        source = self.producer_source()
        with patch(
            "run_gemma4_lora_mlx_benchmark.attest_benchmark_producer_source",
            return_value=source,
        ) as attest:
            attestation = runner_source_attestation()
        self.assertEqual(source, attestation)
        attest.assert_called_once_with(
            Path(__file__).with_name("run_gemma4_lora_mlx_benchmark.py").resolve(),
            expected_entrypoint=MLX_RUNNER_RELATIVE_PATH,
        )

        wrong_schema = dict(source)
        wrong_schema["schema_version"] = "wrong"
        with patch(
            "run_gemma4_lora_mlx_benchmark.attest_benchmark_producer_source",
            return_value=wrong_schema,
        ):
            with self.assertRaisesRegex(ContractError, "source schema drifted"):
                runner_source_attestation()

    def test_atomic_publication_is_validated_fsynced_and_no_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "sample.json"
            validated: list[dict] = []

            def validator(path: Path) -> None:
                validated.append(json.loads(path.read_text(encoding="utf-8")))

            published = atomic_publish_json(output, {"b": 2, "a": 1}, validator=validator)
            self.assertEqual(output.resolve(), published)
            self.assertEqual([{"a": 1, "b": 2}], validated)
            self.assertEqual({"a": 1, "b": 2}, json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual(0o644, stat.S_IMODE(output.stat().st_mode))
            self.assertEqual([], list(root.glob(".*.tmp")))
            with self.assertRaisesRegex(ContractError, "refusing to replace"):
                atomic_publish_json(output, {"a": 3})

    def test_failed_atomic_validation_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "sample.json"

            def reject(_path: Path) -> None:
                raise ContractError("rejected")

            with self.assertRaisesRegex(ContractError, "rejected"):
                atomic_publish_json(output, {"a": 1}, validator=reject)
            self.assertFalse(output.exists())
            self.assertEqual([], list(root.iterdir()))

    def test_zig_model_provenance_binds_model_tokenizer_and_chat_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"model_type":"gemma4"}\n', encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"model")
            (root / "tokenizer.json").write_bytes(b"tokenizer")
            (root / "tokenizer_config.json").write_bytes(b"config")
            first = zig_model_provenance(root)
            self.assertEqual(
                hashlib.sha256(PREPARED_CHAT_TEMPLATE_IDENTITY).hexdigest(),
                first["chat_template_sha256"],
            )
            require_prepared_model_binding(first, root)
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ContractError, "base_model_sha256"):
                require_prepared_model_binding(first, root)

    def test_initial_adapter_inspection_binds_f32_values_and_canonical_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 2,
                "lora_alpha": 4.0,
                "lora_dropout": 0.0,
                "inference_mode": False,
                "target_modules": ["q_proj"],
            }
            (root / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
            self.write_safetensors(
                root / "adapter_model.safetensors",
                [
                    ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", (2, 3), [1.0] * 6),
                    ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", (4, 2), [0.0] * 8),
                ],
            )
            lock = {
                "performance_gate": {"rank": 2, "alpha": 4.0},
                "target_inventory": {"model": {"peft-qv": {"q_proj": 1}}},
            }
            artifact = inspect_initial_adapter(root, lock, "model", "peft-qv", {})
            self.assertEqual(2, len(artifact.tensors))
            self.assertEqual({"model.layers.0.self_attn.q_proj"}, {key[0] for key in artifact.tensors})
            self.assertTrue(all(tensor.data_sha256.startswith("sha256:") for tensor in artifact.tensors.values()))
            self.assertEqual(
                canonical_initial_adapter_sha256(
                    [
                        {
                            "module": "model.layers.0.self_attn.q_proj",
                            "role": "lora_A",
                            "shape": [2, 3],
                            "values": [1.0] * 6,
                        },
                        {
                            "module": "model.layers.0.self_attn.q_proj",
                            "role": "lora_B",
                            "shape": [4, 2],
                            "values": [0.0] * 8,
                        },
                    ]
                ),
                artifact.semantic_sha256,
            )
            config_path = root / "adapter_config.json"
            config_path.write_text(
                json.dumps(config)[:-1] + ',"r":2}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "duplicate JSON object key"):
                inspect_initial_adapter(root, lock, "model", "peft-qv", {})
            config_path.write_text(json.dumps(config), encoding="utf-8")
            checkpoint = root / "adapter_model.safetensors"
            raw = bytearray(checkpoint.read_bytes())
            raw[-4:] = struct.pack("<f", float("nan"))
            checkpoint.write_bytes(raw)
            with self.assertRaisesRegex(ContractError, "non-finite"):
                inspect_initial_adapter(root, lock, "model", "peft-qv", {})

    def test_file_identity_detects_in_run_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bound"
            path.write_bytes(b"before")
            identity = capture_file_identities((path,))
            require_files_unchanged(identity)
            path.write_bytes(b"after-with-different-size")
            with self.assertRaisesRegex(ContractError, "drifted"):
                require_files_unchanged(identity)

    def test_darwin_memory_parsers_and_measured_window_deltas(self) -> None:
        vm_output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 12.
Pageins: 100.
Pageouts: 4.
Swapins: 2.
Swapouts: 3.
"""
        page_size, counters = parse_vm_stat(vm_output)
        self.assertEqual(16384, page_size)
        self.assertEqual({"Pageins": 100, "Pageouts": 4, "Swapins": 2, "Swapouts": 3}, counters)
        self.assertEqual(
            63.0,
            parse_memory_pressure_available_percent(
                "System-wide memory free percentage: 63%\n"
            ),
        )
        before = DarwinSystemMemorySnapshot(16384, 100, 4, 2, 3, 63.0)
        after = DarwinSystemMemorySnapshot(16384, 102, 5, 2, 3, 61.5)
        self.assertEqual(
            {
                "swapins_bytes": 0,
                "swapouts_bytes": 0,
                "pageins_bytes": 32768,
                "pageouts_bytes": 16384,
                "pressure_available_percent_delta": -1.5,
            },
            darwin_system_memory_deltas(before, after),
        )
        with self.assertRaisesRegex(ContractError, "counter regressed"):
            darwin_system_memory_deltas(after, before)

    def test_process_memory_sampler_reports_current_footprint_peak_and_count(self) -> None:
        samples = iter((100, 250, 175))
        observed_pids: list[int] = []
        sampler = DarwinProcessMemorySampler(
            pid=321,
            interval_ms=10,
            probe=lambda pid: (observed_pids.append(pid), next(samples))[1],
        )
        sampler.start()
        measurement = sampler.stop()
        self.assertEqual(2, measurement.sample_count)
        self.assertEqual(250, measurement.peak_phys_footprint_bytes)
        self.assertEqual([321, 321], observed_pids)

    def test_parent_child_control_barriers_bracket_measured_window(self) -> None:
        import socket
        from unittest.mock import patch

        parent_socket, worker_socket = socket.socketpair()
        parent_channel = JsonControlChannel(parent_socket)
        worker_channel = JsonControlChannel(worker_socket)
        reporter = WorkerPhaseReporter(worker_channel)
        failures: list[BaseException] = []

        def worker_protocol() -> None:
            try:
                reporter.barrier("measured-optimizer-steps-start")
                reporter.barrier("measured-optimizer-steps-end")
                worker_channel.send(
                    {
                        "schema_version": CONTROL_PROTOCOL_VERSION,
                        "kind": "result",
                        "payload": {"worker": "payload"},
                    }
                )
                acknowledgement = worker_channel.receive()
                self.assertEqual("result", acknowledgement["phase"])
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=worker_protocol)
        thread.start()
        before = DarwinSystemMemorySnapshot(16384, 10, 5, 2, 3, 60.0)
        after = DarwinSystemMemorySnapshot(16384, 10, 5, 2, 3, 59.0)

        class FakeSampler:
            def stop(self) -> ProcessMemoryMeasurement:
                return ProcessMemoryMeasurement(1234, 9)

        lock = {
            "benchmark_contract": {
                "memory": {
                    "maximum_swapins_bytes": 0,
                    "maximum_swapouts_bytes": 0,
                    "maximum_pageins_bytes": 0,
                    "maximum_pageouts_bytes": 0,
                    "minimum_pressure_available_percent_delta": -5.0,
                }
            }
        }
        with patch(
            "run_gemma4_lora_mlx_benchmark.capture_darwin_system_memory_snapshot",
            side_effect=(before, after),
        ) as snapshot:
            payload, process, deltas = _coordinator_result(
                parent_channel,
                FakeSampler(),  # type: ignore[arg-type]
                lock,
            )
        thread.join(timeout=2)
        parent_socket.close()
        worker_socket.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual([], failures)
        self.assertEqual({"worker": "payload"}, payload)
        self.assertEqual(ProcessMemoryMeasurement(1234, 9), process)
        self.assertEqual(-1.0, deltas["pressure_available_percent_delta"])
        self.assertEqual([unittest.mock.call(before_measured=True), unittest.mock.call(before_measured=False)], snapshot.call_args_list)

    def test_control_channel_fails_closed_on_eof_and_duplicate_fields(self) -> None:
        import socket

        receiver_socket, sender_socket = socket.socketpair()
        receiver = JsonControlChannel(receiver_socket)
        sender_socket.sendall(
            ('{"schema_version":"' + CONTROL_PROTOCOL_VERSION + '",').encode("utf-8")
            +
            b'"kind":"phase","kind":"result"}\n'
        )
        with self.assertRaisesRegex(ContractError, "duplicate JSON object key"):
            receiver.receive()
        sender_socket.close()
        with self.assertRaisesRegex(ContractError, "closed the control channel"):
            receiver.receive()
        receiver_socket.close()

    def test_control_channel_has_a_finite_idle_deadline(self) -> None:
        import socket

        receiver_socket, sender_socket = socket.socketpair()
        receiver = JsonControlChannel(receiver_socket, receive_timeout_seconds=0.01)
        with self.assertRaisesRegex(ContractError, "idle for 0.01 seconds"):
            receiver.receive()
        receiver_socket.close()
        sender_socket.close()

    def test_target_selection_requires_exact_locked_inventory(self) -> None:
        lock = {
            "target_presets": {"peft-qv": ["q_proj", "v_proj"]},
            "target_inventory": {
                "model": {"peft-qv": {"q_proj": 1, "v_proj": 1}}
            },
        }
        model = FakeModel(
            [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.self_attn.k_proj",
            ]
        )
        self.assertEqual(
            [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.v_proj",
            ],
            target_module_names(model, lock, "model", "peft-qv"),
        )
        with self.assertRaisesRegex(ContractError, "incomplete target inventory"):
            target_module_names(
                FakeModel(["model.layers.0.self_attn.q_proj"]),
                lock,
                "model",
                "peft-qv",
            )

    def test_checkpoint_coverage_admits_only_exact_unused_shared_kv_weights(self) -> None:
        config = {
            "text_config": {
                "num_hidden_layers": 3,
                "num_kv_shared_layers": 1,
                "layer_types": [
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                ],
                "attention_k_eq_v": False,
            }
        }
        used = {
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.1.self_attn.q_proj.weight",
            "language_model.model.layers.2.self_attn.q_proj.weight",
        }
        ignored = expected_unused_shared_kv_parameter_names(config)
        self.assertEqual(3, len(ignored))
        self.assertEqual(
            ignored,
            validate_mlx_gemma4_checkpoint_coverage(config, used, used | ignored),
        )
        self.assertEqual(
            set(),
            validate_mlx_gemma4_checkpoint_coverage(config, used, used),
        )
        with self.assertRaisesRegex(ContractError, "missing MLX model"):
            validate_mlx_gemma4_checkpoint_coverage(config, used, ignored)
        with self.assertRaisesRegex(ContractError, "surplus differs"):
            validate_mlx_gemma4_checkpoint_coverage(
                config, used, used | ignored | {"unexpected.weight"}
            )

    def test_native_runtime_attestation_binds_closed_loaded_artifact_inventory(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "mlx-source"
            attestation_path = source_root / "build" / "native-attestation.json"
            attestation_path.parent.mkdir(parents=True)
            package_root = root / "venv" / "mlx"
            (package_root / "lib").mkdir(parents=True)
            core = package_root / "core.cpython-312-darwin.so"
            jaccl = package_root / "lib" / "libjaccl.dylib"
            metallib = package_root / "lib" / "mlx.metallib"
            dylib = package_root / "lib" / "libmlx.dylib"
            core.write_bytes(b"core")
            jaccl.write_bytes(b"jaccl")
            metallib.write_bytes(b"metal")
            dylib.write_bytes(b"dylib")

            def digest(path: Path) -> str:
                return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

            artifacts = [
                {
                    "role": "jaccl-runtime-dylib",
                    "relative_path": "lib/libjaccl.dylib",
                    "size_bytes": jaccl.stat().st_size,
                    "sha256": digest(jaccl),
                },
                {
                    "role": "metal-library",
                    "relative_path": "lib/mlx.metallib",
                    "size_bytes": metallib.stat().st_size,
                    "sha256": digest(metallib),
                },
                {
                    "role": "python-extension",
                    "relative_path": core.name,
                    "size_bytes": core.stat().st_size,
                    "sha256": digest(core),
                },
                {
                    "role": "runtime-dylib",
                    "relative_path": "lib/libmlx.dylib",
                    "size_bytes": dylib.stat().st_size,
                    "sha256": digest(dylib),
                },
            ]
            inventory_sha = canonical_mlx_native_artifact_inventory_sha256(artifacts)
            revision = "a" * 40
            precision_sha = "sha256:" + "b" * 64
            attestation_path.write_text(
                json.dumps(
                    {
                        "schema_version": "antfly_mlx_native_build_attestation/v1",
                        "source_revision": revision,
                        "source_clean": True,
                        "native_artifact_inventory_sha256": inventory_sha,
                        "build_command_sha256": "sha256:" + "c" * 64,
                        "precision_policy_sha256": precision_sha,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(mlx_build_attestation=attestation_path)
            lock = {
                "mlx_reference": {
                    "native_runtime": {
                        "extension_module": "mlx.core",
                        "artifact_roles": [
                            "jaccl-runtime-dylib",
                            "metal-library",
                            "python-extension",
                            "runtime-dylib",
                        ],
                        "build_attestation_schema_version": "antfly_mlx_native_build_attestation/v1",
                        "precision_policy_sha256": precision_sha,
                    }
                }
            }
            mx = argparse.Namespace(__name__="mlx.core")
            bundle = {
                "native_artifact_inventory_sha256": inventory_sha,
                "attestation_sha256": digest(attestation_path),
                "bound_paths": [str(attestation_path)],
            }
            with patch(
                "run_gemma4_lora_mlx_benchmark.inspect.getfile", return_value=str(core)
            ), patch(
                "run_gemma4_lora_mlx_benchmark.loaded_dyld_image_paths",
                return_value=(core.resolve(), jaccl.resolve(), dylib.resolve()),
            ) as dyld_images:
                verified = verify_mlx_native_runtime(
                    args,
                    lock,
                    {"path": str(source_root), "revision": revision},
                    mx,
                    bundle,
                )
                self.assertEqual(inventory_sha, verified["native_artifact_inventory"]["sha256"])
                self.assertEqual(
                    str(core.resolve()),
                    verified["native_artifact_inventory"]["loaded_core_path"],
                )
                (package_root / "lib" / "unbound.dylib").write_bytes(b"extra")
                with self.assertRaisesRegex(ContractError, "closed four-artifact inventory"):
                    verify_mlx_native_runtime(
                        args,
                        lock,
                        {"path": str(source_root), "revision": revision},
                        mx,
                        bundle,
                    )

                (package_root / "lib" / "unbound.dylib").unlink()

                stale_bundle = dict(bundle)
                stale_bundle["attestation_sha256"] = "sha256:" + "0" * 64
                with self.assertRaisesRegex(ContractError, "drifted after pre-import"):
                    verify_mlx_native_runtime(
                        args,
                        lock,
                        {"path": str(source_root), "revision": revision},
                        mx,
                        stale_bundle,
                    )
                unbound_loaded_dylib = root / "unbound" / "libmlx.dylib"
                unbound_loaded_dylib.parent.mkdir()
                unbound_loaded_dylib.write_bytes(b"unbound")
                dyld_images.return_value = (
                    core.resolve(),
                    unbound_loaded_dylib.resolve(),
                )
                with self.assertRaisesRegex(ContractError, "loaded MLX Mach-O images"):
                    verify_mlx_native_runtime(
                        args,
                        lock,
                        {"path": str(source_root), "revision": revision},
                        mx,
                        bundle,
                    )

    def test_dyld_image_paths_admit_sealed_system_cache_but_resolve_real_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "libmlx.dylib"
            real.write_bytes(b"mlx")
            self.assertEqual(real.resolve(), canonicalize_loaded_dyld_image_path(str(real), 1))
        cached = Path("/System/Library/Frameworks/NotARealFramework.framework/NotARealFramework")
        self.assertEqual(cached, canonicalize_loaded_dyld_image_path(str(cached), 2))
        with self.assertRaisesRegex(ContractError, "not absolute"):
            canonicalize_loaded_dyld_image_path("relative/libmlx.dylib", 3)

    def test_preimport_build_admission_propagates_missing_forged_and_stale_receipts(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            mlx = root / "mlx"
            mlx_lm = root / "mlx-lm"
            mlx.mkdir()
            mlx_lm.mkdir()
            args = argparse.Namespace(mlx_build_attestation=mlx / "build" / "antfly-native-build.json")
            checkout = {"path": str(mlx), "revision": "a" * 40}
            mlx_lm_checkout = {"path": str(mlx_lm), "revision": "b" * 40}
            for failure in (
                "missing sibling receipt",
                "forged receipt digest",
                "stale receipt inputs",
            ):
                with self.subTest(bundle_failure=failure), patch(
                    "build_and_attest_gemma4_mlx.ignored_untracked_files",
                    return_value=(),
                ), patch(
                    "build_and_attest_gemma4_mlx.verify_attestation_bundle",
                    side_effect=ContractError(failure),
                ):
                    with self.assertRaisesRegex(ContractError, failure):
                        verify_mlx_native_build_before_import(
                            args,
                            {},
                            checkout,
                            mlx_lm_checkout,
                        )

    def test_environment_admits_bundle_before_any_mlx_import(self) -> None:
        import builtins
        import types
        from unittest.mock import patch

        events: list[str] = []
        real_import = builtins.__import__
        mlx_package = types.ModuleType("mlx")
        mlx_package.__path__ = []
        mlx_core = types.ModuleType("mlx.core")
        mlx_utils = types.ModuleType("mlx.utils")
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_package.core = mlx_core
        mlx_package.utils = mlx_utils
        modules = {
            "mlx": mlx_package,
            "mlx.core": mlx_core,
            "mlx.utils": mlx_utils,
            "mlx_lm": mlx_lm,
        }

        def recording_import(name, *args, **kwargs):
            if name == "mlx_lm" or name.startswith("mlx."):
                events.append(f"import:{name}")
            return real_import(name, *args, **kwargs)

        def admit(*_args, **_kwargs):
            events.append("bundle")
            return {"native_artifact_inventory_sha256": "sha256:" + "1" * 64}

        root = Path(tempfile.gettempdir()).resolve()
        lock = {
            "mlx_reference": {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "source_revisions": {"mlx": "a" * 40, "mlx-lm": "b" * 40},
            }
        }
        args = argparse.Namespace(
            mlx_source=root / "mlx",
            mlx_lm_source=root / "mlx-lm",
            mlx_build_attestation=root / "attestation.json",
        )
        original_path = list(sys.path)
        original_dont_write_bytecode = sys.dont_write_bytecode
        try:
            with patch.dict(sys.modules, modules), patch(
                "run_gemma4_lora_mlx_benchmark.verify_source_checkout",
                side_effect=(
                    {"path": str(root / "mlx"), "revision": "a" * 40},
                    {"path": str(root / "mlx-lm"), "revision": "b" * 40},
                ),
            ), patch(
                "run_gemma4_lora_mlx_benchmark.verify_requirements_match_lock"
            ), patch(
                "run_gemma4_lora_mlx_benchmark.verify_packages",
                return_value={"mlx": "0.31.2", "mlx-lm": "0.31.3"},
            ), patch(
                "run_gemma4_lora_mlx_benchmark.verify_mlx_native_build_before_import",
                side_effect=admit,
            ), patch(
                "run_gemma4_lora_mlx_benchmark.verify_import_source"
            ), patch(
                "run_gemma4_lora_mlx_benchmark.verify_mlx_native_runtime",
                return_value={"admitted": True},
            ), patch("builtins.__import__", side_effect=recording_import):
                environment = verify_mlx_environment(args, lock)
            self.assertTrue(environment["admitted"])
            self.assertEqual("bundle", events[0])
            self.assertTrue(all(event.startswith("import:") for event in events[1:]))
            self.assertTrue(sys.dont_write_bytecode)
        finally:
            sys.path[:] = original_path
            sys.dont_write_bytecode = original_dont_write_bytecode

    def test_sample_assembly_conforms_to_benchmark_contract(self) -> None:
        lock = load_lock(LOCK_PATH)
        labels = [-100] * 96 + [5] * 32
        workload = build_workload(
            {"input_ids": [5] * 128, "labels": labels},
            sequence_length=128,
            grad_accum=1,
        )
        model_lock = lock["models"]["gemma-4-E2B-it"]
        prepared_payload = {
            "schema_version": "gemma4_prepared/v6",
            "artifact_sha256": "sha256:" + "1" * 64,
            "example_index": 0,
            "source_dataset_sha256": "2" * 64,
            "source_record_sha256": "3" * 64,
            "rendered_chat_sha256": "4" * 64,
        }
        preflight = Preflight(
            lock=lock,
            model={
                "local_artifact_sha256": "sha256:" + "d" * 64,
                "directory": "/models/test",
            },
            prepared_summary={},
            prepared=prepared_payload,
            adapter=AdapterArtifact(
                directory=Path("/adapters/test"),
                checkpoint=Path("/adapters/test/adapter_model.safetensors"),
                checkpoint_sha256="sha256:" + "a" * 64,
                config_sha256="sha256:" + "b" * 64,
                semantic_sha256="sha256:" + "c" * 64,
                semantics={},
                tensors={
                    (module, role): AdapterTensor(
                        source_name=f"{module}.{role}.weight",
                        shape=(16, 16),
                        data_sha256="sha256:" + "e" * 64,
                        data_offset_start=0,
                        data_offset_end=1024,
                    )
                    for module in (
                        *(
                            f"model.layers.{index}.self_attn.q_proj"
                            for index in range(35)
                        ),
                        *(
                            f"model.layers.{index}.self_attn.v_proj"
                            for index in range(15)
                        ),
                    )
                    for role in ("lora_A", "lora_B")
                },
                bound_files=(),
            ),
            workload=workload,
            bound_files=(),
        )
        args = argparse.Namespace(
            model_key="gemma-4-E2B-it",
            target_preset="peft-qv",
            sequence_length=128,
            grad_accum=1,
            campaign_id="campaign-1",
            run_id="mlx-0",
            repetition=0,
            sequence_index=0,
        )
        environment = {
            "versions": dict(lock["mlx_reference"]["packages"]),
            "mlx_checkout": {
                "revision": lock["mlx_reference"]["source_revisions"]["mlx"]
            },
            "mlx_lm_checkout": {
                "revision": lock["mlx_reference"]["source_revisions"]["mlx-lm"]
            },
        }
        native_artifacts = [
            {
                "role": "jaccl-runtime-dylib",
                "relative_path": "lib/libjaccl.dylib",
                "size_bytes": 1,
                "sha256": "sha256:" + "4" * 64,
            },
            {
                "role": "metal-library",
                "relative_path": "lib/mlx.metallib",
                "size_bytes": 1,
                "sha256": "sha256:" + "5" * 64,
            },
            {
                "role": "python-extension",
                "relative_path": "core.cpython-312-darwin.so",
                "size_bytes": 2,
                "sha256": "sha256:" + "6" * 64,
            },
            {
                "role": "runtime-dylib",
                "relative_path": "lib/libmlx.dylib",
                "size_bytes": 3,
                "sha256": "sha256:" + "7" * 64,
            },
        ]
        native_inventory_sha = canonical_mlx_native_artifact_inventory_sha256(native_artifacts)
        environment["native_artifact_inventory"] = {
            "schema_version": "antfly_mlx_native_artifact_inventory/v2",
            "sha256": native_inventory_sha,
            "loaded_core_path": "/venv/mlx/core.cpython-312-darwin.so",
            "artifacts": native_artifacts,
        }
        environment["build_attestation"] = {
            "schema_version": "antfly_mlx_native_build_attestation/v1",
            "path": "/src/mlx/build/native-attestation.json",
            "sha256": "sha256:" + "8" * 64,
            "source_revision": lock["mlx_reference"]["source_revisions"]["mlx"],
            "source_clean": True,
            "native_artifact_inventory_sha256": native_inventory_sha,
            "build_command_sha256": "sha256:" + "9" * 64,
            "precision_policy_sha256": lock["mlx_reference"]["native_runtime"]["precision_policy_sha256"],
        }
        metrics = {
            "load_seconds": 1.0,
            "cold_compile_and_step_seconds": 2.0,
            "first_steady_step_seconds": 1.5,
            "step_seconds": [1.0] * lock["mlx_reference"]["measured_steps"],
            "input_tokens": workload.input_tokens,
            "supervised_tokens": workload.supervised_tokens,
            "memory": {
                "process_peak_phys_footprint_bytes": 20_000,
                "sampler_interval_ms": 10,
                "sampler_sample_count": 200,
                "framework_allocator_peak_bytes": 10_000,
                "framework_allocator_peak_source": "mlx-metal-get-peak-memory",
                "system_deltas": {
                    "swapins_bytes": 0,
                    "swapouts_bytes": 0,
                    "pageins_bytes": 0,
                    "pageouts_bytes": 0,
                    "pressure_available_percent_delta": 0.0,
                },
            },
        }
        hardware = {
            "platform": "Darwin",
            "machine": "arm64",
            "chip": "Apple M4 Max",
            "memory_bytes": 64 * 1024**3,
            "os_version": "15.6",
            "os_build": "24G84",
        }
        producer_source = self.producer_source()
        # The evidence contract deliberately records the real interpreter, but
        # this unit test must remain runnable by contributors whose host Python
        # differs from the pinned MLX benchmark interpreter.
        with mock.patch(
            "run_gemma4_lora_mlx_benchmark.platform.python_version",
            return_value=f"{lock['mlx_reference']['python']}.0",
        ):
            payload = build_sample_payload(
                args,
                preflight,
                environment,
                metrics,
                hardware,
                ("--output", "/evidence/sample.json"),
                producer_source,
            )
            diagnostic_source = {
                **producer_source,
                "schema_version": DIAGNOSTIC_SOURCE_SCHEMA_VERSION,
            }
            diagnostic = build_diagnostic_payload(
                args,
                preflight,
                environment,
                metrics,
                hardware,
                ("--diagnostic-only", "--output", "/diagnostics/sample.json"),
                diagnostic_source,
            )
        diagnostic["precision_observations"] = self.precision_observations(diagnostic)
        self.assertEqual(
            DIAGNOSTIC_SAMPLE_SCHEMA_VERSION,
            validate_diagnostic_payload(diagnostic)["schema_version"],
        )
        self.assertFalse(diagnostic["diagnostic"]["release_eligible"])
        with tempfile.TemporaryDirectory() as tmp:
            diagnostic_path = Path(tmp) / "diagnostic.json"
            atomic_publish_json(diagnostic_path, diagnostic)
            with self.assertRaisesRegex(ContractError, "schema|field contract"):
                validate_sample(diagnostic_path, lock, LOCK_PATH)
        self.assertEqual(model_lock["revision"], payload["case"]["revision"])
        evidence = build_precision_evidence(
            payload,
            self.precision_observations(payload),
            producer_source,
            lock,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            evidence_path = precision_evidence_output_path(path)
            atomic_publish_json(evidence_path, evidence)
            payload["precision_evidence"] = precision_evidence_reference(path, evidence)
            atomic_publish_json(
                path,
                payload,
                validator=lambda temporary: validate_sample(
                    temporary,
                    lock,
                    LOCK_PATH,
                    precision_evidence_path=evidence_path,
                ),
            )
            validated = validate_sample(path, lock, LOCK_PATH)
        self.assertEqual("mlx-lm", validated.payload["framework"])


if __name__ == "__main__":
    unittest.main()
