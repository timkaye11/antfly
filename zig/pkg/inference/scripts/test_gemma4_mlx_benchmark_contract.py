from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from bench_gemma4_lora_mlx_zig import (
    MLX_RUNNER_RELATIVE_PATH,
    PRECISION_EVIDENCE_REFERENCE_SCHEMA_VERSION,
    PRECISION_EVIDENCE_SCHEMA_VERSION,
    SYNC_POINT,
    TIMED_UNIT,
    ZIG_COMMAND_PLAN_EVIDENCE_FIELDS,
    ZIG_EXECUTION_EVIDENCE_SCHEMA_VERSION,
    ZIG_PHASE_EVIDENCE_FIELDS,
    canonical_initial_adapter_sha256,
    canonical_moment_inventory_sha256,
    canonical_precision_sample_binding_sha256,
    canonical_target_inventory_sha256,
    canonical_tensor_inventory_sha256,
    benchmark_workload_sha256,
    compare_campaign,
    expected_semantic_contract,
    validate_sample,
)
from gemma4_oracle_contract import (
    BENCHMARK_PRODUCER_RELATIVE_PATHS,
    BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION,
    BENCHMARK_SAMPLE_SCHEMA_VERSION,
    ContractError,
    LOCK_PATH,
    canonical_benchmark_producer_source_sha256,
    canonical_mlx_native_artifact_inventory_sha256,
    load_lock,
    lock_digest,
    prefixed_sha256,
)
from run_gemma4_lora_benchmark_campaign import (
    MANIFEST_SCHEMA_VERSION,
    ORCHESTRATOR_RELATIVE_PATH,
    ORCHESTRATOR_SCHEMA_VERSION,
    SCRIPT_PATH as CAMPAIGN_ORCHESTRATOR_PATH,
    canonical_argv_sha256,
)


class Gemma4MlxBenchmarkContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = load_lock(LOCK_PATH)

    @staticmethod
    def zig_execution_evidence(grad_accum: int = 1) -> dict:
        steps = []
        for index in range(25):
            phase = {field: 0 for field in ZIG_PHASE_EVIDENCE_FIELDS}
            command = {field: 0 for field in ZIG_COMMAND_PLAN_EVIDENCE_FIELDS}
            if index == 0:
                phase["compile_ns"] = 1
                phase["graph_executor_plan_build_ns"] = 1
                phase["graph_executor_buffer_plan_build_ns"] = 1
            command.update({
                "graph_executor_partitions": grad_accum,
                "graph_executor_command_dispatches": 3 * grad_accum,
                "graph_executor_planned_dispatches": 2 * grad_accum,
                "graph_executor_plan_cache_hits": grad_accum - (1 if index == 0 else 0),
                "graph_executor_plan_cache_misses": 1 if index == 0 else 0,
                "metal_command_dot_general_dispatches": grad_accum,
                "metal_command_elementwise_dispatches": grad_accum,
                "metal_command_other_dispatches": grad_accum,
            })
            steps.append({
                "index": index,
                "phase": "cold" if index == 0 else "first" if index == 1 else "warmup" if index < 5 else "measured",
                "phase_evidence": phase,
                "command_plan_evidence": command,
            })
        return {
            "schema_version": ZIG_EXECUTION_EVIDENCE_SCHEMA_VERSION,
            "optimizer_steps": steps,
        }

    @staticmethod
    def producer_source(framework: str) -> dict:
        entrypoint = (
            MLX_RUNNER_RELATIVE_PATH
            if framework == "mlx-lm"
            else "zig/pkg/inference/scripts/run_antfly_gemma4_lora_benchmark.py"
        )
        files = [
            {
                "relative_path": relative_path,
                "source_sha256": (
                    prefixed_sha256(CAMPAIGN_ORCHESTRATOR_PATH)
                    if relative_path == ORCHESTRATOR_RELATIVE_PATH
                    else "sha256:" + hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
                ),
            }
            for relative_path in BENCHMARK_PRODUCER_RELATIVE_PATHS
        ]
        entrypoint_sha256 = next(
            item["source_sha256"]
            for item in files
            if item["relative_path"] == entrypoint
        )
        source_revision = "b" * 40
        source_tree = "d" * 40
        return {
            "schema_version": BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION,
            "relative_path": entrypoint,
            "source_revision": source_revision,
            "source_tree": source_tree,
            "source_clean": True,
            "source_sha256": entrypoint_sha256,
            "files": files,
            "manifest_sha256": canonical_benchmark_producer_source_sha256(
                relative_path=entrypoint,
                source_revision=source_revision,
                source_tree=source_tree,
                files=files,
            ),
        }

    def precision_evidence(self, payload: dict) -> dict:
        modules = payload["case"]["target_inventory"]["canonical_modules"]
        lora_tensors = sorted(
            (
                {"name": f"{module}.{suffix}", "dtype": "float32", "shape": [16, 16]}
                for module in modules
                for suffix in ("lora_a", "lora_b")
            ),
            key=lambda tensor: tensor["name"],
        )
        base_tensors = [{"name": "model.embed_tokens.weight", "dtype": "bfloat16", "shape": [2, 2]}]

        def inventory(tensors: list[dict], evidence_kind: str, dtype: str) -> dict:
            return {
                "evidence_kind": evidence_kind,
                "dtype": dtype,
                "tensor_count": len(tensors),
                "inventory_sha256": canonical_tensor_inventory_sha256(tensors),
                "tensors": tensors,
            }

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
        implementation = payload["implementation"]
        mlx = implementation["mlx"]
        attestation = mlx["build_attestation"]
        precision = self.lock["benchmark_contract"]["precision"]
        return {
            "schema_version": PRECISION_EVIDENCE_SCHEMA_VERSION,
            "framework": "mlx-lm",
            "oracle_lock_sha256": payload["oracle_lock_sha256"],
            "sample_binding": {
                "campaign_id": payload["campaign_id"],
                "run_id": payload["run_id"],
                "repetition": payload["repetition"],
                "sequence_index": payload["sequence_index"],
                "command_sha256": implementation["command_sha256"],
                "semantic_contract_sha256": payload["semantic_contract"]["sha256"],
                "sample_payload_sha256": canonical_precision_sample_binding_sha256(payload),
            },
            "runner": copy.deepcopy(implementation["producer_source"]),
            "native_runtime": {
                "mlx_source_revision": mlx["source_revision"],
                "mlx_lm_source_revision": implementation["mlx_lm"]["source_revision"],
                "native_artifact_inventory_sha256": mlx["native_artifact_inventory"]["sha256"],
                "build_attestation_sha256": attestation["sha256"],
                "build_command_sha256": attestation["build_command_sha256"],
                "precision_policy_sha256": attestation["precision_policy_sha256"],
            },
            "verified": dict(precision["verified"]),
            "not_asserted": list(precision["not_asserted"]),
            "comparison_policy": precision["comparison_policy"],
            "observations": {
                "base_model_storage": inventory(
                    base_tensors, "materialized-parameter-inventory", "bfloat16",
                ),
                "lora_parameter_storage": inventory(
                    lora_tensors, "materialized-trainable-parameter-inventory", "float32",
                ),
                "gradient_storage": {
                    "evidence_kind": "compiled-gradient-tree-inventory",
                    "dtype": "float32",
                    "tensor_count": len(lora_tensors),
                    "stages": [
                        {
                            "stage": stage,
                            "inventory_sha256": canonical_tensor_inventory_sha256(lora_tensors),
                            "tensors": lora_tensors,
                        }
                        for stage in ("raw", "accumulated", "clipped")
                    ],
                },
                "optimizer_moment_storage": {
                    "evidence_kind": "materialized-post-cold-adamw-moment-inventory",
                    "dtype": "float32",
                    "parameter_count": len(lora_tensors),
                    "moment_tensor_count": len(moments),
                    "inventory_sha256": canonical_moment_inventory_sha256(moments),
                    "moments": moments,
                },
                "loss": {
                    "evidence_kind": "evaluated-training-loss-graph",
                    "loss_tensor_dtype": "float32",
                    "reduction_input_dtype": "float32",
                },
            },
        }

    def write_sample(self, path: Path, payload: dict, *, evidence: dict | None = None) -> None:
        payload.pop("precision_evidence", None)
        if payload["framework"] == "mlx-lm":
            evidence = self.precision_evidence(payload) if evidence is None else evidence
            evidence_path = path.with_name(path.name + ".precision.json")
            encoded = (
                json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            ).encode("utf-8")
            evidence_path.write_bytes(encoded)
            payload["precision_evidence"] = {
                "schema_version": PRECISION_EVIDENCE_REFERENCE_SCHEMA_VERSION,
                "relative_path": evidence_path.name,
                "artifact_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
            }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def sample(self, framework: str, repetition: int, sequence_index: int) -> dict:
        model = self.lock["models"]["gemma-4-E2B-it"]
        input_ids = [1] * 128
        labels = [-100] * 96 + [1] * 32
        attention_mask = [1] * 128
        canonical_modules = sorted(
            [f"model.layers.{layer}.self_attn.q_proj" for layer in range(35)]
            + [f"model.layers.{layer}.self_attn.v_proj" for layer in range(15)]
        )
        if framework == "mlx-lm":
            native_artifacts = [
                {
                    "role": "jaccl-runtime-dylib",
                    "relative_path": "lib/libjaccl.dylib",
                    "size_bytes": 100,
                    "sha256": "sha256:" + "5" * 64,
                },
                {
                    "role": "metal-library",
                    "relative_path": "lib/mlx.metallib",
                    "size_bytes": 101,
                    "sha256": "sha256:" + "6" * 64,
                },
                {
                    "role": "python-extension",
                    "relative_path": "core.cpython-312-darwin.so",
                    "size_bytes": 102,
                    "sha256": "sha256:" + "7" * 64,
                },
                {
                    "role": "runtime-dylib",
                    "relative_path": "lib/libmlx.dylib",
                    "size_bytes": 103,
                    "sha256": "sha256:" + "8" * 64,
                },
            ]
            native_inventory_sha256 = canonical_mlx_native_artifact_inventory_sha256(native_artifacts)
            implementation = {
                "command_sha256": "sha256:"
                + hashlib.sha256(
                    f"{framework}:{repetition}:{sequence_index}".encode("utf-8")
                ).hexdigest(),
                "producer_source": self.producer_source(framework),
                "mlx": {
                    "version": self.lock["mlx_reference"]["packages"]["mlx"],
                    "source_revision": self.lock["mlx_reference"]["source_revisions"]["mlx"],
                    "source_clean": True,
                    "native_artifact_inventory": {
                        "schema_version": "antfly_mlx_native_artifact_inventory/v2",
                        "sha256": native_inventory_sha256,
                        "loaded_core_path": "/src/mlx/python/mlx/core.cpython-312-darwin.so",
                        "artifacts": native_artifacts,
                    },
                    "build_attestation": {
                        "schema_version": "antfly_mlx_native_build_attestation/v1",
                        "path": "/src/mlx/native-build-attestation.json",
                        "sha256": "sha256:" + "9" * 64,
                        "source_revision": self.lock["mlx_reference"]["source_revisions"]["mlx"],
                        "source_clean": True,
                        "native_artifact_inventory_sha256": native_inventory_sha256,
                        "build_command_sha256": "sha256:" + "a" * 64,
                        "precision_policy_sha256": self.lock["mlx_reference"]["native_runtime"]["precision_policy_sha256"],
                    },
                },
                "mlx_lm": {
                    "version": self.lock["mlx_reference"]["packages"]["mlx-lm"],
                    "source_revision": self.lock["mlx_reference"]["source_revisions"]["mlx-lm"],
                    "source_clean": True,
                },
                "python": {
                    "version": "3.12.9",
                    "executable": "/usr/bin/python3",
                    "executable_sha256": "sha256:" + "e" * 64,
                },
            }
            duration = 1.0
            peak_memory = 10_000
        else:
            implementation = {
                "command_sha256": "sha256:"
                + hashlib.sha256(
                    f"{framework}:{repetition}:{sequence_index}".encode("utf-8")
                ).hexdigest(),
                "producer_source": self.producer_source(framework),
                "antfly": {
                    "version": "antfly-test",
                    "source_revision": "b" * 40,
                    "source_clean": True,
                    "executable_sha256": "sha256:" + "f" * 64,
                },
            }
            duration = 0.9
            peak_memory = 10_500
        payload = {
            "schema_version": BENCHMARK_SAMPLE_SCHEMA_VERSION,
            "framework": framework,
            "oracle_lock_sha256": lock_digest(LOCK_PATH),
            "campaign_id": "campaign-1",
            "run_id": f"{framework}-{repetition}",
            "repetition": repetition,
            "sequence_index": sequence_index,
            "implementation": implementation,
            "process": {"pid": 1000 + sequence_index, "started_unix_ns": 1_000_000 + sequence_index},
            "hardware": {
                "platform": "Darwin",
                "machine": "arm64",
                "chip": "Apple M4 Max",
                "memory_bytes": 64 * 1024**3,
                "os_version": "15.6",
                "os_build": "24G84",
                "metal_device": "Apple M4 Max",
            },
            "case": {
                "model_key": "gemma-4-E2B-it",
                "revision": model["revision"],
                "local_artifact_sha256": "sha256:" + "d" * 64,
                "target_preset": "peft-qv",
                "rank": 16,
                "alpha": 32.0,
                "sequence_length": 128,
                "grad_accum": 1,
                "microbatch": 1,
                "prepared": {
                    "schema_version": "gemma4_prepared/v6",
                    "artifact_sha256": "sha256:" + "1" * 64,
                    "example_index": 0,
                    "source_dataset_sha256": "2" * 64,
                    "source_record_sha256": "3" * 64,
                    "rendered_chat_sha256": "4" * 64,
                    "workload_sha256": benchmark_workload_sha256([input_ids], [labels], [attention_mask]),
                },
                "initial_adapter": {
                    "schema_version": "antfly_gemma4_initial_adapter_semantics/v1",
                    "semantic_sha256": "sha256:" + "5" * 64,
                    "tensor_count": 2 * len(canonical_modules),
                    "tensor_dtype": "float32",
                },
                "target_inventory": {
                    "schema_version": "antfly_gemma4_target_inventory/v1",
                    "sha256": canonical_target_inventory_sha256(canonical_modules),
                    "module_count": len(canonical_modules),
                    "canonical_modules": canonical_modules,
                },
            },
            "protocol": {
                "fresh_process": True,
                "cold_optimizer_steps": 1,
                "cold_step_mutates_optimizer_state": True,
                "first_steady_steps": 1,
                "warmup_steps": 3,
                "measured_steps": 20,
                "explicit_device_sync": True,
                "sync_point": SYNC_POINT,
                "timed_unit": TIMED_UNIT,
            },
            "metrics": {
                "load_seconds": 2.0,
                "cold_compile_and_step_seconds": 3.0,
                "first_steady_step_seconds": 1.5,
                "step_seconds": [duration] * 20,
                "input_tokens": 128,
                "supervised_tokens": 32,
                "memory": {
                    "process_peak_phys_footprint_bytes": peak_memory,
                    "sampler_interval_ms": 10,
                    "sampler_sample_count": 100,
                    "framework_allocator_peak_bytes": 8_000 if framework == "mlx-lm" else None,
                    "framework_allocator_peak_source": (
                        "mlx-metal-get-peak-memory" if framework == "mlx-lm"
                        else "antfly-metal-allocator-unavailable"
                    ),
                    "system_deltas": {
                        "swapins_bytes": 0,
                        "swapouts_bytes": 0,
                        "pageins_bytes": 0,
                        "pageouts_bytes": 0,
                        "pressure_available_percent_delta": -0.5,
                    },
                },
            },
        }
        if framework == "antfly-zig-metal":
            payload["metrics"]["execution_evidence"] = self.zig_execution_evidence()
        payload["semantic_contract"] = expected_semantic_contract(self.lock, payload["case"])
        return payload

    def campaign(self, root: Path) -> list:
        root.mkdir(parents=True, exist_ok=True)
        samples = []
        sequence_index = 0
        for repetition in range(5):
            order = ("antfly-zig-metal", "mlx-lm") if repetition % 2 == 0 else ("mlx-lm", "antfly-zig-metal")
            for framework in order:
                path = root / f"{sequence_index:02d}-{framework}.json"
                self.write_sample(path, self.sample(framework, repetition, sequence_index))
                samples.append(validate_sample(path, self.lock, LOCK_PATH))
                sequence_index += 1
        return samples

    def write_campaign_manifest(self, root: Path, samples: list) -> Path:
        root = root.resolve()
        runs = []
        for sample in sorted(samples, key=lambda item: item.payload["sequence_index"]):
            payload = sample.payload
            sequence_index = payload["sequence_index"]
            framework = payload["framework"]
            runner_name = (
                "run_antfly_gemma4_lora_benchmark.py"
                if framework == "antfly-zig-metal"
                else "run_gemma4_lora_mlx_benchmark.py"
            )
            argv = [
                sys.executable,
                str(Path(__file__).resolve().parent / runner_name),
                "--fixture", framework,
                "--campaign-id", payload["campaign_id"],
                "--run-id", payload["run_id"],
                "--repetition", str(payload["repetition"]),
                "--sequence-index", str(sequence_index),
                "--output", str(sample.path),
            ]
            started = payload["process"]["started_unix_ns"]
            runs.append({
                "cell_id": "e2b-peft-qv-s128-ga1",
                "framework": framework,
                "repetition": payload["repetition"],
                "sequence_index": sequence_index,
                "run_id": payload["run_id"],
                "argv": argv,
                "argv_sha256": canonical_argv_sha256(argv),
                "started_unix_ns": started,
                "completed_unix_ns": started + 1,
                "sample": {
                    "relative_path": sample.path.relative_to(root).as_posix(),
                    "sha256": prefixed_sha256(sample.path),
                    "size_bytes": sample.path.stat().st_size,
                },
            })
        path = root / "COMPLETE.json"
        path.write_text(
            json.dumps({
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "status": "complete",
                "campaign_id": samples[0].payload["campaign_id"],
                "created_unix_ns": runs[0]["started_unix_ns"],
                "completed_unix_ns": runs[-1]["completed_unix_ns"],
                "plan": {
                    "sha256": "sha256:" + "1" * 64,
                    "cell_count": 1,
                    "repetitions_per_cell": 5,
                },
                "orchestrator": {
                    "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
                    "relative_path": ORCHESTRATOR_RELATIVE_PATH,
                    "source_sha256": prefixed_sha256(CAMPAIGN_ORCHESTRATOR_PATH),
                },
                "run_count": len(runs),
                "runs": runs,
            }) + "\n",
            encoding="utf-8",
        )
        return path

    def test_partial_same_mac_campaign_passes_with_five_fresh_paired_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = compare_campaign(self.campaign(Path(tmp)), self.lock, require_full_matrix=False)
        self.assertTrue(result["ok"], result["failures"])
        self.assertEqual(1, result["cell_count"])
        self.assertGreater(result["geomean_throughput_ratio"], 1.0)
        self.assertEqual(
            ["antfly-zig-metal", "mlx-lm", "antfly-zig-metal", "mlx-lm", "antfly-zig-metal"],
            result["cells"][0]["execution_order"],
        )

    def test_partial_comparison_verifies_a_supplied_campaign_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = self.campaign(root)
            manifest = self.write_campaign_manifest(root, samples)
            result = compare_campaign(
                samples,
                self.lock,
                require_full_matrix=False,
                campaign_manifest_path=manifest,
            )
            self.assertTrue(result["ok"], result["failures"])
            self.assertEqual(prefixed_sha256(manifest), result["campaign_manifest"]["sha256"])

    def test_command_invocation_digests_are_unique_but_not_campaign_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples = self.campaign(Path(tmp))
            identity = samples[-1].payload["implementation"]
            original_command = identity["command_sha256"]
            identity["command_sha256"] = samples[0].payload["implementation"]["command_sha256"]
            with self.assertRaisesRegex(ContractError, "unique command invocation digest"):
                compare_campaign(samples, self.lock, require_full_matrix=False)

            identity["command_sha256"] = original_command
            result = compare_campaign(samples, self.lock, require_full_matrix=False)
            self.assertTrue(result["ok"], result["failures"])
            self.assertEqual(
                {"antfly-zig-metal", "mlx-lm"},
                set(result["implementation_identities"]),
            )

    def test_campaign_implementation_identity_is_content_based_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples = self.campaign(Path(tmp))
            mlx = samples[-1].payload["implementation"]["mlx"]
            mlx["native_artifact_inventory"]["loaded_core_path"] = "/another/root/core.so"
            mlx["build_attestation"]["path"] = "/another/root/attestation.json"
            samples[-1].payload["implementation"]["python"]["executable"] = "/another/python"
            result = compare_campaign(samples, self.lock, require_full_matrix=False)
            self.assertTrue(result["ok"], result["failures"])

            mlx["build_attestation"]["sha256"] = "sha256:" + "0" * 64
            with self.assertRaisesRegex(ContractError, "changed within one campaign"):
                compare_campaign(samples, self.lock, require_full_matrix=False)

    def test_execution_contract_is_pair_cell_wide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples = self.campaign(Path(tmp))
            samples[-1].payload["protocol"]["warmup_steps"] += 1
            with self.assertRaisesRegex(ContractError, "execution contract changed"):
                compare_campaign(samples, self.lock, require_full_matrix=False)

    def test_producer_source_inventory_and_revision_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = self.sample("mlx-lm", 0, 0)
            incomplete["implementation"]["producer_source"]["files"].pop()
            path = root / "incomplete-source.json"
            self.write_sample(path, incomplete)
            with self.assertRaisesRegex(ContractError, "source inventory is incomplete"):
                validate_sample(path, self.lock, LOCK_PATH)

            unrelated = self.sample("antfly-zig-metal", 0, 1)
            unrelated["implementation"]["producer_source"]["source_revision"] = "e" * 40
            source = unrelated["implementation"]["producer_source"]
            source["manifest_sha256"] = canonical_benchmark_producer_source_sha256(
                relative_path=source["relative_path"],
                source_revision=source["source_revision"],
                source_tree=source["source_tree"],
                files=source["files"],
            )
            path = root / "unrelated-source.json"
            self.write_sample(path, unrelated)
            with self.assertRaisesRegex(ContractError, "wrapper source revision"):
                validate_sample(path, self.lock, LOCK_PATH)

    def test_hardware_mismatch_and_reused_process_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples = self.campaign(Path(tmp))
            changed = copy.deepcopy(samples[-1].payload)
            changed["hardware"]["os_build"] = "different"
            path = Path(tmp) / "changed.json"
            self.write_sample(path, changed)
            samples[-1] = validate_sample(path, self.lock, LOCK_PATH)
            with self.assertRaisesRegex(ContractError, "hardware identity"):
                compare_campaign(samples, self.lock, require_full_matrix=False)

            samples = self.campaign(Path(tmp) / "second")
            samples[-1].payload["process"] = dict(samples[0].payload["process"])
            with self.assertRaisesRegex(ContractError, "process identity"):
                compare_campaign(samples, self.lock, require_full_matrix=False)

    def test_mlx_revision_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.sample("mlx-lm", 0, 0)
            payload["implementation"]["mlx_lm"]["source_revision"] = "0" * 40
            path = Path(tmp) / "sample.json"
            self.write_sample(path, payload)
            with self.assertRaisesRegex(ContractError, "not pinned"):
                validate_sample(path, self.lock, LOCK_PATH)

    def test_mlx_stack_and_python_provenance_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirty = self.sample("mlx-lm", 0, 0)
            dirty["implementation"]["mlx"]["source_clean"] = False
            path = root / "dirty.json"
            self.write_sample(path, dirty)
            with self.assertRaisesRegex(ContractError, "mlx source checkout must be clean"):
                validate_sample(path, self.lock, LOCK_PATH)

            wrong_python = self.sample("mlx-lm", 0, 0)
            wrong_python["implementation"]["python"]["version"] = "3.11.9"
            path = root / "python.json"
            self.write_sample(path, wrong_python)
            with self.assertRaisesRegex(ContractError, "Python major/minor is not pinned"):
                validate_sample(path, self.lock, LOCK_PATH)

    def test_mlx_native_artifact_inventory_and_build_attestation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drifted = self.sample("mlx-lm", 0, 0)
            drifted["implementation"]["mlx"]["native_artifact_inventory"]["artifacts"][0]["sha256"] = (
                "sha256:" + "0" * 64
            )
            path = root / "drifted-native.json"
            self.write_sample(path, drifted)
            with self.assertRaisesRegex(ContractError, "inventory digest mismatch"):
                validate_sample(path, self.lock, LOCK_PATH)

            stale_attestation = self.sample("mlx-lm", 0, 0)
            stale_attestation["implementation"]["mlx"]["build_attestation"][
                "native_artifact_inventory_sha256"
            ] = "sha256:" + "0" * 64
            path = root / "stale-attestation.json"
            self.write_sample(path, stale_attestation)
            with self.assertRaisesRegex(ContractError, "does not bind the native artifact inventory"):
                validate_sample(path, self.lock, LOCK_PATH)

            incomplete = self.sample("mlx-lm", 0, 0)
            incomplete["implementation"]["mlx"]["native_artifact_inventory"]["artifacts"].pop()
            path = root / "incomplete-native.json"
            self.write_sample(path, incomplete)
            with self.assertRaisesRegex(ContractError, "exact sorted Metal runtime roles"):
                validate_sample(path, self.lock, LOCK_PATH)

    def test_release_campaign_rejects_unproven_precision_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = self.campaign(root)
            with self.assertRaisesRegex(ContractError, "campaign manifest"):
                compare_campaign(samples, self.lock, require_full_matrix=True)
            manifest = self.write_campaign_manifest(root, samples)
            with self.assertRaisesRegex(ContractError, "runtime proof"):
                compare_campaign(
                    samples,
                    self.lock,
                    require_full_matrix=True,
                    campaign_manifest_path=manifest,
                )

    def test_mlx_precision_evidence_is_required_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self.sample("mlx-lm", 0, 0)
            missing = root / "missing.json"
            missing.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "precision_evidence"):
                validate_sample(missing, self.lock, LOCK_PATH)

            bound = root / "bound.json"
            self.write_sample(bound, payload)
            evidence_path = bound.with_name(bound.name + ".precision.json")
            evidence_path.write_bytes(evidence_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ContractError, "artifact digest mismatch"):
                validate_sample(bound, self.lock, LOCK_PATH)

            escaped_payload = self.sample("mlx-lm", 0, 1)
            escaped = root / "escaped.json"
            self.write_sample(escaped, escaped_payload)
            escaped_payload["precision_evidence"]["relative_path"] = "../escaped.precision.json"
            escaped.write_text(json.dumps(escaped_payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "one sibling file name"):
                validate_sample(escaped, self.lock, LOCK_PATH)

            tampered_payload = self.sample("mlx-lm", 0, 2)
            tampered = root / "tampered.json"
            self.write_sample(tampered, tampered_payload)
            tampered_payload["metrics"]["step_seconds"][0] = 9.0
            tampered.write_text(json.dumps(tampered_payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "sample binding differs"):
                validate_sample(tampered, self.lock, LOCK_PATH)

    def test_mlx_precision_evidence_rejects_runner_native_and_claim_drift(self) -> None:
        mutations = (
            (
                "runner-hash",
                lambda evidence: evidence["runner"].__setitem__("source_sha256", "sha256:" + "0" * 64),
                "entrypoint digest differs",
            ),
            (
                "native-attestation",
                lambda evidence: evidence["native_runtime"].__setitem__("build_attestation_sha256", "sha256:" + "0" * 64),
                "native runtime binding",
            ),
            (
                "promoted-activation",
                lambda evidence: evidence["verified"].__setitem__("activation_dtype", "bfloat16"),
                "verified precision fields",
            ),
            (
                "missing-gradient-stage",
                lambda evidence: evidence["observations"]["gradient_storage"]["stages"].pop(),
                "raw, accumulated, and clipped",
            ),
            (
                "missing-optimizer-moment",
                lambda evidence: evidence["observations"]["optimizer_moment_storage"]["moments"].pop(),
                "optimizer m/v inventory",
            ),
            (
                "false-loss-reduction",
                lambda evidence: evidence["observations"]["loss"].__setitem__("reduction_input_dtype", "bfloat16"),
                "loss tensor/reduction-input",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (name, mutate, message) in enumerate(mutations):
                with self.subTest(name=name):
                    payload = self.sample("mlx-lm", 0, index)
                    evidence = self.precision_evidence(payload)
                    mutate(evidence)
                    path = root / f"{name}.json"
                    self.write_sample(path, payload, evidence=evidence)
                    with self.assertRaisesRegex(ContractError, message):
                        validate_sample(path, self.lock, LOCK_PATH)

    def test_antfly_sample_cannot_carry_mlx_precision_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.sample("antfly-zig-metal", 0, 0)
            payload["precision_evidence"] = {
                "schema_version": PRECISION_EVIDENCE_REFERENCE_SCHEMA_VERSION,
                "relative_path": "sample.json.precision.json",
                "artifact_sha256": "sha256:" + "0" * 64,
            }
            path = Path(tmp) / "sample.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "unknown=precision_evidence"):
                validate_sample(path, self.lock, LOCK_PATH)

    def test_zig_execution_evidence_is_required_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = self.sample("antfly-zig-metal", 0, 0)
            missing["metrics"].pop("execution_evidence")
            path = root / "missing.json"
            path.write_text(json.dumps(missing), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "missing=execution_evidence"):
                validate_sample(path, self.lock, LOCK_PATH)

            cache_drift = self.sample("antfly-zig-metal", 0, 0)
            command = cache_drift["metrics"]["execution_evidence"]["optimizer_steps"][2]["command_plan_evidence"]
            command["graph_executor_plan_cache_hits"] = 0
            command["graph_executor_plan_cache_misses"] = 1
            path = root / "cache-drift.json"
            path.write_text(json.dumps(cache_drift), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "unexpected cache miss count"):
                validate_sample(path, self.lock, LOCK_PATH)

            missing_route_counter = self.sample("antfly-zig-metal", 0, 0)
            command = missing_route_counter["metrics"]["execution_evidence"]["optimizer_steps"][2]["command_plan_evidence"]
            command.pop("metal_gemma4_bf16_gate_up_fused_calls")
            path = root / "missing-route-counter.json"
            path.write_text(json.dumps(missing_route_counter), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "missing=metal_gemma4_bf16_gate_up_fused_calls"):
                validate_sample(path, self.lock, LOCK_PATH)

            missing_backward_route_counter = self.sample("antfly-zig-metal", 0, 0)
            command = missing_backward_route_counter["metrics"]["execution_evidence"]["optimizer_steps"][2]["command_plan_evidence"]
            command.pop("metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls")
            path = root / "missing-backward-route-counter.json"
            path.write_text(json.dumps(missing_backward_route_counter), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "missing=metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls"):
                validate_sample(path, self.lock, LOCK_PATH)

            legacy = self.sample("antfly-zig-metal", 0, 0)
            legacy["schema_version"] = "antfly_gemma4_lora_benchmark_sample/v4"
            path = root / "legacy.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "unsupported benchmark sample schema"):
                validate_sample(path, self.lock, LOCK_PATH)

    def test_mlx_sample_cannot_carry_zig_execution_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.sample("mlx-lm", 0, 0)
            payload["metrics"]["execution_evidence"] = self.zig_execution_evidence()
            path = Path(tmp) / "sample.json"
            evidence = self.precision_evidence(payload)
            self.write_sample(path, payload, evidence=evidence)
            with self.assertRaisesRegex(ContractError, "unknown=execution_evidence"):
                validate_sample(path, self.lock, LOCK_PATH)

    def test_initial_adapter_canonical_f32_digest_binds_values_and_roles(self) -> None:
        tensors = [
            {"module": "model.layers.0.self_attn.q_proj", "role": "lora_A", "shape": [1, 2], "values": [-0.5, 0.25]},
            {"module": "model.layers.0.self_attn.q_proj", "role": "lora_B", "shape": [2, 1], "values": [0.0, 0.0]},
        ]
        digest = canonical_initial_adapter_sha256(tensors)
        self.assertEqual(digest, canonical_initial_adapter_sha256(list(reversed(tensors))))
        changed = copy.deepcopy(tensors)
        changed[0]["values"][0] = -0.25
        self.assertNotEqual(digest, canonical_initial_adapter_sha256(changed))
        changed = copy.deepcopy(tensors)
        changed[0]["role"] = "lora_B"
        with self.assertRaisesRegex(ContractError, "unique LoRA A/B"):
            canonical_initial_adapter_sha256(changed)

    def test_target_inventory_and_semantic_contract_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrong_inventory = self.sample("antfly-zig-metal", 0, 0)
            wrong_inventory["case"]["target_inventory"]["sha256"] = "sha256:" + "0" * 64
            path = root / "inventory.json"
            self.write_sample(path, wrong_inventory)
            with self.assertRaisesRegex(ContractError, "digest does not bind"):
                validate_sample(path, self.lock, LOCK_PATH)

            wrong_precision = self.sample("antfly-zig-metal", 0, 0)
            wrong_precision["semantic_contract"]["precision"]["activation_dtype"] = "bfloat16"
            path = root / "precision.json"
            self.write_sample(path, wrong_precision)
            with self.assertRaisesRegex(ContractError, "semantic contract differs"):
                validate_sample(path, self.lock, LOCK_PATH)

    def test_cold_step_sync_and_memory_definition_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            no_mutation = self.sample("antfly-zig-metal", 0, 0)
            no_mutation["protocol"]["cold_step_mutates_optimizer_state"] = False
            path = root / "cold.json"
            self.write_sample(path, no_mutation)
            with self.assertRaisesRegex(ContractError, "protocol differs"):
                validate_sample(path, self.lock, LOCK_PATH)

            wrong_memory = self.sample("mlx-lm", 0, 0)
            wrong_memory["metrics"]["memory"]["sampler_interval_ms"] = 100
            path = root / "memory.json"
            self.write_sample(path, wrong_memory)
            with self.assertRaisesRegex(ContractError, "sampler interval"):
                validate_sample(path, self.lock, LOCK_PATH)

            paged = self.sample("mlx-lm", 0, 0)
            paged["metrics"]["memory"]["system_deltas"]["pageouts_bytes"] = 4096
            path = root / "pageout.json"
            self.write_sample(path, paged)
            with self.assertRaisesRegex(ContractError, "zero-I/O gate"):
                validate_sample(path, self.lock, LOCK_PATH)

            pressured = self.sample("mlx-lm", 0, 0)
            pressured["metrics"]["memory"]["system_deltas"]["pressure_available_percent_delta"] = -5.1
            path = root / "pressure.json"
            self.write_sample(path, pressured)
            with self.assertRaisesRegex(ContractError, "pressure drop"):
                validate_sample(path, self.lock, LOCK_PATH)

    def test_workload_digest_binds_ordered_model_inputs(self) -> None:
        masks = [[1, 1], [1, 1]]
        first = benchmark_workload_sha256([[1, 2], [3, 4]], [[-100, 2], [-100, 4]], masks)
        self.assertEqual(
            first,
            benchmark_workload_sha256([[1, 2], [3, 4]], [[-100, 2], [-100, 4]], masks),
        )
        self.assertNotEqual(
            first,
            benchmark_workload_sha256([[3, 4], [1, 2]], [[-100, 4], [-100, 2]], masks),
        )
        self.assertNotEqual(
            first,
            benchmark_workload_sha256([[1, 2], [3, 4]], [[-100, 2], [3, 4]], masks),
        )
        self.assertNotEqual(
            first,
            benchmark_workload_sha256([[1, 2], [3, 4]], [[-100, 2], [-100, 4]], [[1, 1], [1, 0]]),
        )
        with self.assertRaisesRegex(ContractError, "equal, non-empty"):
            benchmark_workload_sha256([[1]], [], [[1]])

    def test_matrix_cell_cannot_mix_prepared_workload_identities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = self.campaign(root)
            changed = copy.deepcopy(samples[-1].payload)
            changed["case"]["prepared"]["workload_sha256"] = "sha256:" + "9" * 64
            changed["semantic_contract"] = expected_semantic_contract(self.lock, changed["case"])
            path = root / "changed-workload.json"
            self.write_sample(path, changed)
            samples[-1] = validate_sample(path, self.lock, LOCK_PATH)
            with self.assertRaisesRegex(ContractError, "mix semantic case identities"):
                compare_campaign(samples, self.lock, require_full_matrix=False)

    def test_paired_frameworks_require_identical_initial_adapter_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = self.campaign(root)
            changed = copy.deepcopy(samples[-1].payload)
            changed["case"]["initial_adapter"]["semantic_sha256"] = "sha256:" + "8" * 64
            changed["semantic_contract"] = expected_semantic_contract(self.lock, changed["case"])
            path = root / "changed-adapter.json"
            self.write_sample(path, changed)
            samples[-1] = validate_sample(path, self.lock, LOCK_PATH)
            with self.assertRaisesRegex(ContractError, "mix semantic case identities"):
                compare_campaign(samples, self.lock, require_full_matrix=False)

    def test_paired_frameworks_require_exact_token_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = self.campaign(root)
            changed = copy.deepcopy(samples[-1].payload)
            changed["metrics"]["supervised_tokens"] -= 1
            path = root / "changed-token-count.json"
            self.write_sample(path, changed)
            samples[-1] = validate_sample(path, self.lock, LOCK_PATH)
            with self.assertRaisesRegex(ContractError, "paired input/supervised token counts differ"):
                compare_campaign(samples, self.lock, require_full_matrix=False)

    def test_input_tokens_cover_the_complete_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.sample("antfly-zig-metal", 0, 0)
            payload["metrics"]["input_tokens"] -= 1
            path = Path(tmp) / "short-step.json"
            self.write_sample(path, payload)
            with self.assertRaisesRegex(ContractError, "complete timed optimizer step"):
                validate_sample(path, self.lock, LOCK_PATH)

    def test_json_schemas_track_structural_validator_contract(self) -> None:
        script_dir = Path(__file__).parent
        sample_schema = json.loads(
            (script_dir / "gemma4_mlx_benchmark.schema.json").read_text(encoding="utf-8")
        )
        implementation = sample_schema["properties"]["implementation"]
        self.assertIn("producer_source", implementation["required"])
        self.assertEqual("#/$defs/producerSource", implementation["properties"]["producer_source"]["$ref"])

        producer = sample_schema["$defs"]["producerSource"]
        producer_fields = {
            "schema_version", "relative_path", "source_revision", "source_tree",
            "source_clean", "source_sha256", "files", "manifest_sha256",
        }
        self.assertFalse(producer["additionalProperties"])
        self.assertEqual(producer_fields, set(producer["required"]))
        self.assertEqual(producer_fields, set(producer["properties"]))
        schema_paths = [
            item["allOf"][1]["properties"]["relative_path"]["const"]
            for item in producer["properties"]["files"]["prefixItems"]
        ]
        self.assertEqual(list(BENCHMARK_PRODUCER_RELATIVE_PATHS), schema_paths)
        self.assertEqual(len(schema_paths), producer["properties"]["files"]["minItems"])
        self.assertEqual(len(schema_paths), producer["properties"]["files"]["maxItems"])

        framework_rule = sample_schema["allOf"][1]
        self.assertEqual(
            {"mlx", "mlx_lm", "python"},
            set(framework_rule["then"]["properties"]["implementation"]["required"]),
        )
        self.assertEqual(
            ["antfly"],
            framework_rule["then"]["properties"]["implementation"]["not"]["required"],
        )
        mlx_memory = framework_rule["then"]["properties"]["metrics"]["properties"]["memory"]["properties"]
        antfly_memory = framework_rule["else"]["properties"]["metrics"]["properties"]["memory"]["properties"]
        self.assertEqual(
            ["execution_evidence"],
            framework_rule["then"]["properties"]["metrics"]["not"]["required"],
        )
        self.assertEqual(
            ["execution_evidence"],
            framework_rule["else"]["properties"]["metrics"]["required"],
        )
        self.assertEqual("mlx-metal-get-peak-memory", mlx_memory["framework_allocator_peak_source"]["const"])
        self.assertIsNone(antfly_memory["framework_allocator_peak_bytes"]["const"])
        self.assertEqual(
            "antfly-metal-allocator-unavailable",
            antfly_memory["framework_allocator_peak_source"]["const"],
        )

        model_rule = sample_schema["allOf"][2]
        self.assertEqual(
            self.lock["models"]["gemma-4-E2B-it"]["revision"],
            model_rule["then"]["properties"]["case"]["properties"]["revision"]["const"],
        )
        self.assertEqual(
            self.lock["models"]["gemma-4-E4B-it"]["revision"],
            model_rule["else"]["properties"]["case"]["properties"]["revision"]["const"],
        )

        mlx_reference = self.lock["mlx_reference"]
        mlx_source = sample_schema["$defs"]["mlxNativeSource"]["properties"]
        mlx_lm_source = sample_schema["$defs"]["packageSource"]["properties"]
        self.assertEqual(mlx_reference["packages"]["mlx"], mlx_source["version"]["const"])
        self.assertEqual(mlx_reference["source_revisions"]["mlx"], mlx_source["source_revision"]["const"])
        self.assertEqual(mlx_reference["packages"]["mlx-lm"], mlx_lm_source["version"]["const"])
        self.assertEqual(mlx_reference["source_revisions"]["mlx-lm"], mlx_lm_source["source_revision"]["const"])
        build_attestation = mlx_source["build_attestation"]["properties"]
        self.assertEqual(mlx_reference["source_revisions"]["mlx"], build_attestation["source_revision"]["const"])
        self.assertEqual(
            mlx_reference["native_runtime"]["precision_policy_sha256"],
            build_attestation["precision_policy_sha256"]["const"],
        )
        self.assertIn("^3\\.12", sample_schema["$defs"]["pythonRuntime"]["properties"]["version"]["pattern"])
        execution = sample_schema["$defs"]["zigExecutionEvidence"]
        self.assertEqual("antfly_gemma4_zig_execution_evidence/v2", execution["properties"]["schema_version"]["const"])
        self.assertEqual(25, execution["properties"]["optimizer_steps"]["minItems"])
        self.assertEqual(
            set(ZIG_PHASE_EVIDENCE_FIELDS),
            set(sample_schema["$defs"]["zigPhaseEvidence"]["required"]),
        )
        self.assertEqual(
            set(ZIG_COMMAND_PLAN_EVIDENCE_FIELDS),
            set(sample_schema["$defs"]["zigCommandPlanEvidence"]["required"]),
        )

        system_deltas = sample_schema["properties"]["metrics"]["properties"]["memory"]["properties"]["system_deltas"]["properties"]
        memory_contract = self.lock["benchmark_contract"]["memory"]
        for field in ("swapins_bytes", "swapouts_bytes", "pageins_bytes", "pageouts_bytes"):
            self.assertEqual(memory_contract[f"maximum_{field}"], system_deltas[field]["const"])
        self.assertEqual(
            memory_contract["minimum_pressure_available_percent_delta"],
            system_deltas["pressure_available_percent_delta"]["minimum"],
        )

        semantic = sample_schema["properties"]["semantic_contract"]
        for field in ("optimizer", "determinism", "runtime", "memory"):
            definition_name = "benchmark" + field.title()
            self.assertEqual(f"#/$defs/{definition_name}", semantic["properties"][field]["$ref"])
            definition = sample_schema["$defs"][definition_name]
            expected = dict(self.lock["benchmark_contract"][field])
            if field == "optimizer":
                expected["gradient_accumulation_steps"] = None
            self.assertFalse(definition["additionalProperties"])
            self.assertEqual(set(expected), set(definition["required"]))
            self.assertEqual(set(expected), set(definition["properties"]))
            for key, value in expected.items():
                if key == "gradient_accumulation_steps":
                    self.assertEqual([1, 4], definition["properties"][key]["enum"])
                elif key == "betas":
                    self.assertEqual(
                        value,
                        [item["const"] for item in definition["properties"][key]["prefixItems"]],
                    )
                else:
                    self.assertEqual(value, definition["properties"][key]["const"])

        precision_schema = json.loads(
            (script_dir / "gemma4_mlx_precision_evidence.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual("#/$defs/producerSource", precision_schema["properties"]["runner"]["$ref"])
        precision_paths = [
            item["allOf"][1]["properties"]["relative_path"]["const"]
            for item in precision_schema["$defs"]["producerSource"]["properties"]["files"]["prefixItems"]
        ]
        self.assertEqual(list(BENCHMARK_PRODUCER_RELATIVE_PATHS), precision_paths)
        precision_native = precision_schema["properties"]["native_runtime"]["properties"]
        self.assertEqual(
            mlx_reference["source_revisions"]["mlx"],
            precision_native["mlx_source_revision"]["const"],
        )
        self.assertEqual(
            mlx_reference["source_revisions"]["mlx-lm"],
            precision_native["mlx_lm_source_revision"]["const"],
        )
        self.assertEqual(
            mlx_reference["native_runtime"]["precision_policy_sha256"],
            precision_native["precision_policy_sha256"]["const"],
        )
        base_dtype = precision_schema["properties"]["observations"]["properties"]["base_model_storage"]["allOf"][1]["properties"]["tensors"]["items"]["properties"]["dtype"]["const"]
        lora_dtype = precision_schema["properties"]["observations"]["properties"]["lora_parameter_storage"]["allOf"][1]["properties"]["tensors"]["items"]["properties"]["dtype"]["const"]
        gradient_dtype = precision_schema["$defs"]["gradientStage"]["properties"]["tensors"]["items"]["allOf"][1]["properties"]["dtype"]["const"]
        self.assertEqual(("bfloat16", "float32", "float32"), (base_dtype, lora_dtype, gradient_dtype))

        publication_schema = json.loads(
            (script_dir / "gemma4_oracle_publication.schema.json").read_text(encoding="utf-8")
        )
        self.assertTrue(publication_schema["properties"]["files"]["uniqueItems"])


if __name__ == "__main__":
    unittest.main()
