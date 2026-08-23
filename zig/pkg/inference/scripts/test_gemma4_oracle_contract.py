from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import gemma4_oracle_contract as contract
from export_gemma4_lora_hf_oracle import publish_staging
from gemma4_oracle_contract import (
    ANTFLY_ADAPTER_KEY_FORMAT,
    ContractError,
    TensorStore,
    antfly_to_stock_peft_tensor_name,
    build_evidence_ledger,
    canonicalize_adapter_tensor_name,
    compare_traces,
    fingerprint_dataset_source,
    fingerprint_prepared_examples_v2,
    fingerprint_prepared_examples_v3,
    git_blob_sha1,
    load_lock,
    load_prepared_example,
    prefixed_sha256,
    read_adapter_config,
    sha256_file,
    validate_trace,
    validate_evidence_ledger,
    validate_target_inventory,
    verify_model_directory,
    verify_requirements_match_lock,
    write_json,
)


class Gemma4OracleContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lock_path = Path(contract.__file__).with_name("gemma4_oracle.lock.json")
        self.lock = load_lock(self.lock_path)

    def test_lock_requirements_and_source_revisions_are_exact(self) -> None:
        script_dir = Path(contract.__file__).parent
        self.assertEqual(
            self.lock["python_oracle"]["packages"],
            verify_requirements_match_lock(
                self.lock,
                "python_oracle",
                script_dir / "requirements-gemma4-oracle.txt",
            ),
        )
        self.assertEqual(
            self.lock["mlx_reference"]["packages"],
            verify_requirements_match_lock(
                self.lock,
                "mlx_reference",
                script_dir / "requirements-gemma4-mlx-reference.txt",
            ),
        )
        for environment in ("python_oracle", "mlx_reference"):
            for revision in self.lock[environment]["source_revisions"].values():
                self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_same_mac_lock_freezes_precision_optimizer_and_runtime_semantics(self) -> None:
        benchmark = self.lock["benchmark_contract"]
        self.assertEqual("bfloat16", benchmark["precision"]["verified"]["base_model_storage_dtype"])
        self.assertEqual(
            {
                "base_model_storage_dtype": "bfloat16",
                "lora_parameter_storage_dtype": "float32",
                "gradient_storage_dtype": "float32",
                "optimizer_moment_storage_dtype": "float32",
                "loss_tensor_dtype": "float32",
                "loss_reduction_input_dtype": "float32",
            },
            benchmark["precision"]["verified"],
        )
        self.assertEqual(
            ["activation_dtype", "matmul_accumulator_dtype"],
            benchmark["precision"]["not_asserted"],
        )
        self.assertEqual(
            contract.benchmark_precision_policy_sha256(benchmark["precision"]),
            self.lock["mlx_reference"]["native_runtime"]["precision_policy_sha256"],
        )
        self.assertTrue(benchmark["optimizer"]["bias_correction"])
        self.assertEqual("constant-no-warmup", benchmark["optimizer"]["schedule"])
        self.assertEqual(
            "after-optimizer-update-before-timer-stop-every-window",
            benchmark["runtime"]["per_step_device_sync"],
        )
        drifted = copy.deepcopy(self.lock)
        drifted["benchmark_contract"]["optimizer"]["weight_decay_scope"] = "unspecified"
        with self.assertRaisesRegex(ContractError, "frozen same-Mac contract"):
            contract.validate_lock(drifted)
        stale_native_policy = copy.deepcopy(self.lock)
        stale_native_policy["mlx_reference"]["native_runtime"]["precision_policy_sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ContractError, "precision policy digest differs"):
            contract.validate_lock(stale_native_policy)

    def test_benchmark_producer_attestation_closes_clean_committed_source_tree(self) -> None:
        repository_root = Path(contract.__file__).resolve().parents[4]
        entrypoint_relative = "zig/pkg/inference/scripts/run_gemma4_lora_mlx_benchmark.py"
        entrypoint = repository_root / entrypoint_relative
        revision = "b" * 40
        source_tree = "d" * 40

        def git_result(*, dirty: bool = False, drift_path: str | None = None):
            def run(command, **kwargs):
                arguments = tuple(command[3:])
                text_mode = kwargs.get("text", True)
                if arguments == ("rev-parse", "--show-toplevel"):
                    stdout = str(repository_root) + "\n"
                elif arguments == ("rev-parse", "HEAD"):
                    stdout = revision + "\n"
                elif arguments == ("rev-parse", "HEAD^{tree}"):
                    stdout = source_tree + "\n"
                elif arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
                    stdout = "?? unrelated.tmp\n" if dirty else ""
                elif arguments[:3] == ("ls-files", "--error-unmatch", "--"):
                    stdout = ""
                elif arguments[0] == "show":
                    relative_path = arguments[1].split(":", 1)[1]
                    stdout = (
                        b"committed drift"
                        if relative_path == drift_path
                        else (repository_root / relative_path).read_bytes()
                    )
                else:  # pragma: no cover - protects the closed Git command surface
                    raise AssertionError(f"unexpected git command: {arguments!r}")
                if not text_mode and isinstance(stdout, str):
                    stdout = stdout.encode("utf-8")
                return mock.Mock(stdout=stdout, stderr=b"" if not text_mode else "", returncode=0)

            return run

        with mock.patch.object(contract.subprocess, "run", side_effect=git_result()):
            attestation = contract.attest_benchmark_producer_source(
                entrypoint,
                expected_entrypoint=entrypoint_relative,
            )
        self.assertEqual(revision, attestation["source_revision"])
        self.assertEqual(source_tree, attestation["source_tree"])
        self.assertEqual(
            list(contract.BENCHMARK_PRODUCER_RELATIVE_PATHS),
            [item["relative_path"] for item in attestation["files"]],
        )

        with mock.patch.object(contract.subprocess, "run", side_effect=git_result(dirty=True)):
            with self.assertRaisesRegex(ContractError, "completely clean"):
                contract.attest_benchmark_producer_source(
                    entrypoint,
                    expected_entrypoint=entrypoint_relative,
                )

        drift_path = "zig/pkg/inference/scripts/gemma4_oracle_contract.py"
        with mock.patch.object(
            contract.subprocess,
            "run",
            side_effect=git_result(drift_path=drift_path),
        ):
            with self.assertRaisesRegex(ContractError, "differs from commit"):
                contract.attest_benchmark_producer_source(
                    entrypoint,
                    expected_entrypoint=entrypoint_relative,
                )

    def test_model_verifier_checks_size_and_digest_without_network(self) -> None:
        synthetic = copy.deepcopy(self.lock)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            for name, spec in synthetic["models"]["gemma-4-E2B-it"]["files"].items():
                path = model_dir / name
                path.write_bytes((name + "\n").encode())
                spec["size"] = path.stat().st_size
                if "sha256" in spec:
                    spec["sha256"] = sha256_file(path)
                else:
                    spec["git_blob_sha1"] = git_blob_sha1(path)
            verified = verify_model_directory(synthetic, "gemma-4-E2B-it", model_dir)
            self.assertRegex(verified["local_artifact_sha256"], r"^sha256:[0-9a-f]{64}$")
            (model_dir / "config.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "size|git_blob_sha1"):
                verify_model_directory(synthetic, "gemma-4-E2B-it", model_dir)

    def test_stock_peft_and_antfly_tensor_keys_have_one_canonical_identity(self) -> None:
        stock = "base_model.model.model.language_model.model.layers.3.self_attn.q_proj.lora_A.default.weight"
        antfly = "model.layers.3.self_attn.q_proj.weight.lora_A.weight"
        self.assertEqual(
            ("model.layers.3.self_attn.q_proj", "lora_A"),
            canonicalize_adapter_tensor_name(stock),
        )
        self.assertEqual(canonicalize_adapter_tensor_name(stock), canonicalize_adapter_tensor_name(antfly))
        self.assertEqual(
            canonicalize_adapter_tensor_name(antfly),
            canonicalize_adapter_tensor_name(
                "base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_A.weight"
            ),
        )
        self.assertEqual(
            "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
            antfly_to_stock_peft_tensor_name(antfly),
        )
        with self.assertRaisesRegex(ContractError, "requires an Antfly"):
            antfly_to_stock_peft_tensor_name(stock)
        self.assertEqual(
            ("model.per_layer_input.per_layer_model_proj", "lora_B"),
            canonicalize_adapter_tensor_name("model.per_layer_model_projection.lora_B.weight"),
        )
        with self.assertRaisesRegex(ContractError, "unsupported adapter tensor name"):
            canonicalize_adapter_tensor_name("model.layers.0.self_attn.q_proj.weight")

    def test_target_preset_requires_the_complete_model_inventory(self) -> None:
        with self.assertRaisesRegex(ContractError, "incomplete target inventory"):
            validate_target_inventory(
                self.lock,
                "gemma-4-E2B-it",
                "peft-qv",
                ["model.layers.0.self_attn.q_proj"],
            )
        complete = [f"model.layers.{layer}.self_attn.q_proj" for layer in range(35)]
        complete.extend(f"model.layers.{layer}.self_attn.v_proj" for layer in range(15))
        self.assertEqual(
            {"q_proj": 35, "v_proj": 15},
            validate_target_inventory(self.lock, "gemma-4-E2B-it", "peft-qv", complete),
        )

    @staticmethod
    def prepared_example() -> dict:
        return {
            "mode": "instruction",
            "prompt_input_ids": [1, 2],
            "response_input_ids": [3],
            "num_prompt_tokens": 2,
            "num_response_tokens": 1,
            "input_ids": [1, 2, 3],
            "labels": [-100, -100, 3],
            "num_input_tokens": 3,
            "num_supervised_tokens": 1,
            "turn_count": 2,
            "has_tool_calls": False,
            "has_tool_messages": False,
            "image_paths": [],
            "audio_paths": [],
            "image_token_counts": [],
            "audio_token_counts": [],
            "teacher_top_k_token_ids": [],
            "teacher_top_k_probs": [],
            "teacher_top_k": 0,
            "teacher_temperature": 1.0,
            "was_truncated": False,
            "turns_dropped_from_left": 0,
            "policy_version": "test/v1",
            "source_id": "row-1",
            "source_group_id": "group-1",
            "source_name": "unit-test",
            "source_record_sha256": "1" * 64,
            "rendered_chat_sha256": "2" * 64,
            "media_content_sha256": [],
        }

    def write_prepared(self, root: Path) -> tuple[Path, Path]:
        source = root / "source.jsonl"
        source.write_text('{"split":"train","id":"row-1"}\n', encoding="utf-8")
        example = self.prepared_example()
        payload = {
            "summary": {
                "artifact_family_version": "gemma4_lora/v1alpha1",
                "model_dir": "/locked/model",
                "schema_version": "gemma4_prepared/v6",
                "max_examples": 1,
                "examples_seen": 1,
                "base_model_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "chat_template_sha256": "c" * 64,
                "prepared_examples_sha256": fingerprint_prepared_examples_v3([example]),
                "source_dataset_path": str(source),
                "source_dataset_sha256": fingerprint_dataset_source(source, "train"),
                "source_split": "train",
                "source_revision": "dataset-revision-1",
                "max_seq_len": 8,
                "max_input_tokens": 3,
                "max_supervised_tokens": 1,
                "examples_with_images": 0,
                "examples_with_audio": 0,
                "examples": [example],
            }
        }
        prepared = root / "prepared.json"
        prepared.write_text(json.dumps(payload), encoding="utf-8")
        return prepared, source

    def test_v6_prepared_loader_recomputes_payload_and_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared, source = self.write_prepared(Path(tmp))
            summary, selected = load_prepared_example(prepared, 0)
            self.assertEqual("gemma4_prepared/v6", selected["schema_version"])
            self.assertEqual("group-1", selected["source_group_id"])
            self.assertEqual(fingerprint_dataset_source(source, "train"), summary["source_dataset_sha256"])

            payload = json.loads(prepared.read_text(encoding="utf-8"))
            payload["summary"]["prepared_examples_sha256"] = fingerprint_prepared_examples_v2(
                payload["summary"]["examples"]
            )
            prepared.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "prepared_examples_sha256"):
                load_prepared_example(prepared, 0)

            payload["summary"]["prepared_examples_sha256"] = fingerprint_prepared_examples_v3(
                payload["summary"]["examples"]
            )
            payload["summary"]["examples"][0]["source_record_sha256"] = "3" * 64
            prepared.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "prepared_examples_sha256"):
                load_prepared_example(prepared, 0)

    def test_antfly_sidecar_owns_policy_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            module = "model.layers.0.self_attn.q_proj"
            checkpoint_bytes = b"synthetic-adapter-checkpoint"
            checkpoint_path = adapter / "adapter_model.safetensors"
            checkpoint_path.write_bytes(checkpoint_bytes)
            (adapter / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "google/gemma-4-E2B-it",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 1,
                "lora_alpha": 2.0,
                "lora_dropout": 0.0,
                "target_modules": [module],
                "use_dora": False,
                "use_rslora": False,
                "modules_to_save": None,
                "init_lora_weights": True,
            }), encoding="utf-8")
            (adapter / "antfly_finetune_manifest.json").write_text(json.dumps({
                "schema_version": "antfly_gemma4_finetune/v2",
                "status": "complete",
                "artifact_family_version": "gemma4_lora/v1alpha1",
                "tensor_key_format": ANTFLY_ADAPTER_KEY_FORMAT,
                "adapter_checkpoint_sha256": sha256_file(checkpoint_path),
                "adapter_checkpoint_size_bytes": len(checkpoint_bytes),
                "base_model_name_or_path": "google/gemma-4-E2B-it",
                "base_model_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "chat_template_sha256": "c" * 64,
                "target_modules": [module],
                "target_preset": "peft-qv",
                "rank": 1,
                "alpha": 2.0,
                "use_dora": False,
                "use_rslora": False,
                "initializer": None,
                "recursive_lora": None,
            }), encoding="utf-8")
            result = read_adapter_config(adapter)
            self.assertEqual("peft-qv", result["target_preset"])
            self.assertEqual("antfly-finetune-manifest/v2", result["policy_source"])
            self.assertEqual("a" * 64, result["provenance"]["base_model_sha256"])
            self.assertEqual(ANTFLY_ADAPTER_KEY_FORMAT, result["provenance"]["tensor_key_format"])
            with self.assertRaisesRegex(ContractError, "conflicts"):
                read_adapter_config(adapter, target_preset="text-all-linear")

            (adapter / "antfly_finetune_manifest.json").unlink()
            with self.assertRaisesRegex(ContractError, "requires an explicit target preset"):
                read_adapter_config(adapter)
            stock = read_adapter_config(adapter, target_preset="peft-qv")
            self.assertEqual("explicit-lock-policy", stock["policy_source"])

            config_path = adapter / "adapter_config.json"
            (adapter / "antfly_peft_export.json").write_text(json.dumps({
                "schema_version": "antfly_gemma4_peft_export/v1",
                "status": "complete",
                "source_artifact_family_version": "gemma4_lora/v1alpha1",
                "source_tensor_key_format": ANTFLY_ADAPTER_KEY_FORMAT,
                "destination_tensor_key_format": "stock-peft/v1",
                "source_adapter_model_sha256": "d" * 64,
                "destination_adapter_model_sha256": sha256_file(checkpoint_path),
                "destination_adapter_model_size_bytes": len(checkpoint_bytes),
                "adapter_config_sha256": sha256_file(config_path),
                "base_model_name_or_path": "google/gemma-4-E2B-it",
                "base_model_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "chat_template_sha256": "c" * 64,
                "target_preset": "peft-qv",
                "tensor_count": 2,
            }), encoding="utf-8")
            exported = read_adapter_config(adapter)
            self.assertEqual("antfly-peft-export/v1", exported["policy_source"])
            self.assertEqual("stock-peft/v1", exported["provenance"]["tensor_key_format"])
            self.assertEqual("d" * 64, exported["provenance"]["source_adapter_checkpoint_sha256"])
            self.assertEqual(2, exported["provenance"]["tensor_count"])
            with self.assertRaisesRegex(ContractError, "conflicts"):
                read_adapter_config(adapter, target_preset="text-all-linear")

            checkpoint_path.write_bytes(checkpoint_bytes + b"tampered")
            with self.assertRaisesRegex(ContractError, "size does not match"):
                read_adapter_config(adapter)

    def synthetic_trace(self, prepared: dict, *, antfly: bool, delta: float = 0.0) -> dict:
        module = "model.layers.0.self_attn.q_proj"
        entries = {}
        targets = []
        values = {
            "lora_A": {
                "initial": [0.1, 0.2],
                "gradient": [0.0, 0.0],
                "updated": [0.1, 0.2],
                "optimizer_m": [0.0, 0.0],
                "optimizer_v": [0.0, 0.0],
            },
            "lora_B": {
                "initial": [0.0, 0.0],
                "gradient": [0.3 + delta, 0.4],
                "updated": [-0.001 + delta, -0.001],
                "optimizer_m": [0.03 + delta, 0.04],
                "optimizer_v": [0.00009 + delta * 0.001, 0.00016],
            },
        }
        for role in ("lora_A", "lora_B"):
            logical = {}
            for state, state_values in values[role].items():
                name = f"{module}:{role}:{state}"
                entries[name] = {"shape": [1, 2] if role == "lora_A" else [2, 1], "dtype": "float64", "values": state_values}
                logical[state] = name
            if antfly:
                source = f"{module}.weight.{role}.weight"
            else:
                source = f"base_model.model.model.language_model.{module}.{role}.default.weight"
            targets.append({
                "canonical_name": module,
                "source_name": source,
                "role": role,
                "shape": [1, 2] if role == "lora_A" else [2, 1],
                "gradient_expectation": "zero-by-zero-b-initialization" if role == "lora_A" else "active",
                "logical_tensors": logical,
            })
        model = self.lock["models"]["gemma-4-E2B-it"]
        return {
            "schema_version": "antfly_gemma4_lora_trace/v1",
            "producer": {
                "name": "synthetic-test",
                "version": "test-v1",
                "source_revision": "test",
                "hardware": {"device": "synthetic"},
            },
            "oracle_lock_sha256": contract.lock_digest(self.lock_path),
            "model": {
                "key": "gemma-4-E2B-it",
                "repo_id": model["repo_id"],
                "revision": model["revision"],
                "local_artifact_sha256": "sha256:" + "d" * 64,
            },
            "prepared": prepared,
            "training": {
                "optimizer": "adamw",
                "seed": 42,
                "step": 1,
                "rank": 1,
                "alpha": 2.0,
                "scale": 2.0,
                "target_preset": "peft-qv",
                "learning_rate": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "grad_accum_steps": 1,
                "supervised_token_normalization": "mean",
                "dropout": 0.0,
                "use_cache": False,
            },
            "metrics": {
                "loss": 1.0 + delta,
                "loss_history": [1.0 + delta],
                "grad_norm": math.hypot(0.3 + delta, 0.4),
                "supervised_tokens": 1,
            },
            "logit_probes": [{
                "predictor_position": 1,
                "target_token_id": 3,
                "token_ids": [0, 3],
                "values": [0.25 + delta, 1.25],
                "logsumexp": 2.0 + delta,
            }],
            "target_tensors": targets,
            "tensor_store": {"format": "inline-f64/v1", "entries": entries},
            "artifact": {
                "adapter_config_semantics": {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": 1,
                    "lora_alpha": 2.0,
                    "target_preset": "peft-qv",
                    "target_modules": [module],
                    "use_dora": False,
                    "lora_dropout": 0.0,
                },
                "adapter_model_sha256": "sha256:" + ("e" if antfly else "f") * 64,
                "tensor_inventory": [f"{module}:lora_A", f"{module}:lora_B"],
                "key_layout": ANTFLY_ADAPTER_KEY_FORMAT if antfly else "stock-peft/v1",
                "policy_source": "antfly-finetune-manifest/v2" if antfly else "explicit-lock-policy",
            },
        }

    def test_trace_comparison_normalizes_layout_but_does_not_claim_direct_interop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_path, _ = self.write_prepared(root)
            _, prepared = load_prepared_example(prepared_path, 0)
            reference_path = root / "hf.json"
            candidate_path = root / "zig.json"
            reference_path.write_text(json.dumps(self.synthetic_trace(prepared, antfly=False)), encoding="utf-8")
            candidate_path.write_text(json.dumps(self.synthetic_trace(prepared, antfly=True, delta=1e-8)), encoding="utf-8")
            synthetic_lock = copy.deepcopy(self.lock)
            synthetic_lock["target_inventory"]["gemma-4-E2B-it"]["peft-qv"] = {"q_proj": 1}
            with self.assertRaisesRegex(ContractError, "explicit test code"):
                validate_trace(reference_path, synthetic_lock, lock_path=self.lock_path)
            reference = validate_trace(
                reference_path,
                synthetic_lock,
                lock_path=self.lock_path,
                allow_synthetic=True,
            )
            candidate = validate_trace(
                candidate_path,
                synthetic_lock,
                lock_path=self.lock_path,
                allow_synthetic=True,
            )
            result = compare_traces(reference, candidate, synthetic_lock, "tiny-f32")
            self.assertTrue(result["ok"], result["failures"])
            self.assertRegex(result["reference_trace_sha256"], r"^sha256:[0-9a-f]{64}$")
            artifact = result["comparisons"]["artifact_semantics"]
            self.assertTrue(artifact["canonical_name_normalization_applied"])
            self.assertFalse(artifact["direct_bidirectional_interoperability_proven"])
            with self.assertRaisesRegex(ContractError, "producer pair"):
                compare_traces(reference, candidate, synthetic_lock, "hf-zig-bf16")

            broken = self.synthetic_trace(prepared, antfly=True)
            gradient_name = broken["target_tensors"][1]["logical_tensors"]["gradient"]
            broken["tensor_store"]["entries"][gradient_name]["values"] = [0.0, 0.0]
            candidate_path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "all-zero gradient"):
                validate_trace(
                    candidate_path,
                    synthetic_lock,
                    lock_path=self.lock_path,
                    allow_synthetic=True,
                )

    def test_trace_accepts_hash_bound_antfly_peft_export_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_path, _ = self.write_prepared(root)
            _, prepared = load_prepared_example(prepared_path, 0)
            synthetic_lock = copy.deepcopy(self.lock)
            synthetic_lock["target_inventory"]["gemma-4-E2B-it"]["peft-qv"] = {"q_proj": 1}
            payload = self.synthetic_trace(prepared, antfly=False)
            payload["artifact"]["policy_source"] = "antfly-peft-export/v1"
            trace_path = root / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            validated = validate_trace(
                trace_path,
                synthetic_lock,
                lock_path=self.lock_path,
                allow_synthetic=True,
            )
            self.assertEqual(
                "antfly-peft-export/v1",
                validated.payload["artifact"]["policy_source"],
            )

    def test_trace_binds_dtype_and_global_raw_gradient_norm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_path, _ = self.write_prepared(root)
            _, prepared = load_prepared_example(prepared_path, 0)
            synthetic_lock = copy.deepcopy(self.lock)
            synthetic_lock["target_inventory"]["gemma-4-E2B-it"]["peft-qv"] = {"q_proj": 1}
            trace_path = root / "trace.json"

            wrong_norm = self.synthetic_trace(prepared, antfly=True)
            wrong_norm["metrics"]["grad_norm"] = 0.75
            trace_path.write_text(json.dumps(wrong_norm), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "serialized raw gradients"):
                validate_trace(
                    trace_path,
                    synthetic_lock,
                    lock_path=self.lock_path,
                    allow_synthetic=True,
                )

            wrong_dtype = self.synthetic_trace(prepared, antfly=True)
            first_entry = next(iter(wrong_dtype["tensor_store"]["entries"].values()))
            first_entry["dtype"] = "float32"
            trace_path.write_text(json.dumps(wrong_dtype), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "inline-f64/v1 requires dtype=float64"):
                validate_trace(
                    trace_path,
                    synthetic_lock,
                    lock_path=self.lock_path,
                    allow_synthetic=True,
                )

    def test_complete_marker_is_a_closed_hash_bound_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_path = root / "trace.json"
            trace_path.write_text("{}\n", encoding="utf-8")
            adapter = root / "reference_adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            write_json(root / "COMPLETE.json", build_evidence_ledger(root))
            marker_sha, files = validate_evidence_ledger(trace_path)
            self.assertRegex(marker_sha, r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(
                {"trace.json", "reference_adapter/adapter_config.json"},
                set(files),
            )
            trace_path.write_text("{\"tampered\":true}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "digest/size mismatch"):
                validate_evidence_ledger(trace_path)
            trace_path.write_text("{}\n", encoding="utf-8")
            (root / "uncommitted.txt").write_text("late evidence\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "closed evidence inventory"):
                validate_evidence_ledger(trace_path)

    def test_external_tensor_dtype_is_checked_against_safetensors_payload(self) -> None:
        class FakeTensor:
            shape = (1,)
            dtype = "float64"

            @staticmethod
            def reshape(*_shape: int) -> list[float]:
                return [0.0]

        class FakeSafeOpen:
            def __enter__(self) -> "FakeSafeOpen":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            @staticmethod
            def keys() -> list[str]:
                return ["tensor_0"]

            @staticmethod
            def get_tensor(_key: str) -> FakeTensor:
                return FakeTensor()

        fake_module = types.ModuleType("safetensors")
        fake_module.safe_open = lambda *_args, **_kwargs: FakeSafeOpen()  # type: ignore[attr-defined]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_path = root / "trace.json"
            trace_path.write_text("{}\n", encoding="utf-8")
            store_path = root / "trace.safetensors"
            store_path.write_bytes(b"synthetic")
            store = TensorStore(trace_path, {
                "format": "safetensors/v1",
                "path": store_path.name,
                "sha256": prefixed_sha256(store_path),
                "entries": {
                    "logical": {
                        "shape": [1],
                        "dtype": "float32",
                        "storage_key": "tensor_0",
                    }
                },
            })
            with mock.patch.dict(sys.modules, {"safetensors": fake_module}):
                with self.assertRaisesRegex(ContractError, "Safetensors dtype"):
                    store.get("logical")

    def test_publication_moves_complete_marker_last_into_a_no_replace_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            staging.mkdir()
            trace_path = staging / "trace.json"
            trace_path.write_text("{}\n", encoding="utf-8")
            write_json(staging / "COMPLETE.json", build_evidence_ledger(staging))
            output = root / "published"
            publish_staging(staging, output)
            self.assertTrue((output / "COMPLETE.json").is_file())
            validate_evidence_ledger(output / "trace.json")

            second_staging = root / "second-staging"
            second_staging.mkdir()
            (second_staging / "trace.json").write_text("{}\n", encoding="utf-8")
            write_json(second_staging / "COMPLETE.json", build_evidence_ledger(second_staging))
            with self.assertRaisesRegex(ContractError, "refusing to replace"):
                publish_staging(second_staging, output)

    def test_release_trace_requires_complete_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_path, _ = self.write_prepared(root)
            _, prepared = load_prepared_example(prepared_path, 0)
            synthetic_lock = copy.deepcopy(self.lock)
            synthetic_lock["target_inventory"]["gemma-4-E2B-it"]["peft-qv"] = {"q_proj": 1}
            payload = self.synthetic_trace(prepared, antfly=False)
            payload["producer"]["name"] = "hf-peft"
            trace_path = root / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "complete oracle evidence requires"):
                validate_trace(trace_path, synthetic_lock, lock_path=self.lock_path)

    def test_max_abs_gate_catches_a_single_coordinate_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_path, _ = self.write_prepared(root)
            _, prepared = load_prepared_example(prepared_path, 0)
            synthetic_lock = copy.deepcopy(self.lock)
            synthetic_lock["target_inventory"]["gemma-4-E2B-it"]["peft-qv"] = {"q_proj": 1}
            synthetic_lock["tolerance_profiles"]["tiny-f32"]["state_max_abs"] = 0.01
            synthetic_lock["tolerance_profiles"]["tiny-f32"]["state_rel_l2"] = 10.0
            synthetic_lock["tolerance_profiles"]["tiny-f32"]["state_cosine_min"] = -1.0
            reference_payload = self.synthetic_trace(prepared, antfly=False)
            candidate_payload = self.synthetic_trace(prepared, antfly=True)
            updated = candidate_payload["target_tensors"][1]["logical_tensors"]["updated"]
            candidate_payload["tensor_store"]["entries"][updated]["values"][0] += 0.1
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")
            reference = validate_trace(
                reference_path,
                synthetic_lock,
                lock_path=self.lock_path,
                allow_synthetic=True,
            )
            candidate = validate_trace(
                candidate_path,
                synthetic_lock,
                lock_path=self.lock_path,
                allow_synthetic=True,
            )
            result = compare_traces(reference, candidate, synthetic_lock, "tiny-f32")
            self.assertFalse(result["ok"])
            state = result["comparisons"]["target_tensors"][1]["states"]["updated"]
            self.assertGreater(state["max_abs"], state["max_abs_limit"])


if __name__ == "__main__":
    unittest.main()
