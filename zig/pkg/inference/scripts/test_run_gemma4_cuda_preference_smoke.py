#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().with_name("run_gemma4_cuda_preference_smoke.py")
SPEC = importlib.util.spec_from_file_location("gemma4_cuda_preference_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def write_safetensors(path: pathlib.Path, dtype: str = "BF16") -> None:
    header = json.dumps(
        {
            "model.embed_tokens.weight": {
                "dtype": dtype,
                "shape": [4, 4],
                "data_offsets": [0, 32 if dtype == "BF16" else 64],
            }
        },
        separators=(",", ":"),
    ).encode()
    data_size = 32 if dtype == "BF16" else 64
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(data_size))


def write_model(root: pathlib.Path, dtype: str = "BF16") -> pathlib.Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": {
                    "hidden_size": 1536,
                    "num_hidden_layers": 35,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 1,
                    "head_dim": 256,
                },
            }
        ),
        encoding="utf-8",
    )
    (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    write_safetensors(model / "model.safetensors", dtype)
    return model


def valid_evidence() -> dict[str, object]:
    return {
        "schema_version": "antfly_training_execution_evidence/v1",
        "train_steps": 2,
        "eval_steps": 1,
        "graph_executor_partitions": 3,
        "graph_executor_planned_dispatches": 40,
        "graph_executor_fallback_steps": 0,
        "graph_executor_native_partitions": 0,
        "graph_executor_unsupported_ops": 0,
        "graph_executor_interpreter_fallbacks": 0,
        "graph_executor_true_host_outputs": 0,
        "runtime_input_uploads": 9,
        "runtime_input_upload_bytes": 96,
        "runtime_input_h2d_bytes": 96,
        "runtime_input_d2h_bytes": 0,
        "declared_runtime_input_uploads": 9,
        "declared_runtime_input_upload_bytes": 96,
        "declared_runtime_input_h2d_bytes": 96,
        "compiled_session_setup_d2h_bytes": 0,
        "graph_execution_h2d_bytes": 0,
        "graph_execution_d2h_bytes": 12,
        "host_gradient_tensors": 0,
        "cuda_kernel_launches": 40,
        "cuda_largest_d2h_transfer_bytes": 4,
    }


def valid_trainable_update() -> dict[str, object]:
    return {
        "tensor_count": 2,
        "changed_tensor_count": 2,
        "max_abs_delta": 0.001,
    }


class Gemma4CudaPreferenceSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)

    def test_preflight_accepts_bf16_without_reading_payload_semantics(self) -> None:
        model = write_model(self.root)
        result = smoke.inspect_model(smoke.ModelSpec("e2b", model))
        self.assertEqual("safetensors", result.artifact_kind)
        self.assertEqual(["BF16"], result.rank2_dtypes)
        self.assertEqual(1536, result.topology["hidden_size"])
        self.assertEqual(1, result.tensor_count)

    def test_preflight_rejects_packed_gguf(self) -> None:
        model = self.root / "gguf-model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"gemma4"}\n', encoding="utf-8")
        (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        (model / "gemma-4-E2B-Q4_0.gguf").write_bytes(b"GGUF")
        with self.assertRaisesRegex(smoke.QualificationError, "packed GGUF"):
            smoke.inspect_model(smoke.ModelSpec("e2b", model))

    def test_preflight_rejects_rank2_f16(self) -> None:
        model = write_model(self.root, "F16")
        with self.assertRaisesRegex(smoke.QualificationError, "unsupported rank-2"):
            smoke.inspect_model(smoke.ModelSpec("e4b", model))

    def test_preflight_resolves_every_sharded_safetensors_dependency(self) -> None:
        model = write_model(self.root)
        (model / "model.safetensors").unlink()
        first = model / "model-00001-of-00002.safetensors"
        second = model / "model-00002-of-00002.safetensors"
        write_safetensors(first)
        write_safetensors(second)
        (model / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.embed_tokens.weight": first.name,
                        "model.layers.0.self_attn.q_proj.weight": second.name,
                    }
                }
            ),
            encoding="utf-8",
        )
        result = smoke.inspect_model(smoke.ModelSpec("e2b", model))
        self.assertEqual("sharded_safetensors", result.artifact_kind)
        self.assertEqual(2, result.shard_count)
        self.assertEqual(2, result.tensor_count)

    def test_report_gate_accepts_strict_cuda_evidence(self) -> None:
        report_path = self.root / "dpo_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "policy_backend": "cuda",
                    "optimizer_steps": 1,
                    "loss": 0.5,
                    "trainable_update": valid_trainable_update(),
                    "device_execution": valid_evidence(),
                }
            ),
            encoding="utf-8",
        )
        parsed = smoke.validate_report(report_path, "dpo")
        self.assertEqual(1, parsed["optimizer_steps"])

    def test_report_gate_rejects_host_gradient_evidence(self) -> None:
        evidence = valid_evidence()
        evidence["host_gradient_tensors"] = 1
        report_path = self.root / "grpo_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "policy_backend": "cuda",
                    "optimizer_steps": 1,
                    "trainable_update": valid_trainable_update(),
                    "device_execution": evidence,
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(smoke.QualificationError, "host_gradient_tensors"):
            smoke.validate_report(report_path, "grpo")

    def test_report_gate_rejects_missing_trainable_tensor_movement(self) -> None:
        report_path = self.root / "dpo_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "policy_backend": "cuda",
                    "optimizer_steps": 1,
                    "device_execution": valid_evidence(),
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(smoke.QualificationError, "tensor movement"):
            smoke.validate_report(report_path, "dpo")

    def test_recipe_is_bounded_cuda_peft_qv(self) -> None:
        model = smoke.ModelSpec("e2b", self.root / "model")
        recipe = smoke.recipe_for("grpo", model, self.root / "data.jsonl", self.root / "run", 128, 8, 32)
        self.assertEqual("cuda", recipe["backend"])
        self.assertEqual("peft-qv", recipe["adapter"]["target_preset"])
        self.assertEqual(1, recipe["dataset"]["max_examples"])
        self.assertEqual(1, recipe["grpo"]["max_completion_tokens"])

    def test_matched_grpo_uses_model_independent_ranked_reward(self) -> None:
        models = [smoke.ModelSpec("e2b", self.root / "model")]
        smoke.validate_grpo_targets(models, {}, ["dpo", "grpo"], True)
        with self.assertRaisesRegex(smoke.QualificationError, "ranked-first"):
            smoke.validate_grpo_targets(models, {"e2b": "Paris"}, ["grpo"], True)

    def test_matched_recipe_reuses_explicit_initial_adapter_and_runs_25_updates(self) -> None:
        model = smoke.ModelSpec("e2b", self.root / "model")
        initial = self.root / "initial-adapter"
        recipe = smoke.recipe_for(
            "dpo",
            model,
            self.root / "data.jsonl",
            self.root / "run",
            128,
            16,
            32,
            epochs=smoke.MATCHED_BENCHMARK_UPDATES,
            initial_adapter_dir=initial,
        )
        self.assertEqual(str(initial), recipe["adapter"]["path"])
        self.assertEqual(16, recipe["adapter"]["rank"])
        self.assertEqual(25, recipe["optimizer"]["epochs"])

        grpo_recipe = smoke.recipe_for(
            "grpo",
            model,
            self.root / "grpo.jsonl",
            self.root / "grpo-run",
            128,
            16,
            32,
            epochs=smoke.MATCHED_BENCHMARK_UPDATES,
            initial_adapter_dir=initial,
            matched_benchmark=True,
        )
        self.assertEqual("rendered-text-grpo", grpo_recipe["dataset"]["format"])
        self.assertEqual("ranked-first", grpo_recipe["grpo"]["reward_mode"])
        self.assertEqual(smoke.MATCHED_GRPO_PAYLOAD_IDS, smoke.MATCHED_GRPO_PROMPT_IDS)

    def test_matched_dpo_report_requires_exact_tokens_and_timing_protocol(self) -> None:
        measured = [0.3 + index * 0.001 for index in range(20)]
        report = {
            "optimizer_steps": 25,
            "examples": 25,
            "input_contract": {
                "prompt_input_ids": smoke.MATCHED_DPO_PROMPT_IDS,
                "chosen_input_ids": smoke.MATCHED_DPO_CHOSEN_IDS,
                "rejected_input_ids": smoke.MATCHED_DPO_REJECTED_IDS,
            },
            "initial_logprob_parity": {"base_equivalent_policy": True, "max_abs_error": 0.0},
            "benchmark": {
                "protocol": smoke.MATCHED_BENCHMARK_PROTOCOL,
                "cold_seconds": 1.0,
                "first_seconds": 0.4,
                "warmup_seconds": [0.35, 0.34, 0.33],
                "measured_seconds": measured,
                "median_seconds": smoke.statistics.median(measured),
                "mean_seconds": smoke.statistics.fmean(measured),
            },
        }
        benchmark = smoke.validate_matched_benchmark_report(report, "dpo")
        self.assertEqual(smoke.statistics.median(measured), benchmark["median_seconds"])
        report["input_contract"]["chosen_input_ids"] = [1904]
        with self.assertRaisesRegex(smoke.QualificationError, "token contract mismatch"):
            smoke.validate_matched_benchmark_report(report, "dpo")

    def test_matched_grpo_report_requires_ranked_reward_contract(self) -> None:
        def update(seconds: float) -> dict[str, object]:
            return {
                "seconds": seconds,
                "completion_tokens": 2,
                "mean_reward": 0.5,
                "reward_stddev": 0.5,
            }

        measured = [update(1.0 + index * 0.001) for index in range(20)]
        measured_seconds = [entry["seconds"] for entry in measured]
        report = {
            "optimizer_steps": 25,
            "groups": 25,
            "input_contract": {
                "prompt_input_ids": smoke.MATCHED_GRPO_PROMPT_IDS,
                "group_size": 2,
                "max_completion_tokens": 1,
                "sampling": "deterministic-ranked-top-k",
            },
            "initial_logprob_parity": {
                "base_equivalent_policy": True,
                "sampling_rescore_max_abs_error": 0.0,
                "policy_reference_max_abs_error": 0.0,
            },
            "benchmark": {
                "protocol": smoke.MATCHED_BENCHMARK_PROTOCOL,
                "cold": update(2.0),
                "first": update(1.2),
                "warmup": [update(1.1), update(1.05), update(1.02)],
                "measured": measured,
                "median_seconds": smoke.statistics.median(measured_seconds),
                "mean_seconds": smoke.statistics.fmean(measured_seconds),
            },
        }
        smoke.validate_matched_benchmark_report(report, "grpo")
        report["benchmark"]["measured"][0]["mean_reward"] = 0.0
        with self.assertRaisesRegex(smoke.QualificationError, "reward distribution"):
            smoke.validate_matched_benchmark_report(report, "grpo")

    def test_run_case_executes_cli_and_gates_persisted_artifacts(self) -> None:
        executable = self.root / "fake-antfly"
        executable.write_text(
            """#!/usr/bin/env python3
import json
import pathlib
import sys

assert sys.argv[1:3] == ["finetune", "run"]
recipe = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
artifacts = recipe["artifacts"]
bootstrap = pathlib.Path(artifacts["adapter_dir"])
trained = pathlib.Path(artifacts["trained_adapter_dir"])
bootstrap.mkdir(parents=True)
trained.mkdir(parents=True)
(bootstrap / "adapter_model.safetensors").write_bytes(b"initial")
(trained / "adapter_model.safetensors").write_bytes(b"trained")
evidence = {
    "schema_version": "antfly_training_execution_evidence/v1",
    "train_steps": 2,
    "eval_steps": 1,
    "graph_executor_partitions": 3,
    "graph_executor_planned_dispatches": 40,
    "graph_executor_fallback_steps": 0,
    "graph_executor_native_partitions": 0,
    "graph_executor_unsupported_ops": 0,
    "graph_executor_interpreter_fallbacks": 0,
    "graph_executor_true_host_outputs": 0,
    "runtime_input_uploads": 9,
    "runtime_input_upload_bytes": 96,
    "runtime_input_h2d_bytes": 96,
    "runtime_input_d2h_bytes": 0,
    "declared_runtime_input_uploads": 9,
    "declared_runtime_input_upload_bytes": 96,
    "declared_runtime_input_h2d_bytes": 96,
    "compiled_session_setup_d2h_bytes": 0,
    "graph_execution_h2d_bytes": 0,
    "graph_execution_d2h_bytes": 12,
    "host_gradient_tensors": 0,
    "cuda_kernel_launches": 40,
    "cuda_largest_d2h_transfer_bytes": 4,
}
report = json.dumps({
    "policy_backend": "cuda",
    "optimizer_steps": 1,
    "micro_batch_steps": 2,
    "loss": 0.5,
    "trainable_update": {"tensor_count": 2, "changed_tensor_count": 2, "max_abs_delta": 0.001},
    "device_execution": evidence,
})
pathlib.Path(artifacts["report_path"]).write_text(report, encoding="utf-8")
(trained / "antfly_preference_run_report.json").write_text(report, encoding="utf-8")
""",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        run_dir = self.root / "run"
        result = smoke.run_case(
            executable,
            smoke.ModelSpec("e2b", self.root / "model"),
            "dpo",
            None,
            run_dir,
            128,
            8,
            32,
            30,
        )
        self.assertEqual("passed", result["status"])
        self.assertNotEqual(result["initial_adapter_sha256"], result["trained_adapter_sha256"])
        self.assertEqual(1, result["optimizer_steps"])

    def test_shared_suite_executes_one_cli_and_attests_one_model_admission(self) -> None:
        executable = self.root / "fake-antfly-suite"
        executable.write_text(
            """#!/usr/bin/env python3
import json
import pathlib
import sys

assert sys.argv[1:3] == ["finetune", "run-suite"]
report_index = sys.argv.index("--report")
suite_report_path = pathlib.Path(sys.argv[report_index + 1])
recipe_paths = [pathlib.Path(value) for value in sys.argv[report_index + 2:]]
evidence = """
            + repr(valid_evidence())
            + """
model_path = None
for run_index, recipe_path in enumerate(recipe_paths, start=1):
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    model_path = recipe["model"]["path"]
    objective = recipe["recipe"]
    artifacts = recipe["artifacts"]
    bootstrap = pathlib.Path(artifacts["adapter_dir"])
    trained = pathlib.Path(artifacts["trained_adapter_dir"])
    bootstrap.mkdir(parents=True)
    trained.mkdir(parents=True)
    (bootstrap / "adapter_model.safetensors").write_bytes(b"initial")
    (trained / "adapter_model.safetensors").write_bytes(b"trained")
    report = json.dumps({
        "policy_backend": "cuda",
        "optimizer_steps": 1,
        "micro_batch_steps": 2,
        "loss": 0.5,
        "reward_mode": recipe.get("grpo", {}).get("reward_mode") if objective == "grpo" else None,
        "trainable_update": {"tensor_count": 2, "changed_tensor_count": 2, "max_abs_delta": 0.001},
        "device_execution": evidence,
        "preference_session": {
            "shared": True,
            "model_admissions": 1,
            "run_index": run_index,
            "reuse_hit": run_index > 1,
        },
    })
    pathlib.Path(artifacts["report_path"]).write_text(report, encoding="utf-8")
    (trained / "antfly_preference_run_report.json").write_text(report, encoding="utf-8")
suite_report_path.parent.mkdir(parents=True, exist_ok=True)
suite_report_path.write_text(json.dumps({
    "schema_version": "antfly_inference_gemma4_preference_suite/v2",
    "status": "succeeded",
    "model_path": model_path,
    "backend": "cuda",
    "runs_planned": len(recipe_paths),
    "runs_completed": len(recipe_paths),
    "model_admissions": 1,
    "reuse_hits": len(recipe_paths) - 1,
    "model_admission_seconds": 2.0,
    "total_duration_seconds": 5.0,
    "runs": [
        {"run_index": index, "objective": json.loads(path.read_text())["recipe"], "duration_seconds": 1.0}
        for index, path in enumerate(recipe_paths, start=1)
    ],
}), encoding="utf-8")
""",
            encoding="utf-8",
        )
        executable.chmod(0o755)

        results, suite = smoke.run_shared_suite(
            executable,
            smoke.ModelSpec("e4b", self.root / "model"),
            ["dpo", "grpo"],
            "Paris",
            self.root / "out",
            1,
            32,
            2,
            4,
            30,
        )
        self.assertEqual(2, len(results))
        self.assertFalse(results[0]["preference_session"]["reuse_hit"])
        self.assertTrue(results[1]["preference_session"]["reuse_hit"])
        self.assertEqual("shared-session-job", results[1]["duration_scope"])
        self.assertEqual(1.0, results[1]["duration_seconds"])
        self.assertEqual(2.0, suite["model_admission_seconds"])
        self.assertEqual(1, suite["report"]["model_admissions"])
        self.assertEqual(1, suite["report"]["reuse_hits"])


if __name__ == "__main__":
    unittest.main()
