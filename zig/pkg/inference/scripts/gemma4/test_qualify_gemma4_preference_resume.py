from __future__ import annotations

import hashlib
import json
import re
import stat
import struct
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import qualify_gemma4_preference_resume as qualifier


FAKE_ANTFLY = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import struct
import sys
import time

recipe = json.loads(pathlib.Path(sys.argv[-1]).read_text())
task = recipe["recipe"]
root = pathlib.Path(recipe["artifacts"]["root"])
trained = pathlib.Path(recipe["artifacts"]["trained_adapter_dir"])
report_path = pathlib.Path(recipe["artifacts"]["report_path"])
checkpoint_cfg = recipe["checkpoint"]
checkpoint = pathlib.Path(checkpoint_cfg.get("resume_path", root / f"gemma4_{task}_trainer_state.safetensors"))
epochs = recipe["optimizer"]["epochs"]
boundary = checkpoint_cfg["every_epochs"]
resume = "resume_path" in checkpoint_cfg
every_examples = checkpoint_cfg.get("every_examples", 0)
per_epoch = 2 * every_examples if every_examples else 1
fingerprint = "sha256:" + "a" * 64
incremental_enabled = recipe.get("runtime", {}).get("grpo_incremental_kv", False)
compiled_sampling_enabled = os.environ.get("ANTFLY_GEMMA4_GRPO_COMPILED_SAMPLING") == "1"

def incremental_telemetry(epoch):
    return {
        "groups": epoch,
        "prompt_prefill_forwards": epoch,
        "prompt_tail_prefill_forwards": epoch,
        "prompt_tail_prefill_candidates": epoch,
        "max_prompt_tail_batch_size": 1 if epoch else 0,
        "prompt_tail_clone_candidates": 0,
        "prompt_tail_clone_tokens": 0,
        "prompt_tail_cloning": False,
        "decode_forwards": epoch,
        "decode_forward_candidates": epoch * 4,
        "max_decode_batch_size": 4 if epoch else 0,
        "active_candidate_batching": True,
        "exact_logprob_rescore_forwards": epoch * 4,
        "resident_ranked_token_selections": 0,
        "host_logit_fallbacks": 0,
        "host_logit_sampling_rows": epoch,
        "shared_prompt_tokens": epoch * 16,
        "reused_candidate_prompt_tokens": epoch * 16,
        "cache_page_tokens": 16,
        "cache_dtype": "f32",
    }

def encode(fields):
    chunks = []
    for field in fields:
        chunks.extend(float((field >> (index * 16)) & 0xffff) for index in range(4))
    return chunks

def write_checkpoint(epoch, cursor=0):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    examples_count = epoch * per_epoch + cursor
    steps = examples_count * (2 if task == "dpo" else 4)
    optimizer_steps = examples_count
    aggregate = {
        "initial_adapter_digest": [0] * 32,
        **({"examples_seen": examples_count, "total_loss": float(examples_count), "total_margin": 0.0, "total_accuracy": 1.0}
           if task == "dpo" else {
               "total_loss": float(examples_count), "total_pg_loss": float(examples_count), "total_kl_loss": 0.0,
               "total_mean_kl": 0.0, "total_clip_fraction": 0.0, "total_groups": examples_count,
               "optimizer_groups": examples_count, "zero_reward_std_groups": 0,
               "all_truncated_groups": 0,
               "kl_rejected_groups": 0,
               "total_completions": examples_count * 4, "truncated_completions": 0,
               "total_tokens": examples_count * 4,
               "total_reward": float(examples_count), "total_reward_squared": float(examples_count),
               "saw_nonzero_reward_advantage": True, "saw_nonzero_policy_gradient": True,
               "initial_sampling_rescore_max_abs_error": 0.0,
               "initial_policy_reference_max_abs_error": 0.0,
               "initial_base_equivalent_policy": True, "captured_initial_logprob_parity": True,
               "policy_rescore_completions": 4, "diagnostic_first_tokens": [1] * 8,
               "diagnostic_policy_first_token_logps": [0.0] * 8,
               "diagnostic_reference_first_token_logps": [0.0] * 8,
               "diagnostic_first_token_count": 4, "kl_current_coef": 0.04,
               "kl_admitted_groups": examples_count, "kl_max_observed_mean": 0.0, "kl_trace": "trace\n",
               "reward_call_index": examples_count * 4, "reward_external_calls": 0,
               "reward_external_failures": 0, "reward_trace": "reward\n",
               "incremental_kv": incremental_telemetry(examples_count) if incremental_enabled else None,
           }),
    }
    state = {
        "schema_version": "antfly_gemma4_preference_checkpoint_state/v2",
        "task": task, "run_fingerprint_sha256": fingerprint, "epoch_index": epoch,
        "examples_into_epoch": cursor,
        "micro_batch_steps": steps, "optimizer_steps": optimizer_steps,
        "accumulation_micro_batches": 0, "dpo": aggregate if task == "dpo" else None,
        "grpo": aggregate if task == "grpo" else None,
    }
    state_bytes = json.dumps(state, separators=(",", ":")).encode()
    state_digest = hashlib.sha256(state_bytes).digest()
    state_path = pathlib.Path(f"{checkpoint}.preference-state-{state_digest.hex()}.json")
    state_path.write_bytes(state_bytes)
    rng = [int.from_bytes(state_digest[index:index + 8], "little") for index in range(0, 32, 8)]
    magic = 0x44504F2D43504B31 if task == "dpo" else 0x4752504F43504B31
    fields = [2, steps, optimizer_steps, steps, 0, 2 if task == "dpo" else 4, 42,
              epoch, 0, examples_count, magic, 0, *rng, 0, 0]
    values = encode(fields)
    data = struct.pack("<72f", *values)
    header = {"__trainer_state_v2": {"dtype": "F32", "shape": [72], "data_offsets": [0, len(data)]}}
    raw = json.dumps(header, separators=(",", ":")).encode()
    padding = b" " * ((8 - len(raw) % 8) % 8)
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_bytes(struct.pack("<Q", len(raw) + len(padding)) + raw + padding + data)
    os.replace(temporary, checkpoint)
    return state_path, "sha256:" + state_digest.hex()

if root.name == "interrupted-unpublished":
    if every_examples:
        write_checkpoint(0, every_examples)
    else:
        write_checkpoint(boundary)
    while True:
        time.sleep(1)

restored_state = None
if resume:
    candidates = sorted(checkpoint.parent.glob(checkpoint.name + ".preference-state-*.json"))
    if len(candidates) != 1:
        raise SystemExit("expected exactly one restored sidecar")
    restored_state = json.loads(candidates[0].read_text())
final_state_path, final_state_sha256 = write_checkpoint(epochs)
trained.mkdir(parents=True)
adapter_payload = b"identical-final-adapter"
(trained / "adapter_model.safetensors").write_bytes(adapter_payload)
(trained / "adapter_config.json").write_text("{}\n")
(trained / "antfly_finetune_manifest.json").write_text(json.dumps({
    "schema_version": "antfly_gemma4_finetune/v2",
    "status": "complete",
    "adapter_checkpoint_sha256": hashlib.sha256(adapter_payload).hexdigest(),
    "adapter_checkpoint_size_bytes": len(adapter_payload),
}))
checkpoint_summary = {
    "enabled": resume,
    "start_epoch": restored_state["epoch_index"] if resume else 0,
    "start_examples_into_epoch": restored_state.get("examples_into_epoch", 0) if resume else 0,
    "checkpoint_path": str(checkpoint), "checkpoint_every_epochs": boundary,
    "checkpoint_every_examples": every_examples or None,
    "checkpoint_state_path": str(final_state_path), "checkpoint_state_sha256": final_state_sha256,
    "checkpoint_epoch": epochs,
    "run_fingerprint_sha256": fingerprint,
    "restored_micro_batch_steps": restored_state["micro_batch_steps"] if resume else 0,
    "restored_optimizer_steps": restored_state["optimizer_steps"] if resume else 0,
    "restored_accumulation_micro_batches": restored_state["accumulation_micro_batches"] if resume else 0,
    "compiled_sampling_execution_cache_retired": (
        task == "grpo" and compiled_sampling_enabled
    ),
}
evaluation_report_path = pathlib.Path(recipe["artifacts"]["evaluation_report_path"])
numerical_policy_boolean_fields = (
    "fused_rms_norm_backward", "fused_gqa_attention_backward",
    "fused_linear_cross_entropy", "sparse_logits_cross_entropy",
    "bf16_tiled32_m16", "bf16_simdgroup_mm", "bf16_simdgroup_m64",
    "bf16_forward_simdgroup_m64_packed", "bf16_simdgroup_m64_prefix_tail",
    "bf16_backward_tiled32_m16", "bf16_backward_small_rows",
    "bf16_backward_simdgroup_mm", "bf16_backward_simdgroup_m64",
    "bf16_backward_simdgroup_m64_coalesced",
    "bf16_backward_simdgroup_m64_packed", "rms_norm_backward_simdgroup",
    "rms_norm_backward_residual_add", "rms_norm_generated",
    "linear_cce_f16_grad", "linear_cce_logit_cache",
    "linear_cce_f16_mps_backward", "dense_mps_linear",
    "gemma4_bf16_mlp_fusion", "gemma4_gate_up_backward_input_sum",
    "q4_0_linear_rms_add_sumsq", "eager_rank1_dot_specialization",
    "dense_device_dot_general", "lora_forward_fused_branch",
    "lora_forward_generic_rank16", "lora_forward_rank1_fused",
    "reference_quant_linear", "quant_backward_force_barriers",
    "contiguous_slice_device_view", "partition_fused_patterns",
    "partition_runtime_commands", "runtime_region_plan", "grouped_mps_dot",
    "gather_promote_input", "reduce_promote_input",
    "lora_backward_runtime_region", "low_rank_lora_backward_runtime_region",
    "rank_adapter_backward_runtime_region", "ffn_gelu_backward_runtime_region",
    "gated_gelu_backward_runtime_region", "gated_gelu_forward_fusion",
    "masked_softmax_runtime_region", "softmax_backward_runtime_region",
    "graph_rank1_dot_specialization", "raw_linear_bias_pair_runtime_region",
    "raw_linear_runtime_regions_suppressed", "gated_ffn_graph_fusion",
    "gemma_gated_mlp_training_graph_fusion",
    "attention_output_residual_graph_fusion", "grouped_lora_a_r16",
    "add3_fusion",
)
numerical_policy = {
    "schema_version": "antfly_gemma4_metal_numerical_policy/v2",
    "fingerprint_flags": 0,
    "sparse_loss_chunk_rows": 512,
    "linear_cce_tile_vocab": 65536,
    **{field: False for field in numerical_policy_boolean_fields},
}
kl_trace_path = root / "grpo_kl_control_trace.jsonl"
reward_trace_path = root / "grpo_reward_trace.jsonl"
evaluation_reward_trace_path = root / "grpo_evaluation_reward_trace.jsonl"
if task == "grpo":
    kl_trace_path.write_bytes(b"trace\n")
    reward_trace_path.write_bytes(b"reward\n")
    evaluation_reward_trace_path.write_bytes(b"evaluation reward\n")
    kl_trace_sha256 = "sha256:" + hashlib.sha256(kl_trace_path.read_bytes()).hexdigest()
    reward_trace_sha256 = "sha256:" + hashlib.sha256(reward_trace_path.read_bytes()).hexdigest()
    evaluation_reward_trace_sha256 = (
        "sha256:" + hashlib.sha256(evaluation_reward_trace_path.read_bytes()).hexdigest()
    )
final_examples = epochs * per_epoch
common = {
    "schema_version": ("antfly_inference_finetune_dpo_report/v7"
                       if task == "dpo" else "antfly_inference_finetune_grpo_report/v8"),
    "execution_mode": "train", "dataset_format": "text-preference" if task == "dpo" else "text-grpo",
    "policy_backend": "metal", "optimizer_steps": final_examples,
    "micro_batch_steps": final_examples * (2 if task == "dpo" else 4),
    "initial_logprob_parity": {"max_abs_error": 0.0}, "checkpoint_resume": checkpoint_summary,
    "metal_numerical_policy": numerical_policy,
    "evaluation_execution_policy":
        "terminal-device-drained-host-weight-snapshot-fresh-backend-private-buffer-reuse-disabled",
}
evaluation = {"report_path": str(root / f"{task}-evaluation.json"), "passed": True}
if task == "dpo":
    report = {**common, "examples": final_examples, "loss": 0.5, "mean_reward_margin": 0.1,
              "accuracy": 1.0, "beta": 0.1, "loss_type": "sigmoid",
              "logprob_aggregation": "sum", "label_smoothing": 0.0,
              "reference_mode": "frozen-base-equivalent-initial-adapter",
              "policy_scoring_mode": "compiled",
              "training_microbatch_mode": "paired", "initial_bucket_signature_parity": {"max_abs_error": 0.0},
              "sequence_length_policy": {"mode": "fixed"},
              "evaluation": {**evaluation, "examples": epochs, "loss": 0.4,
                             "mean_reward_margin": 0.1, "accuracy": 1.0,
                             "loss_type": "sigmoid", "logprob_aggregation": "sum",
                             "label_smoothing": 0.0,
                             "reference_mode": "frozen-base-equivalent-initial-adapter"}}
else:
    report = {**common, "completions": final_examples * 4,
              "tokens": final_examples * 4, "groups": final_examples,
              "training_order": {
                  "algorithm": "seeded-fisher-yates-per-epoch/v1",
                  "stream_derivation": "run-seed-order-domain-epoch-dataset-size/v1",
                  "prompt_index_semantics": "original-dataset-index",
              },
              "optimizer_groups": final_examples, "zero_reward_std_groups": 0,
              "all_truncated_groups": 0,
              "kl_rejected_groups": 0,
              "frac_reward_zero_std": 0.0, "frac_kl_rejected": 0.0,
              "truncated_completions": 0, "frac_completions_truncated": 0.0,
              "mask_truncated_completions": False,
              "loss_type": "bnpo", "scale_rewards": "group",
              "epsilon_low": 0.2, "epsilon_high": 0.2,
              "max_completion_tokens": 4, "num_iterations": 1,
              "loss": 0.5, "pg_loss": 0.5, "kl_loss": 0.0, "mean_kl": 0.0,
              "clip_fraction": 0.0, "mean_reward": 0.5, "reward_stddev": 0.5,
              "policy_rescore_completions": (final_examples * 4 if compiled_sampling_enabled else 4),
              "sampling_mode": ("compiled-shared-prompt-seeded-categorical-sparse-row-each-step"
                                if compiled_sampling_enabled else "seeded-categorical"),
              "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                           "seed_derivation": "splitmix64-run-domain-epoch-group-completion"},
              "policy_logprob_mode": (
                  "compiled-token-selection-with-eager-per-completion-canonical-logprob-rescore"
                  if compiled_sampling_enabled else "reuse"
              ),
              "training_microbatch_mode": "per-completion", "training_microbatch_batch_size": 1,
              "training_physical_micro_batches_per_group": 4, "reference_mode": "frozen",
              "kl_control": {"mode": "adaptive", "final_kl_coef": 0.04,
                             "budget_policy": "skip_group", "admitted_groups": final_examples,
                             "rejected_groups": 0,
                             "trace_path": str(kl_trace_path), "trace_digest": kl_trace_sha256},
              "reward_pipeline": {"providers": 1, "configuration_digest": "sha256:" + "e" * 64,
                                  "trace_path": str(reward_trace_path),
                                  "trace_digest": reward_trace_sha256},
              "incremental_kv": incremental_telemetry(final_examples) if incremental_enabled else None,
              "evaluation": {**evaluation, "groups": epochs, "completions": epochs * 4,
                             "zero_reward_std_groups": 0, "frac_reward_zero_std": 0.0,
                             "truncated_completions": 0, "frac_completions_truncated": 0.0,
                             "mask_truncated_completions": False,
                             "mean_reward": 0.5, "top_rank_mean_reward": 0.5,
                             "positive_reward_group_rate": 1.0, "reward_stddev": 0.5,
                             "kl_loss": 0.0, "mean_kl": 0.0, "sampling_seconds": 1.0}}
report_path.write_text(json.dumps(report))
evaluation_report = {
    "schema_version": f"antfly_inference_finetune_{task}_evaluation/v{'3' if task == 'dpo' else '4'}",
    "status": "passed",
    "dataset_path": str(recipe["dataset"]["eval_path"]),
    "policy_backend": "metal",
    "execution_policy":
        "terminal-device-drained-host-weight-snapshot-fresh-backend-private-buffer-reuse-disabled",
    "metal_numerical_policy": numerical_policy,
    "examples": epochs,
    "loss": 0.4,
    "mean_reward_margin": 0.1,
    "accuracy": 1.0,
}
if task == "grpo":
    evaluation_report.update({
        "groups": epochs,
        "completions": epochs * 4,
        "zero_reward_std_groups": 0,
        "frac_reward_zero_std": 0.0,
        "truncated_completions": 0,
        "frac_completions_truncated": 0.0,
        "mask_truncated_completions": False,
        "mean_reward": 0.5,
        "top_rank_mean_reward": 0.5,
        "positive_reward_group_rate": 1.0,
        "reward_stddev": 0.5,
        "kl_loss": 0.0,
        "mean_kl": 0.0,
        "sampling_seconds": 1.0,
        "reward_pipeline": {
            "providers": 1,
            "configuration_digest": "sha256:" + "e" * 64,
            "trace_path": str(evaluation_reward_trace_path),
            "trace_digest": evaluation_reward_trace_sha256,
        },
    })
evaluation_report_path.write_text(json.dumps(evaluation_report))
'''


class PreferenceResumeQualificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binary = self.root / "antfly"
        self.binary.write_text(FAKE_ANTFLY, encoding="utf-8")
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR)
        self.model = self.root / "model"
        self.adapter = self.root / "adapter"
        self.model.mkdir()
        self.adapter.mkdir()
        (self.model / "config.json").write_text("{}\n", encoding="utf-8")
        seed_payload = b"seed"
        (self.adapter / "adapter_model.safetensors").write_bytes(seed_payload)
        (self.adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (self.adapter / "antfly_finetune_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "antfly_gemma4_finetune/v2",
                    "status": "complete",
                    "adapter_checkpoint_sha256": hashlib.sha256(seed_payload).hexdigest(),
                    "adapter_checkpoint_size_bytes": len(seed_payload),
                }
            ),
            encoding="utf-8",
        )
        self.train = self.root / "train.jsonl"
        self.eval = self.root / "eval.jsonl"
        self.train.write_text("{}\n", encoding="utf-8")
        self.eval.write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_adapter_tree_evidence_accepts_seeded_v3_manifest(self) -> None:
        manifest_path = self.adapter / "antfly_finetune_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "antfly_gemma4_finetune/v3"
        manifest["initialization_seed"] = 17
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        evidence = qualifier._adapter_tree_evidence(self.adapter, "seed adapter")

        self.assertEqual(
            evidence["manifest_identity"]["schema_version"],
            "antfly_gemma4_finetune/v3",
        )
        self.assertEqual(evidence["manifest_identity"]["initialization_seed"], 17)

    def test_adapter_tree_evidence_enforces_schema_specific_seed_contract(self) -> None:
        manifest_path = self.adapter / "antfly_finetune_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["initialization_seed"] = 17
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(qualifier.ContractError, "must not carry"):
            qualifier._adapter_tree_evidence(self.adapter, "v2 adapter")

        manifest["schema_version"] = "antfly_gemma4_finetune/v3"
        manifest.pop("initialization_seed")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(qualifier.ContractError, "expected integer"):
            qualifier._adapter_tree_evidence(self.adapter, "v3 adapter")

    def write_recipe(self, task: str) -> Path:
        recipe = {
            "recipe": task,
            "execution": {"mode": "train"},
            "model": {"path": str(self.model), "family": "gemma4"},
            "dataset": {
                "path": str(self.train), "eval_path": str(self.eval),
                "format": "text-preference" if task == "dpo" else "text-grpo",
            },
            "adapter": {"path": str(self.adapter), "rank": 8, "alpha": 16},
            "optimizer": {"epochs": 1},
            "eval": {"path": str(self.eval)},
            "artifacts": {"root": str(self.root / "unused")},
            "backend": "metal",
        }
        if task == "grpo":
            recipe["grpo"] = {"max_completion_tokens": 4}
        path = self.root / f"{task}.recipe.json"
        path.write_text(json.dumps(recipe), encoding="utf-8")
        return path

    def args(self, task: str) -> Namespace:
        return Namespace(
            binary=self.binary,
            recipe=self.write_recipe(task),
            output_dir=self.root / f"{task}-qualification",
            epochs=2,
            interrupt_after_epoch=1,
            interrupt_after_examples=0,
            timeout_seconds=5.0,
            poll_seconds=0.01,
            direct_gguf_training=False,
            experimental_gguf_qlora=False,
            incremental_kv=False,
            incremental_kv_serial=False,
            incremental_kv_clone_prompt_tail=False,
            incremental_kv_shadow_exact=False,
            compiled_sampling=False,
            expected_final_adapter_sha256=None,
        )

    def test_numerical_policy_fields_match_zig_report_schema(self) -> None:
        recipe_source = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "finetune"
            / "recipe.zig"
        ).read_text(encoding="utf-8")
        body = recipe_source.split("const GemmaMetalNumericalPolicy = struct {", 1)[1].split(
            "\n};", 1
        )[0]
        zig_boolean_fields = tuple(
            re.findall(r"^\s+([a-z0-9_]+): bool,\s*$", body, flags=re.MULTILINE)
        )
        self.assertEqual(
            zig_boolean_fields,
            qualifier.METAL_NUMERICAL_POLICY_BOOLEAN_FIELDS,
        )
        self.assertIn("sparse_loss_chunk_rows: u32,", body)
        self.assertIn("linear_cce_tile_vocab: usize,", body)

    def test_shared_environment_policy_sanitizes_correctness_switches(self) -> None:
        self.assertEqual(
            qualifier.ENVIRONMENT_POLICY_SHA256,
            "sha256:" + hashlib.sha256(qualifier.ENVIRONMENT_POLICY_PATH.read_bytes()).hexdigest(),
        )
        inherited = {
            "PATH": "/usr/bin",
            "TERMITE_DISABLE_GRAPH_OUTPUT_OWNED_COPY": "1",
            "TERMITE_DISABLE_GRAPH_OUTPUT_ELISION_OVERRIDE": "1",
            "TERMITE_DISABLE_OUTPUT_HOST_MIRROR_RESYNC": "1",
            "TERMITE_DISABLE_PAGED_KV": "1",
            "TERMITE_METAL_DISABLE_LINEAR_CCE": "1",
            "ANTFLY_GEMMA4_GRPO_INCREMENTAL_KV": "0",
            "ANTFLY_GEMMA4_GRPO_COMPILED_SAMPLING": "1",
            "ANTFLY_GEMMA4_PREFERENCE_TRACE": "1",
            "ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA": "1",
        }
        sanitized = qualifier._strict_environment(inherited)
        self.assertEqual(sanitized["PATH"], "/usr/bin")
        for name in inherited.keys() - {"PATH"}:
            self.assertNotIn(name, sanitized)
        self.assertEqual(
            {
                name: sanitized[name]
                for name in qualifier.STRICT_METAL_ENV
            },
            qualifier.STRICT_METAL_ENV,
        )

    def test_qualification_inputs_reject_symlink_leaves(self) -> None:
        link = self.root / "antfly-link"
        link.symlink_to(self.binary)
        with self.assertRaisesRegex(qualifier.ContractError, "must not be a symlink"):
            qualifier._regular_file(link, "antfly binary", executable=True)

    def test_immutable_input_trees_reject_nested_symlinks(self) -> None:
        (self.model / "nested-link").symlink_to(self.model / "config.json")
        with self.assertRaisesRegex(qualifier.ContractError, "contains symlink"):
            qualifier._closed_immutable_path(self.model, "model")

    def test_effective_dataset_path_rejects_conflicting_aliases(self) -> None:
        self.assertEqual(
            qualifier._effective_regular_file(
                str(self.train), str(self.train), "training dataset"
            ),
            self.train.resolve(),
        )
        with self.assertRaisesRegex(qualifier.ContractError, "conflicting training"):
            qualifier._effective_regular_file(
                str(self.train), str(self.eval), "training dataset"
            )

    def test_tree_snapshot_binds_regular_file_bytes(self) -> None:
        snapshot = qualifier._tree_snapshot(self.model)
        config = next(entry for entry in snapshot if entry["path"] == "config.json")
        self.assertEqual(config["sha256"], qualifier._sha256(self.model / "config.json"))

    def test_output_root_must_not_overlap_immutable_inputs(self) -> None:
        args = self.args("dpo")
        args.output_dir = self.model / "qualification"
        with self.assertRaisesRegex(qualifier.ContractError, "overlaps immutable model"):
            qualifier.qualify(args)

    def test_final_adapter_must_differ_from_seed(self) -> None:
        seed = "sha256:" + "a" * 64
        trained = "sha256:" + "b" * 64
        self.assertEqual(qualifier._require_changed_adapter(seed, trained), trained)
        with self.assertRaisesRegex(qualifier.ContractError, "byte-identical to the seed"):
            qualifier._require_changed_adapter(seed, seed)

    def test_adapter_tree_allows_only_payload_and_manifest_to_change(self) -> None:
        seed = {
            "adapter_model.safetensors": {"sha256": "seed"},
            "adapter_config.json": {"sha256": "config"},
            "antfly_finetune_manifest.json": {"sha256": "seed-manifest"},
        }
        trained = {
            **seed,
            "adapter_model.safetensors": {"sha256": "trained"},
            "antfly_finetune_manifest.json": {"sha256": "trained-manifest"},
        }
        qualifier._require_adapter_tree_contract(seed, trained)
        trained["adapter_config.json"] = {"sha256": "drift"}
        with self.assertRaisesRegex(qualifier.ContractError, "immutable companion"):
            qualifier._require_adapter_tree_contract(seed, trained)

    def test_dpo_exact_resume_contract(self) -> None:
        report = qualifier.qualify(self.args("dpo"))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["task"], "dpo")
        self.assertEqual(report["checkpoints"]["interrupted_boundary"]["epoch_index"], 1)
        self.assertEqual(report["checkpoints"]["resumed_final"]["epoch_index"], 2)
        self.assertEqual(
            report["parity"]["training_checkpoint_sha256"],
            report["checkpoints"]["uninterrupted_final"]["sha256"],
        )
        self.assertEqual(
            report["parity"]["checkpoint_state_sha256"],
            report["checkpoints"]["uninterrupted_final"]["state_sha256"],
        )
        self.assertIn(
            "antfly_finetune_manifest.json",
            report["parity"]["adapter_tree"],
        )

    def test_grpo_exact_resume_contract_preserves_trace_digests(self) -> None:
        report = qualifier.qualify(self.args("grpo"))
        self.assertEqual(report["status"], "pass")
        semantic = report["parity"]["semantic_report"]
        self.assertEqual(
            semantic["kl_control"]["trace_digest"],
            report["parity"]["verified_artifacts"]["resumed"]["training_kl_trace"]["sha256"],
        )
        self.assertEqual(
            semantic["reward_pipeline"]["trace_digest"],
            report["parity"]["verified_artifacts"]["resumed"]["training_reward_trace"]["sha256"],
        )
        self.assertEqual(
            report["parity"]["terminal_metal_float_comparison"]["mode"],
            "exact-except-bounded-terminal-metal-grpo-kl",
        )
        self.assertEqual(
            semantic["training_order"],
            qualifier.GRPO_TRAINING_ORDER,
        )

    def test_grpo_mid_epoch_examples_resume_contract(self) -> None:
        args = self.args("grpo")
        args.epochs = 1
        args.interrupt_after_epoch = 0
        args.interrupt_after_examples = 1
        report = qualifier.qualify(args)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["contract"]["interrupt_after_examples"], 1)
        interrupted = report["checkpoints"]["interrupted_boundary"]
        self.assertEqual(interrupted["epoch_index"], 0)
        self.assertEqual(interrupted["examples_into_epoch"], 1)
        self.assertEqual(
            interrupted["state_schema_version"],
            qualifier.STATE_SCHEMA_VERSION_V2,
        )
        final = report["checkpoints"]["resumed_final"]
        self.assertEqual(final["epoch_index"], 1)
        self.assertEqual(final["examples_into_epoch"], 0)
        interrupted_recipe = json.loads(Path(report["recipes"]["interrupted"]).read_text())
        self.assertEqual(interrupted_recipe["checkpoint"]["every_examples"], 1)
        self.assertEqual(interrupted_recipe["checkpoint"]["every_epochs"], 1)
        self.assertEqual(
            report["parity"]["training_checkpoint_sha256"],
            report["checkpoints"]["uninterrupted_final"]["sha256"],
        )
        retained = report["retained_interrupted_boundary"]
        self.assertEqual(
            retained["training_checkpoint"]["sha256"],
            interrupted["sha256"],
        )
        self.assertEqual(
            retained["preference_sidecar"]["sha256"],
            interrupted["state_sha256"],
        )
        for artifact in retained.values():
            self.assertTrue(Path(artifact["path"]).is_file())
            self.assertEqual(qualifier._sha256(Path(artifact["path"])), artifact["sha256"])

    def test_dpo_mid_epoch_examples_resume_contract(self) -> None:
        args = self.args("dpo")
        args.epochs = 1
        args.interrupt_after_epoch = 0
        args.interrupt_after_examples = 1
        report = qualifier.qualify(args)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["checkpoints"]["interrupted_boundary"]["examples_into_epoch"], 1)
        self.assertEqual(report["checkpoints"]["resumed_final"]["examples_into_epoch"], 0)

    def test_mid_epoch_compositions_fail_closed(self) -> None:
        one_epoch_boundary = self.args("grpo")
        one_epoch_boundary.epochs = 1
        one_epoch_boundary.interrupt_after_epoch = 0
        with self.assertRaisesRegex(qualifier.ContractError, "between 1 and epochs-1"):
            qualifier.qualify(one_epoch_boundary)

        bad_epoch = self.args("grpo")
        bad_epoch.epochs = 1
        bad_epoch.interrupt_after_epoch = 1
        bad_epoch.interrupt_after_examples = 1
        with self.assertRaisesRegex(qualifier.ContractError, "between 0 and epochs-1"):
            qualifier.qualify(bad_epoch)

        compiled = self.args("grpo")
        compiled.epochs = 1
        compiled.interrupt_after_epoch = 0
        compiled.interrupt_after_examples = 1
        compiled.compiled_sampling = True
        with self.assertRaisesRegex(qualifier.ContractError, "eager sampling only"):
            qualifier.qualify(compiled)

        incremental = self.args("grpo")
        incremental.epochs = 1
        incremental.interrupt_after_epoch = 0
        incremental.interrupt_after_examples = 1
        incremental.incremental_kv = True
        with self.assertRaisesRegex(qualifier.ContractError, "incremental KV"):
            qualifier.qualify(incremental)

        out_of_range = self.args("grpo")
        out_of_range.epochs = 1
        out_of_range.interrupt_after_epoch = 0
        out_of_range.interrupt_after_examples = 1
        recipe = json.loads(out_of_range.recipe.read_text(encoding="utf-8"))
        recipe["dataset"]["max_examples"] = 1
        out_of_range.recipe.write_text(json.dumps(recipe), encoding="utf-8")
        with self.assertRaisesRegex(qualifier.ContractError, "must be smaller"):
            qualifier.qualify(out_of_range)

    def test_expected_final_adapter_digest_is_enforced(self) -> None:
        expected = "sha256:" + hashlib.sha256(b"identical-final-adapter").hexdigest()
        args = self.args("dpo")
        args.expected_final_adapter_sha256 = expected.removeprefix("sha256:").upper()
        report = qualifier.qualify(args)
        self.assertEqual(report["contract"]["expected_final_adapter_sha256"], expected)

        mismatch = self.args("dpo")
        mismatch.output_dir = self.root / "dpo-digest-mismatch"
        mismatch.expected_final_adapter_sha256 = "0" * 64
        with self.assertRaisesRegex(qualifier.ContractError, "does not match"):
            qualifier.qualify(mismatch)

    def test_final_report_counters_are_bound_to_checkpoint(self) -> None:
        report = {
            "optimizer_steps": 2,
            "micro_batch_steps": 8,
            "groups": 3,
        }
        checkpoint = {
            "optimizer_steps": 2,
            "micro_batch_steps": 8,
            "examples_seen": 3,
            "accumulation_micro_batches": 0,
        }
        self.assertEqual(
            qualifier._require_final_report_checkpoint_consistency(
                report, checkpoint, "grpo"
            )["groups"],
            3,
        )
        with self.assertRaisesRegex(qualifier.ContractError, "optimizer_steps"):
            qualifier._require_final_report_checkpoint_consistency(
                {**report, "optimizer_steps": 1}, checkpoint, "grpo"
            )
        with self.assertRaisesRegex(qualifier.ContractError, "groups"):
            qualifier._require_final_report_checkpoint_consistency(
                {**report, "groups": 4}, checkpoint, "grpo"
            )

    def test_checkpoint_inspection_rejects_stochastic_counter_drift(self) -> None:
        args = self.args("dpo")
        report = qualifier.qualify(args)
        checkpoint = Path(
            report["retained_interrupted_boundary"]["training_checkpoint"]["path"]
        )
        payload = bytearray(checkpoint.read_bytes())
        header_size = struct.unpack_from("<Q", payload, 0)[0]
        data_start = 8 + header_size
        stochastic_steps_field = 3
        for chunk_index in range(4):
            value = 1 if chunk_index == 0 else 0
            struct.pack_into(
                "<f",
                payload,
                data_start + (stochastic_steps_field * 4 + chunk_index) * 4,
                float(value),
            )
        checkpoint.write_bytes(payload)
        with self.assertRaisesRegex(qualifier.ContractError, "inconsistent counters"):
            qualifier.inspect_training_checkpoint(checkpoint)

    def test_grpo_incremental_kv_resume_contract_covers_whole_run(self) -> None:
        args = self.args("grpo")
        args.incremental_kv = True
        report = qualifier.qualify(args)
        telemetry = report["parity"]["semantic_report"]["incremental_kv"]
        self.assertEqual(telemetry["groups"], 2)
        self.assertEqual(telemetry["host_logit_fallbacks"], 0)
        self.assertEqual(
            telemetry,
            report["checkpoints"]["resumed_final"]["incremental_kv"],
        )
        generated = json.loads(Path(report["recipes"]["resumed"]).read_text())
        self.assertTrue(generated["runtime"]["grpo_incremental_kv"])

    def test_grpo_compiled_sampling_resume_is_explicit_attested_and_non_incremental(self) -> None:
        args = self.args("grpo")
        args.compiled_sampling = True
        recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
        recipe["runtime"] = {
            "grpo_incremental_kv": True,
            "grpo_incremental_kv_batch_active": True,
            "grpo_incremental_kv_clone_prompt_tail": True,
            "grpo_incremental_kv_shadow_exact": True,
        }
        args.recipe.write_text(json.dumps(recipe), encoding="utf-8")

        report = qualifier.qualify(args)

        self.assertTrue(report["contract"]["compiled_sampling"])
        self.assertEqual(
            report["contract"]["strict_metal_environment"][
                qualifier.COMPILED_GRPO_SAMPLING_ENV
            ],
            "1",
        )
        self.assertEqual(
            report["parity"]["semantic_report"]["sampling_mode"],
            qualifier.COMPILED_GRPO_SAMPLING_MODE,
        )
        self.assertEqual(
            report["parity"]["semantic_report"]["policy_logprob_mode"],
            qualifier.COMPILED_GRPO_POLICY_LOGPROB_MODE,
        )
        self.assertEqual(
            report["parity"]["semantic_report"]["completions"],
            report["parity"]["semantic_report"]["policy_rescore_completions"],
        )
        self.assertEqual(
            {
                "uninterrupted": True,
                "resumed": True,
            },
            report["parity"][
                "compiled_sampling_execution_cache_retirement"
            ],
        )
        generated = json.loads(Path(report["recipes"]["resumed"]).read_text())
        self.assertFalse(generated["runtime"]["grpo_incremental_kv"])
        self.assertNotIn("grpo_incremental_kv_batch_active", generated["runtime"])
        self.assertNotIn("grpo_incremental_kv_clone_prompt_tail", generated["runtime"])
        self.assertNotIn("grpo_incremental_kv_shadow_exact", generated["runtime"])

    def test_compiled_sampling_compositions_fail_closed(self) -> None:
        dpo = self.args("dpo")
        dpo.compiled_sampling = True
        with self.assertRaisesRegex(qualifier.ContractError, "valid only for GRPO"):
            qualifier.qualify(dpo)

        incremental = self.args("grpo")
        incremental.compiled_sampling = True
        incremental.incremental_kv = True
        incremental.output_dir = self.root / "compiled-incremental-conflict"
        with self.assertRaisesRegex(qualifier.ContractError, "conflicts with incremental-KV"):
            qualifier.qualify(incremental)

    def test_direct_gguf_incremental_kv_composition_fails_closed(self) -> None:
        gguf = self.root / "model.gguf"
        gguf.write_bytes(b"GGUFstub")
        args = self.args("grpo")
        recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
        recipe["model"]["path"] = str(gguf)
        args.recipe.write_text(json.dumps(recipe), encoding="utf-8")
        args.direct_gguf_training = True
        args.incremental_kv = True
        with self.assertRaisesRegex(
            qualifier.ContractError,
            "direct GGUF GRPO plus incremental KV is not qualified",
        ):
            qualifier.qualify(args)

    def test_grpo_terminal_metal_float_tolerance_is_narrow_and_fail_closed(self) -> None:
        expected = {"groups": 4, "evaluation": {"kl_loss": 1e-6, "mean_kl": 3e-5}}
        within = {"groups": 4, "evaluation": {"kl_loss": 1.9e-6, "mean_kl": 3.9e-5}}
        comparison = qualifier._compare_semantic_reports(expected, within, "grpo")
        self.assertTrue(comparison["fields"]["kl_loss"]["passed"])
        self.assertTrue(comparison["fields"]["mean_kl"]["passed"])

        outside = {"groups": 4, "evaluation": {"kl_loss": 2.1e-6, "mean_kl": 3e-5}}
        with self.assertRaisesRegex(qualifier.ContractError, "exceeds tolerance"):
            qualifier._compare_semantic_reports(expected, outside, "grpo")

        discrete_drift = {"groups": 5, "evaluation": {"kl_loss": 1e-6, "mean_kl": 3e-5}}
        with self.assertRaisesRegex(qualifier.ContractError, "semantic trajectory differs"):
            qualifier._compare_semantic_reports(expected, discrete_drift, "grpo")

    def test_final_checkpoint_and_sidecar_must_be_byte_identical(self) -> None:
        expected = {
            "sha256": "sha256:" + "a" * 64,
            "state_sha256": "sha256:" + "b" * 64,
        }
        self.assertEqual(
            qualifier._require_exact_final_checkpoint_parity(expected, dict(expected)),
            {
                "training_checkpoint_sha256": expected["sha256"],
                "checkpoint_state_sha256": expected["state_sha256"],
            },
        )
        with self.assertRaisesRegex(qualifier.ContractError, "training checkpoint"):
            qualifier._require_exact_final_checkpoint_parity(
                expected,
                {**expected, "sha256": "sha256:" + "c" * 64},
            )
        with self.assertRaisesRegex(qualifier.ContractError, "preference sidecar"):
            qualifier._require_exact_final_checkpoint_parity(
                expected,
                {**expected, "state_sha256": "sha256:" + "d" * 64},
            )

    def test_reported_trace_bytes_are_verified(self) -> None:
        args = self.args("grpo")
        qualifier.qualify(args)
        output = args.output_dir
        (output / "resumed" / "grpo_reward_trace.jsonl").write_text(
            "tampered\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(qualifier.ContractError, "digest does not match"):
            qualifier._validate_outputs(
                output / "uninterrupted",
                output / "resumed",
                "grpo",
                output / "interrupted-unpublished" / "gemma4_grpo_trainer_state.safetensors",
                1,
            )

    def test_standalone_evaluation_report_is_required(self) -> None:
        args = self.args("dpo")
        qualifier.qualify(args)
        output = args.output_dir
        (output / "resumed" / "dpo-evaluation.json").unlink()
        with self.assertRaises((qualifier.ContractError, FileNotFoundError)):
            qualifier._validate_outputs(
                output / "uninterrupted",
                output / "resumed",
                "dpo",
                output / "interrupted-unpublished" / "gemma4_dpo_trainer_state.safetensors",
                1,
            )

    def test_existing_output_fails_closed(self) -> None:
        args = self.args("dpo")
        args.output_dir.mkdir()
        with self.assertRaisesRegex(qualifier.ContractError, "already exists"):
            qualifier.qualify(args)


if __name__ == "__main__":
    unittest.main()
