#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gemma4_cuda_l4_release_gate import (
    CAPTURE_KV_CAPACITY,
    DEFAULT_MIN_COMPARABLE_RATIO,
    E2B_ANTFLY_TOKENS,
    E2B_LLAMA_TOKENS,
    FROZEN_PROFILE,
    GEMMA12B_TOKENS,
    FORBIDDEN_GENERATED_Q4_0_ROUTE_COUNTERS,
    MAX_TOK_S_CV,
    RELEASE_SCHEMA,
    RELEASE_SCOPE,
    REQUIRED_Q8_PREFILL_ROUTE_COUNTERS,
    TOKEN_IDS_RE,
    benchmark_contract,
    canonical_sha256,
    disabled_candidate_errors,
    diagnostic_mode_errors,
    e2b_contract_errors,
    e2b_pair_contract_errors,
    frozen_profile,
    gemma12b_evidence,
    generation_replay_errors,
    git_provenance,
    gpu_provenance,
    l4_errors,
    matrix_command,
    matrix_process_errors,
    matrix_timeout_sec,
    parse_token_ids,
    path_provenance,
    provenance_binding,
    provenance_errors,
    release_environment,
    reset_matrix_outputs,
    run_logged,
    toolchain_provenance,
)


def matrix(ratio: float = 0.81, passed: bool = True) -> dict:
    return {
        "entries": [{
            "output_tokens": E2B_LLAMA_TOKENS,
            "pair_ok": True,
            "graph_replay_ok": True,
            "comparable_ratio": ratio,
        }],
        "passed": passed,
    }


def pair() -> dict:
    return {
        "ok": True,
        "ok_graph_replay": True,
        "ok_lm_head_argmax": True,
        "comparison": {
            "antfly_tokens": E2B_ANTFLY_TOKENS,
            "llama_tokens": E2B_LLAMA_TOKENS,
            "antfly_cache_dtype": "f32",
            "llama_cache_type_k": "f32",
            "llama_cache_type_v": "f32",
            "antfly_generated_attention_decode": "0",
            "antfly_generated_q6_k_q8_1_lm_head_argmax": "0",
            "antfly_generated_q4_0_e2b_ffn": "0",
            "antfly_generated_q4_0_e2b_ffn_exact": "0",
            "antfly_q4_0_q8_1_lm_head_argmax": "1",
        },
        "rows": [{
            "antfly_generated_q6_lm_head_argmax": 0,
            "antfly_generated_q6_lm_head_argmax_fallbacks": 0,
            "antfly_generated_q4_0_mmv": 0,
            "antfly_generated_q4_0_mmv_fallbacks": 0,
            "antfly_generated_q4_0_mm": 0,
            "antfly_generated_q4_0_mm_fallbacks": 0,
            "antfly_generated_q4_0_pair": 0,
            "antfly_generated_q4_0_pair_fallbacks": 0,
            "antfly_generated_q4_0_pair_q8": 0,
            "antfly_generated_q4_0_pair_q8_fallbacks": 0,
            "antfly_generated_q4_0_down_q8": 0,
            "antfly_generated_q4_0_down_q8_fallbacks": 0,
            "antfly_q8_1_prefill_linear": 1,
            "antfly_q8_1_prefill_pair": 1,
            "antfly_generated_e2b_pair": 0,
            "antfly_generated_e2b_down": 0,
            "antfly_generated_e2b_pair_fallbacks": 0,
            "antfly_generated_e2b_down_fallbacks": 0,
            "antfly_generated_e2b_exact_pair": 0,
            "antfly_generated_e2b_exact_down": 0,
            "antfly_generated_e2b_exact_pair_fallbacks": 0,
            "antfly_generated_e2b_exact_down_fallbacks": 0,
            "antfly_generated_attention": 0,
        }],
    }


def generation_run() -> dict:
    return {
        "returncode": 0,
        "token_ids": list(range(GEMMA12B_TOKENS)),
        "timing_data": {
            "tokens": GEMMA12B_TOKENS,
            "decode_tok_per_s": 42.0,
            "cuda": {
                "graph_capture_persistent_replays": GEMMA12B_TOKENS - 1,
                "graph_capture_discards": 0,
                "graph_capture_capacity_skips": 0,
                "launch_attention_gqa_decode_generated": 0,
                "lm_head_argmax_generated_q6_k_q8_1_hits": 0,
                "lm_head_argmax_generated_q6_k_q8_1_fallbacks": 0,
                "q4_0_generated_e2b_pair_q8_hits": 0,
                "q4_0_generated_e2b_down_q8_hits": 0,
                "q4_0_generated_e2b_pair_q8_fallbacks": 0,
                "q4_0_generated_e2b_down_q8_fallbacks": 0,
                "q4_0_generated_e2b_exact_pair_f32_hits": 0,
                "q4_0_generated_e2b_exact_down_f32_hits": 0,
                "q4_0_generated_e2b_exact_pair_f32_fallbacks": 0,
                "q4_0_generated_e2b_exact_down_f32_fallbacks": 0,
            },
        },
    }


class L4ReleaseGateTest(unittest.TestCase):
    def test_release_schema_versions_bound_provenance(self) -> None:
        self.assertEqual("antfly.gemma4_cuda_l4_release_gate.v2", RELEASE_SCHEMA)

    def test_release_scope_is_target_only(self) -> None:
        self.assertEqual("target_only", RELEASE_SCOPE)

    def test_workflow_runs_mtp_only_as_non_gating_nightly_diagnostics(self) -> None:
        workflow = (pathlib.Path(__file__).resolve().parents[4] / ".github/workflows/cuda-gemma4-l4.yml").read_text(encoding="utf-8")
        start = workflow.index("- name: Collect experimental fixed-corpus MTP diagnostics")
        end = workflow.index("- name: Publish evidence summary", start)
        mtp_step = workflow[start:end]
        self.assertIn("if: ${{ (inputs.gate || 'nightly') == 'nightly' }}", mtp_step)
        self.assertIn("continue-on-error: true", mtp_step)
        missing_draft = mtp_step[mtp_step.index('if [[ ! -f "$MTP_DRAFT_MODEL" ]]'):mtp_step.index("exit 0")]
        self.assertIn("diagnostic_status=skipped_missing_draft", missing_draft)
        self.assertIn('> "$mtp_dir/mtp_collection_profile.txt"', missing_draft)
        self.assertIn("release_contract=none; experimental diagnostic only", mtp_step)

    def test_cuda_evidence_workflow_sets_runner_temp_caches_at_runtime(self) -> None:
        workflow = (pathlib.Path(__file__).resolve().parents[4] / ".github/workflows/cuda-gemma4-l4.yml").read_text(encoding="utf-8")
        prepare = workflow[workflow.index("- name: Prepare CUDA evidence inputs"):workflow.index("- name: Check generated CUDA sources")]
        self.assertNotIn("${{ runner.temp }}", workflow)
        self.assertIn('ZIG_LOCAL_CACHE_DIR="$RUNNER_TEMP/antfly-zig-local"', prepare)
        self.assertIn('ZIG_GLOBAL_CACHE_DIR="$RUNNER_TEMP/antfly-zig-global"', prepare)
        self.assertIn('>> "$GITHUB_ENV"', prepare)

    def test_release_publication_does_not_depend_on_unpublished_cuda_artifact(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[4]
        workflow = (repo / ".github/workflows/cuda-gemma4-l4.yml").read_text(encoding="utf-8")
        release = (repo / ".github/workflows/antfly-release.yml").read_text(encoding="utf-8")
        archive_builder = (repo / "scripts/packaging/build_zig_release_archive.sh").read_text(encoding="utf-8")
        self.assertIn("schedule:", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("uses: ./.github/workflows/cuda-gemma4-l4.yml", release)
        self.assertIn("-Dcuda=false", archive_builder)
        publish = release[release.index("  publish-release-assets:"):release.index("  package-cli-artifacts:")]
        self.assertNotIn("cuda-gemma4-release-gate", publish)

    def test_workflow_requires_dedicated_runner_labels(self) -> None:
        workflow = (pathlib.Path(__file__).resolve().parents[4] / ".github/workflows/cuda-gemma4-l4.yml").read_text(encoding="utf-8")
        configure = workflow[workflow.index("  configure:"):workflow.index("  l4-evidence:")]
        evidence = workflow[workflow.index("  l4-evidence:"):]
        self.assertIn("ANTFLY_CUDA_L4_RUNNER_LABELS_JSON must be configured for release gates", configure)
        self.assertIn("a dedicated L4 runner label", configure)
        self.assertIn("enabled=false", configure)
        self.assertIn("if: ${{ needs.configure.outputs.enabled == 'true' }}", evidence)
        self.assertNotIn("|| '[\"self-hosted\"]'", evidence)

    def test_workflow_uses_accepted_release_and_batching_regression_floors(self) -> None:
        workflow = (pathlib.Path(__file__).resolve().parents[4] / ".github/workflows/cuda-gemma4-l4.yml").read_text(encoding="utf-8")
        self.assertIn("--enforce-performance --min-comparable-ratio 0.70 --verify-artifacts", workflow)
        batching = workflow[workflow.index("- name: Validate Polar4 server batching"):workflow.index("- name: Collect fixed E2B", workflow.index("- name: Validate Polar4 server batching"))]
        self.assertIn("--min-c2-speedup 0.40", batching)
        self.assertNotIn('if [[ "$CUDA_RELEASE_MODE" == "nightly" ]]', batching)

    def test_profile_locks_candidate_gates_and_returns_a_copy(self) -> None:
        profile = frozen_profile()
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SCORE_PREWORK"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION"])
        self.assertEqual("853", profile["ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD"])
        self.assertEqual("2500", profile["ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_CATALOG_FFN_CANDIDATES"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_MMV"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_MM"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_PAIR"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_PAIR_Q8"])
        self.assertEqual("1", profile["ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_DP4A"])
        self.assertEqual("1", profile["ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_Q8_1_DP4A"])
        self.assertEqual("0", profile["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_DOWN_Q8"])
        self.assertEqual("1", profile["ANTFLY_INFERENCE_CUDA_Q4_0_LM_HEAD_Q8_1_ARGMAX"])
        self.assertEqual("required", profile["ANTFLY_DECODE_GRAPH_REPLAY"])
        self.assertEqual(str(CAPTURE_KV_CAPACITY), profile["ANTFLY_CAPTURE_FORCE_KV_CAPACITY"])
        self.assertEqual("1", profile["ANTFLY_INFERENCE_DISABLE_CONTINUOUS_BATCHING"])
        profile["ANTFLY_DECODE_GRAPH_REPLAY"] = "off"
        self.assertEqual("required", FROZEN_PROFILE["ANTFLY_DECODE_GRAPH_REPLAY"])

    def test_release_environment_discards_inherited_experiment_switches(self) -> None:
        args = argparse.Namespace(
            binary=pathlib.Path("/binary"),
            llama_cpp_bin=pathlib.Path("/llama"),
            e2b_model=pathlib.Path("/model.gguf"),
            timeout_sec=123,
        )
        with mock.patch.dict(os.environ, {
            "CUDA_VISIBLE_DEVICES": "0",
            "ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX": "1",
            "ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT": "1",
            "ANTFLY_EXPERIMENTAL_SWITCH": "1",
            "REQUIRE_LM_HEAD_ARGMAX": "0",
            "LLAMA_CACHE_TYPE_K": "q8_0",
        }, clear=True):
            environment = release_environment(args)
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX"])
        self.assertEqual("0", environment["ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT"])
        self.assertEqual("1", environment["REQUIRE_LM_HEAD_ARGMAX"])
        self.assertEqual("f32", environment["LLAMA_CACHE_TYPE_K"])
        self.assertNotIn("ANTFLY_EXPERIMENTAL_SWITCH", environment)
        self.assertEqual("0", environment["CUDA_VISIBLE_DEVICES"])
        self.assertEqual("123", environment["TIMEOUT"])

    def test_matrix_ratio_is_only_a_failure_when_enforced(self) -> None:
        self.assertEqual(0.70, DEFAULT_MIN_COMPARABLE_RATIO)
        self.assertEqual([], e2b_contract_errors(matrix(0.69, passed=False), False, DEFAULT_MIN_COMPARABLE_RATIO))
        self.assertEqual([], e2b_contract_errors(matrix(0.70), True, DEFAULT_MIN_COMPARABLE_RATIO))
        errors = e2b_contract_errors(matrix(0.69, passed=False), True, DEFAULT_MIN_COMPARABLE_RATIO)
        self.assertTrue(any("below required 0.700" in error for error in errors))
        self.assertTrue(any("did not pass" in error for error in errors))

    def test_benchmark_provenance_binds_performance_and_route_contract(self) -> None:
        contract = benchmark_contract(argparse.Namespace(
            enforce_performance=True,
            min_comparable_ratio=DEFAULT_MIN_COMPARABLE_RATIO,
        ))
        self.assertTrue(contract["performance_enforced"])
        self.assertEqual(0.70, contract["min_comparable_ratio"])
        self.assertEqual(list(REQUIRED_Q8_PREFILL_ROUTE_COUNTERS), contract["required_positive_route_counters"])
        self.assertEqual(list(FORBIDDEN_GENERATED_Q4_0_ROUTE_COUNTERS), contract["required_zero_route_counters"])

    def test_matrix_contract_rejects_replay_or_pair_failures(self) -> None:
        bad = matrix()
        bad["entries"][0]["pair_ok"] = False
        bad["entries"][0]["graph_replay_ok"] = False
        errors = e2b_contract_errors(bad, False, DEFAULT_MIN_COMPARABLE_RATIO)
        self.assertTrue(any("paired benchmark" in error for error in errors))
        self.assertTrue(any("persistent graph replay" in error for error in errors))

    def test_pair_contract_rejects_generated_q6_and_profile_drift(self) -> None:
        self.assertEqual([], e2b_pair_contract_errors(pair()))
        bad = pair()
        bad["comparison"]["antfly_generated_q6_k_q8_1_lm_head_argmax"] = "1"
        bad["rows"][0]["antfly_generated_q6_lm_head_argmax"] = 1
        errors = e2b_pair_contract_errors(bad)
        self.assertTrue(any("profile drifted" in error for error in errors))
        self.assertTrue(any("generated Q6 LM-head" in error for error in errors))

    def test_pair_contract_rejects_exact_ffn_and_profile_drift(self) -> None:
        bad = pair()
        bad["comparison"]["antfly_generated_q4_0_e2b_ffn_exact"] = "1"
        bad["rows"][0]["antfly_generated_e2b_exact_pair"] = 1
        errors = e2b_pair_contract_errors(bad)
        self.assertTrue(any("antfly_generated_q4_0_e2b_ffn_exact" in error for error in errors))
        self.assertTrue(any("antfly_generated_e2b_exact_pair" in error for error in errors))

    def test_pair_contract_requires_positive_q8_prefill_routes(self) -> None:
        for key in REQUIRED_Q8_PREFILL_ROUTE_COUNTERS:
            with self.subTest(key=key):
                bad = pair()
                bad["rows"][0][key] = 0
                errors = e2b_pair_contract_errors(bad)
                self.assertTrue(any(key in error for error in errors))

    def test_pair_contract_rejects_generated_q4_route_hits_and_fallbacks(self) -> None:
        for key in FORBIDDEN_GENERATED_Q4_0_ROUTE_COUNTERS:
            with self.subTest(key=key):
                bad = pair()
                bad["rows"][0][key] = 1
                errors = e2b_pair_contract_errors(bad)
                self.assertTrue(any(key in error for error in errors))

    def test_pair_contract_rejects_missing_or_malformed_route_counters(self) -> None:
        counters = REQUIRED_Q8_PREFILL_ROUTE_COUNTERS + FORBIDDEN_GENERATED_Q4_0_ROUTE_COUNTERS
        for key in counters:
            with self.subTest(key=key, case="missing"):
                bad = pair()
                del bad["rows"][0][key]
                errors = e2b_pair_contract_errors(bad)
                self.assertTrue(any(f"missing route counter {key}" in error for error in errors))
            with self.subTest(key=key, case="malformed"):
                bad = pair()
                bad["rows"][0][key] = "1" if key in REQUIRED_Q8_PREFILL_ROUTE_COUNTERS else "0"
                errors = e2b_pair_contract_errors(bad)
                self.assertTrue(any(f"invalid route counter {key}" in error for error in errors))

    def test_12b_replay_contract_requires_exact_tokens_and_disabled_candidates(self) -> None:
        run = generation_run()
        self.assertEqual([], generation_replay_errors(run, GEMMA12B_TOKENS, "12B"))
        run["timing_data"]["cuda"]["lm_head_argmax_generated_q6_k_q8_1_hits"] = 1
        errors = generation_replay_errors(run, GEMMA12B_TOKENS, "12B")
        self.assertTrue(any("generated_q6" in error for error in errors))

    def test_12b_evidence_detects_nondeterministic_token_ids(self) -> None:
        first = generation_run()
        second = generation_run()
        second["token_ids"][17] = 999
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                output_dir=pathlib.Path(temp_dir),
                gemma12b_q4_model=pathlib.Path("/model.gguf"),
            )
            with mock.patch(
                "gemma4_cuda_l4_release_gate.run_generation_case",
                side_effect=[first, second],
            ):
                evidence = gemma12b_evidence(args, {}, {"path": "/model.gguf", "exists": True})
        self.assertFalse(evidence["passed"])
        self.assertFalse(evidence["checks"]["deterministic_tokens"])
        self.assertTrue(any("differ at index 17" in error for error in evidence["errors"]))

    def test_disabled_candidate_counter_errors_are_specific(self) -> None:
        timing = {"cuda": {"launch_attention_gqa_decode_generated": 2}}
        errors = disabled_candidate_errors(timing, "run")
        self.assertEqual(["run unexpectedly used disabled candidate counter launch_attention_gqa_decode_generated"], errors)

        timing = {"cuda": {"launch_attention_gqa_decode_score_prework": 2}}
        errors = disabled_candidate_errors(timing, "run")
        self.assertEqual(
            ["run unexpectedly used disabled candidate counter launch_attention_gqa_decode_score_prework"],
            errors,
        )

        timing = {"cuda": {"q4_0_generated_e2b_exact_pair_f32_hits": 2}}
        errors = disabled_candidate_errors(timing, "run")
        self.assertEqual(
            ["run unexpectedly used disabled candidate counter q4_0_generated_e2b_exact_pair_f32_hits"],
            errors,
        )

    def test_l4_requirement_is_exact(self) -> None:
        l4 = {"name": "NVIDIA L4", "compute_capability": "8.9", "driver_version": "580.159.03"}
        self.assertEqual([], l4_errors({"devices": [l4]}))
        self.assertIn("expected NVIDIA L4", l4_errors({"devices": [{**l4, "name": "NVIDIA A10"}]})[0])
        self.assertIn("driver version", l4_errors({"devices": [{**l4, "driver_version": ""}]})[0])
        self.assertIn("expected exactly one", l4_errors({"devices": []})[0])

    def test_git_provenance_separates_tracked_and_source_untracked_drift(self) -> None:
        clean = {"returncode": 0, "stdout": None, "stderr": None}
        captures = [
            {**clean, "stdout": "a" * 40},
            {**clean, "stdout": "a" * 40},
            clean,
            {**clean, "stdout": "?? zig/pkg/inference/new_source.zig"},
            {**clean, "stdout": "?? artifacts/cuda-gemma4-l4/release_summary.json"},
        ]
        with mock.patch("gemma4_cuda_l4_release_gate.command_capture", side_effect=captures):
            provenance = git_provenance(pathlib.Path("/repo"))
        self.assertIs(provenance["tracked_dirty"], False)
        self.assertIs(provenance["dirty"], True)
        self.assertEqual(["zig/pkg/inference/new_source.zig"], provenance["source_untracked_paths"])

    def test_provenance_errors_fail_closed_on_git_absence_and_source_drift(self) -> None:
        tool = {"returncode": 0, "path": "/tool", "sha256": "f" * 64, "version": "ok"}
        toolchains = {name: dict(tool) for name in ("python", "git", "cuobjdump", "nvidia_smi")}
        toolchains["zig"] = {**tool, "version": "0.16.0"}
        toolchains["nvcc"] = {**tool, "version": "Cuda compilation tools, release 13.2"}
        base = {
            "git": {
                "commit": "a" * 40,
                "commit_returncode": 0,
                "tracked_status_returncode": 0,
                "tracked_dirty": False,
                "source_status_returncode": 0,
                "source_untracked_paths": [],
            },
            "toolchains": toolchains,
        }
        self.assertEqual([], provenance_errors(base))

        missing = {"git": {}, "toolchains": toolchains}
        errors = provenance_errors(missing)
        self.assertTrue(any("commit provenance" in error for error in errors))
        self.assertTrue(any("tracked-source status" in error for error in errors))
        self.assertTrue(any("untracked-source status" in error for error in errors))

        drift = {"git": {**base["git"], "tracked_dirty": True, "source_untracked_paths": ["zig/new.zig"]}, "toolchains": toolchains}
        errors = provenance_errors(drift)
        self.assertTrue(any("tracked source differs" in error for error in errors))
        self.assertTrue(any("untracked source files" in error for error in errors))

    def test_toolchain_provenance_shape_is_deterministic_on_missing_tools(self) -> None:
        with mock.patch("gemma4_cuda_l4_release_gate.shutil.which", return_value=None), \
                mock.patch("gemma4_cuda_l4_release_gate.pathlib.Path.is_file", return_value=False):
            provenance = toolchain_provenance()
        self.assertEqual(
            {"python", "git", "zig", "nvcc", "cuobjdump", "nvidia_smi", "platform"},
            set(provenance),
        )
        for name in ("python", "git", "zig", "nvcc", "cuobjdump", "nvidia_smi"):
            self.assertEqual({"command", "path", "sha256", "version", "returncode"}, set(provenance[name]))
            self.assertIsNone(provenance[name]["path"])
        errors = provenance_errors({
            "git": {
                "commit": "a" * 40,
                "commit_returncode": 0,
                "tracked_status_returncode": 0,
                "tracked_dirty": False,
                "source_status_returncode": 0,
                "source_untracked_paths": [],
            },
            "toolchains": provenance,
        })
        self.assertTrue(any("nvcc toolchain provenance is unavailable" in error for error in errors))
        self.assertTrue(any("cuobjdump toolchain provenance is unavailable" in error for error in errors))

    def test_provenance_binding_hashes_exact_written_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "release_provenance.json"
            path.write_bytes(b'{"schema":"test"}\n')
            binding = provenance_binding(path)
        self.assertEqual("release_provenance.json", binding["provenance"])
        self.assertEqual(hashlib.sha256(b'{"schema":"test"}\n').hexdigest(), binding["provenance_sha256"])

    def test_diagnostic_bypasses_cannot_pass_release_summary(self) -> None:
        self.assertEqual([], diagnostic_mode_errors(True, False))
        self.assertEqual(1, len(diagnostic_mode_errors(False, False)))
        self.assertEqual(1, len(diagnostic_mode_errors(True, True)))
        errors = diagnostic_mode_errors(False, True)
        self.assertEqual(2, len(errors))
        self.assertTrue(any("--no-require-l4" in error for error in errors))
        self.assertTrue(any("--skip-12b" in error for error in errors))

    def test_gpu_provenance_honors_numeric_cuda_visible_devices(self) -> None:
        output = "0, NVIDIA L4, 550.54, 8.9\n1, NVIDIA L4, 550.54, 8.9\n"
        with mock.patch("gemma4_cuda_l4_release_gate.command_output", return_value=output), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=True):
            provenance = gpu_provenance()
        self.assertEqual(["1"], [device["index"] for device in provenance["devices"]])
        self.assertEqual([], l4_errors(provenance))

    def test_matrix_command_freezes_256_token_contract(self) -> None:
        args = argparse.Namespace(
            matrix_script=pathlib.Path("/matrix.py"),
            output_dir=pathlib.Path("/tmp/release"),
            warmups=1,
            repeats=3,
            enforce_performance=True,
            min_comparable_ratio=DEFAULT_MIN_COMPARABLE_RATIO,
        )
        command = matrix_command(args)
        self.assertIn("--lengths", command)
        self.assertEqual(str(E2B_LLAMA_TOKENS), command[command.index("--lengths") + 1])
        self.assertEqual(str(E2B_LLAMA_TOKENS), command[command.index("--target-length") + 1])
        self.assertEqual("0.7", command[command.index("--min-comparable-ratio") + 1])
        self.assertIn("--no-require-generated-attention", command)
        self.assertIn("--no-require-generated-q6-lm-head-argmax", command)
        self.assertIn("--collect-only", command)
        self.assertEqual(str(MAX_TOK_S_CV), command[command.index("--max-cv") + 1])

    def test_matrix_outputs_are_removed_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            matrix_path = output_dir / "e2b/matrix_summary.json"
            pair_path = output_dir / f"e2b/tokens-{E2B_LLAMA_TOKENS}/paired_summary.json"
            matrix_path.parent.mkdir(parents=True)
            pair_path.parent.mkdir(parents=True)
            matrix_path.write_text("stale")
            pair_path.write_text("stale")

            self.assertEqual((matrix_path, pair_path), reset_matrix_outputs(output_dir))
            self.assertFalse(matrix_path.exists())
            self.assertFalse(pair_path.exists())

    def test_nonzero_matrix_process_is_one_infrastructure_error(self) -> None:
        self.assertEqual([], matrix_process_errors(0))
        self.assertEqual(["E2B matrix exited 17"], matrix_process_errors(17))

    def test_matrix_timeout_covers_every_paired_command(self) -> None:
        args = argparse.Namespace(timeout_sec=123, warmups=1, repeats=3)
        self.assertEqual(123 * 9, matrix_timeout_sec(args))

    def test_run_logged_terminates_the_process_group_on_timeout(self) -> None:
        process = mock.Mock(pid=4321, returncode=-signal.SIGTERM)
        process.communicate.side_effect = [subprocess.TimeoutExpired(["slow"], 1), ("stopped", None)]
        with tempfile.TemporaryDirectory() as temp_dir:
            log = pathlib.Path(temp_dir) / "command.log"
            with mock.patch("gemma4_cuda_l4_release_gate.subprocess.Popen", return_value=process) as popen, \
                    mock.patch("gemma4_cuda_l4_release_gate.os.killpg") as killpg:
                returncode, output = run_logged(["slow"], {}, log, 1)
            self.assertIn("timed out", log.read_text())
        self.assertEqual(124, returncode)
        self.assertIn("timed out", output)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(4321, signal.SIGTERM)

    def test_token_parsing_and_provenance_fingerprint_are_stable(self) -> None:
        self.assertIsNotNone(TOKEN_IDS_RE.search("token_ids: 1 2 3\n"))
        self.assertEqual([1, 2, 3], parse_token_ids("before\ntoken_ids: 1 2 3\nafter\n"))
        self.assertEqual([], parse_token_ids("token_ids=unavailable\n"))
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}), canonical_sha256({"b": 2, "a": 1}))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "artifact.bin"
            path.write_bytes(b"artifact")
            provenance = path_provenance(path)
        self.assertTrue(provenance["exists"])
        self.assertEqual(hashlib.sha256(b"artifact").hexdigest(), provenance["sha256"])


if __name__ == "__main__":
    unittest.main()
