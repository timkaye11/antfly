#!/usr/bin/env python3

import pathlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from validate_gemma4_cuda_candidate import (
    CAPTURE_KV_CAPACITY_ENV,
    CANDIDATE_CATALOG,
    CANDIDATE_SPECS,
    DEFAULT_CAPTURE_KV_PROMPT_HEADROOM,
    EXACT_E2B_FFN_F32_COMPARISON_ENVIRONMENT,
    GENERATED_ATTENTION_KERNEL_ID,
    GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV,
    GENERATED_ATTENTION_SPLIT_KV_SPLITS_ENV,
    Q6_K_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID,
    Q4_0_E2B_FFN_EXACT_KERNEL_ID,
    Q4_0_Q8_1_FFN_KERNEL_ID,
    RUNTIME_CAPTURE_KV_CAPACITY_ENV,
    SCORE_PREWORK_ATTENTION_COMPARISON_ENVIRONMENT,
    SCORE_PREWORK_ATTENTION_KERNEL_ID,
    TOKEN_IDS_RE,
    CandidateKind,
    CandidateSpec,
    RouteCounter,
    TimingMetadata,
    candidate_environment_metadata,
    candidate_metadata,
    candidate_spec,
    case_promotion_errors,
    coefficient_of_variation,
    configure_candidate_environment,
    execution_order,
    paired_throughput,
    parse_args,
    parse_candidate_environment,
    parse_candidate_environment_list,
    parse_common_environment,
    parse_common_environment_list,
    parse_route_counter_list,
    repetition_stem,
    resolve_candidate_spec,
    resolve_candidate_environment,
    resolve_capture_kv_capacity,
    resolve_common_environment,
    result_config_metadata,
    run_case,
    summarize_case,
    summarize_throughput,
    timing_counter,
    timing_metadata,
    timing_metadata_from_args,
    timing_throughput,
    validate_pair,
)


def timing(
    *,
    generated: int = 0,
    score_prework: int = 0,
    lm_argmax: int = 0,
    lm_argmax_fallbacks: int = 0,
    generated_q6_lm_argmax: int = 0,
    generated_q6_lm_argmax_fallbacks: int = 0,
    ffn_pair: int = 0,
    ffn_down: int = 0,
    ffn_pair_fallbacks: int = 0,
    ffn_down_fallbacks: int = 0,
    exact_ffn_pair: int = 0,
    exact_ffn_down: int = 0,
    exact_ffn_pair_fallbacks: int = 0,
    exact_ffn_down_fallbacks: int = 0,
    tokens: int = 64,
    tok_s: float = 100.0,
) -> dict:
    return {
        "token_ids": list(range(tokens)),
        "decode_tok_per_s": tok_s,
        "cuda": {
            "graph_capture_persistent_replays": tokens - 4,
            "graph_capture_discards": 0,
            "graph_capture_capacity_skips": 0,
            "launch_attention_gqa_decode_generated": generated,
            "launch_attention_gqa_decode_score_prework": score_prework,
            "lm_head_argmax_fused_q4_0_q8_1": lm_argmax,
            "lm_head_argmax_q4_0_q8_1_fallbacks": lm_argmax_fallbacks,
            "lm_head_argmax_generated_q6_k_q8_1_hits": generated_q6_lm_argmax,
            "lm_head_argmax_generated_q6_k_q8_1_fallbacks": generated_q6_lm_argmax_fallbacks,
            "q4_0_generated_e2b_pair_q8_hits": ffn_pair,
            "q4_0_generated_e2b_down_q8_hits": ffn_down,
            "q4_0_generated_e2b_pair_q8_fallbacks": ffn_pair_fallbacks,
            "q4_0_generated_e2b_down_q8_fallbacks": ffn_down_fallbacks,
            "q4_0_generated_e2b_exact_pair_f32_hits": exact_ffn_pair,
            "q4_0_generated_e2b_exact_down_f32_hits": exact_ffn_down,
            "q4_0_generated_e2b_exact_pair_f32_fallbacks": exact_ffn_pair_fallbacks,
            "q4_0_generated_e2b_exact_down_f32_fallbacks": exact_ffn_down_fallbacks,
        },
    }


class CandidateParityTest(unittest.TestCase):
    def test_default_candidate_remains_generated_attention(self):
        with mock.patch.object(sys, "argv", ["validate_gemma4_cuda_candidate.py"]):
            args = parse_args()
            self.assertEqual(CandidateKind.GENERATED_ATTENTION, args.candidate)
            self.assertEqual(1, args.repeats)
            self.assertEqual(0.0, args.min_candidate_ratio)
            self.assertEqual(1.0, args.max_cv)
            self.assertIsNone(args.kernel_id)
            self.assertEqual("default", args.config_label)
            self.assertEqual("f32", args.cache_dtype)
            self.assertEqual([], args.candidate_env)
            self.assertEqual([], args.common_env)
            self.assertIsNone(args.capture_kv_capacity)

    def test_cli_accepts_repeatable_candidate_environment_and_capture_capacity(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_gemma4_cuda_candidate.py",
                "--candidate-env",
                "ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS=128",
                "--candidate-env",
                "ANTFLY_TEST_SCHEDULE=split=8",
                "--capture-kv-capacity",
                "4096",
            ],
        ):
            args = parse_args()
        self.assertEqual(
            (
                (GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV, "128"),
                ("ANTFLY_TEST_SCHEDULE", "split=8"),
            ),
            resolve_candidate_environment(args, resolve_candidate_spec(args)),
        )
        self.assertEqual((4096, "explicit"), resolve_capture_kv_capacity(args))

    def test_cli_records_non_default_cache_dtype(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--cache-dtype", "polar4"],
        ):
            args = parse_args()
        self.assertEqual("polar4", args.cache_dtype)
        self.assertEqual(
            "polar4",
            result_config_metadata(args, timing_metadata_from_args(args))["cache_dtype"],
        )

    def test_cli_accepts_repeatable_common_environment(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_gemma4_cuda_candidate.py",
                "--common-env",
                "ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING=0",
                "--common-env",
                "ANTFLY_TEST_READBACK_MODE=device=handoff",
            ],
        ):
            args = parse_args()
        spec = resolve_candidate_spec(args)
        candidate_environment = resolve_candidate_environment(args, spec)
        self.assertEqual(
            (
                ("ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING", "0"),
                ("ANTFLY_TEST_READBACK_MODE", "device=handoff"),
            ),
            resolve_common_environment(args, spec, candidate_environment),
        )

    def test_exact_e2b_ffn_pins_the_f32_comparison_profile(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--candidate", "q4-0-e2b-ffn-exact"],
        ):
            args = parse_args()
        spec = resolve_candidate_spec(args)
        self.assertEqual(
            EXACT_E2B_FFN_F32_COMPARISON_ENVIRONMENT,
            resolve_common_environment(args, spec, resolve_candidate_environment(args, spec)),
        )

    def test_score_prework_catalog_pins_the_paged_f32_value_comparison_profile(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--kernel-id", SCORE_PREWORK_ATTENTION_KERNEL_ID],
        ):
            args = parse_args()
        spec = resolve_candidate_spec(args)
        self.assertEqual(CANDIDATE_CATALOG[SCORE_PREWORK_ATTENTION_KERNEL_ID], spec)
        self.assertEqual(
            SCORE_PREWORK_ATTENTION_COMPARISON_ENVIRONMENT,
            resolve_common_environment(args, spec, resolve_candidate_environment(args, spec)),
        )
        self.assertEqual(
            "launch_attention_gqa_decode_score_prework",
            spec.route_counter,
        )

    def test_exact_e2b_ffn_rejects_overrides_of_the_f32_comparison_profile(self):
        locked_name = "ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_ACTIVATION_Q8_1_DP4A"
        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_gemma4_cuda_candidate.py",
                "--candidate",
                "q4-0-e2b-ffn-exact",
                "--candidate-env",
                f"{locked_name}=1",
            ],
        ):
            args = parse_args()
        spec = resolve_candidate_spec(args)
        with self.assertRaisesRegex(ValueError, "fixed comparison environment"):
            resolve_candidate_environment(args, spec)

        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_gemma4_cuda_candidate.py",
                "--candidate",
                "q4-0-e2b-ffn-exact",
                "--common-env",
                f"{locked_name}=1",
            ],
        ):
            args = parse_args()
        spec = resolve_candidate_spec(args)
        with self.assertRaisesRegex(ValueError, "fixed comparison environment"):
            resolve_common_environment(args, spec, resolve_candidate_environment(args, spec))

    def test_candidate_environment_parser_rejects_malformed_or_unsafe_values(self):
        self.assertEqual(("ANTFLY_TEST", "with=equals"), parse_candidate_environment("ANTFLY_TEST=with=equals"))
        self.assertEqual(("ANTFLY_EMPTY", ""), parse_candidate_environment("ANTFLY_EMPTY="))
        for value, message in (
            ("ANTFLY_TEST", "NAME=VALUE"),
            ("antfly_test=value", "name is invalid"),
            ("=value", "name is invalid"),
            ("ANTFLY_TEST=line\nbreak", "control character"),
            ("ANTFLY_TEST=tab\tvalue", "control character"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                parse_candidate_environment(value)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            parse_candidate_environment_list(["ANTFLY_TEST=first", "ANTFLY_TEST=second"])

    def test_common_environment_parser_rejects_malformed_or_unsafe_values(self):
        self.assertEqual(("ANTFLY_TEST", "with=equals"), parse_common_environment("ANTFLY_TEST=with=equals"))
        self.assertEqual(("ANTFLY_EMPTY", ""), parse_common_environment("ANTFLY_EMPTY="))
        for value, message in (
            ("ANTFLY_TEST", "NAME=VALUE"),
            ("antfly_test=value", "name is invalid"),
            ("=value", "name is invalid"),
            ("ANTFLY_TEST=line\nbreak", "control character"),
            ("ANTFLY_TEST=tab\tvalue", "control character"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                parse_common_environment(value)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            parse_common_environment_list(["ANTFLY_TEST=first", "ANTFLY_TEST=second"])

    def test_candidate_environment_cannot_override_selected_gate(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_gemma4_cuda_candidate.py",
                "--candidate-env",
                "ANTFLY_GENERATED_ATTENTION_DECODE=0",
            ],
        ):
            args = parse_args()
        with self.assertRaisesRegex(ValueError, "must not override selected candidate gate"):
            resolve_candidate_environment(args, resolve_candidate_spec(args))

    def test_common_environment_cannot_override_gate_or_capture_capacity(self):
        for name in (
            "ANTFLY_GENERATED_ATTENTION_DECODE",
            CAPTURE_KV_CAPACITY_ENV,
            RUNTIME_CAPTURE_KV_CAPACITY_ENV,
        ):
            with self.subTest(name=name), mock.patch.object(
                sys,
                "argv",
                ["validate_gemma4_cuda_candidate.py", "--common-env", f"{name}=0"],
            ):
                args = parse_args()
                spec = resolve_candidate_spec(args)
                candidate_environment = resolve_candidate_environment(args, spec)
                expected = "selected candidate gate" if name == spec.environment_variable else "capture KV capacity"
                with self.assertRaisesRegex(ValueError, expected):
                    resolve_common_environment(args, spec, candidate_environment)

    def test_common_and_candidate_environment_cannot_overlap(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_gemma4_cuda_candidate.py",
                "--candidate-env",
                "ANTFLY_TEST_SCHEDULE=split8",
                "--common-env",
                "ANTFLY_TEST_SCHEDULE=serial",
            ],
        ):
            args = parse_args()
        spec = resolve_candidate_spec(args)
        candidate_environment = resolve_candidate_environment(args, spec)
        with self.assertRaisesRegex(ValueError, "must not override the same variable"):
            resolve_common_environment(args, spec, candidate_environment)

    def test_default_capture_capacity_has_prompt_headroom(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--lengths", "64", "768"],
        ):
            args = parse_args()
        self.assertEqual(
            (768 + DEFAULT_CAPTURE_KV_PROMPT_HEADROOM, "max-output-plus-prompt-headroom"),
            resolve_capture_kv_capacity(args),
        )
        args.capture_kv_capacity = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolve_capture_kv_capacity(args)

    def test_cli_selects_e2b_ffn_candidate(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--candidate", "q4-0-q8-1-e2b-ffn"],
        ):
            self.assertEqual(CandidateKind.Q4_0_Q8_1_E2B_FFN, parse_args().candidate)

    def test_cli_selects_exact_e2b_ffn_candidate(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--candidate", "q4-0-e2b-ffn-exact"],
        ):
            self.assertEqual(CandidateKind.Q4_0_E2B_FFN_EXACT, parse_args().candidate)

    def test_cli_selects_q6_lm_head_candidate(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--candidate", "q6-k-q8-1-lm-head-argmax"],
        ):
            args = parse_args()
        self.assertEqual(CandidateKind.Q6_K_Q8_1_LM_HEAD_ARGMAX, args.candidate)
        self.assertIsNone(args.model)
        self.assertIsNone(args.model_label)

    def test_cli_selects_model_neutral_catalog_kernel(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--kernel-id", Q4_0_Q8_1_FFN_KERNEL_ID],
        ):
            args = parse_args()
            self.assertIsNone(args.candidate)
            self.assertEqual(Q4_0_Q8_1_FFN_KERNEL_ID, args.kernel_id)
            self.assertEqual(
                CANDIDATE_CATALOG[Q4_0_Q8_1_FFN_KERNEL_ID],
                resolve_candidate_spec(args),
            )
            self.assertEqual(resolve_candidate_spec(args), candidate_spec(Q4_0_Q8_1_FFN_KERNEL_ID))

    def test_cli_selects_exact_e2b_ffn_catalog_kernel(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--kernel-id", Q4_0_E2B_FFN_EXACT_KERNEL_ID],
        ):
            args = parse_args()
        self.assertIsNone(args.candidate)
        self.assertEqual(Q4_0_E2B_FFN_EXACT_KERNEL_ID, args.kernel_id)
        self.assertEqual(
            CANDIDATE_CATALOG[Q4_0_E2B_FFN_EXACT_KERNEL_ID],
            resolve_candidate_spec(args),
        )

    def test_cli_selects_q6_model_neutral_catalog_kernel(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--kernel-id", Q6_K_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID],
        ):
            args = parse_args()
        self.assertIsNone(args.candidate)
        self.assertEqual(Q6_K_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID, args.kernel_id)
        self.assertEqual(
            CANDIDATE_CATALOG[Q6_K_Q8_1_LM_HEAD_ARGMAX_KERNEL_ID],
            resolve_candidate_spec(args),
        )

    def test_cli_records_e4b_12b_model_and_config_labels(self):
        model = pathlib.Path("/models/google/gemma-4-12B-it-QAT-E4B.gguf")
        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_gemma4_cuda_candidate.py",
                "--model",
                str(model),
                "--model-label",
                "gemma4-12b-e4b",
                "--config-label",
                "e4b-12b-sm89-f32",
            ],
        ):
            args = parse_args()
        config = result_config_metadata(args, timing_metadata_from_args(args))
        self.assertEqual(str(model), config["model_path"])
        self.assertEqual("gemma4-12b-e4b", config["model_label"])
        self.assertEqual("e4b-12b-sm89-f32", config["config_label"])
        self.assertEqual(
            {"path": str(model), "label": "gemma4-12b-e4b"},
            config["model"],
        )
        self.assertIsNone(config["candidate_environment"])
        self.assertIsNone(config["capture_kv_capacity"])

    def test_result_config_metadata_records_common_environment(self):
        with mock.patch.object(sys, "argv", ["validate_gemma4_cuda_candidate.py"]):
            args = parse_args()
        spec = resolve_candidate_spec(args)
        config = result_config_metadata(
            args,
            timing_metadata_from_args(args),
            spec,
            (),
            (4096, "explicit"),
            (("ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK", "0"),),
        )
        self.assertEqual(
            {"ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK": "0"},
            config["candidate_environment"]["common_overrides"],
        )
        self.assertEqual({"tokens": 4096, "source": "explicit"}, config["capture_kv_capacity"])

    def test_candidate_environment_metadata_records_split_threshold_provenance(self):
        attention = candidate_spec(CandidateKind.GENERATED_ATTENTION)
        metadata = candidate_environment_metadata(
            attention,
            (
                (GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV, "128"),
                (GENERATED_ATTENTION_SPLIT_KV_SPLITS_ENV, "4"),
                ("ANTFLY_TEST_SCHEDULE", "split8"),
            ),
        )
        self.assertEqual({attention.environment_variable: "0"}, metadata["baseline_gate"])
        self.assertEqual({attention.environment_variable: "1"}, metadata["candidate_gate"])
        self.assertEqual(
            "128",
            metadata["generated_attention_split_kv_min_tokens"]["value"],
        )
        self.assertEqual(
            "candidate-env",
            metadata["generated_attention_split_kv_min_tokens"]["source"],
        )
        self.assertEqual("4", metadata["generated_attention_split_kv_splits"]["value"])
        self.assertEqual("candidate-env", metadata["generated_attention_split_kv_splits"]["source"])
        self.assertEqual("split8", metadata["candidate_overrides"]["ANTFLY_TEST_SCHEDULE"])

        common = candidate_environment_metadata(
            attention,
            (),
            (
                (GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV, "512"),
                (GENERATED_ATTENTION_SPLIT_KV_SPLITS_ENV, "2"),
                ("ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING", "0"),
            ),
        )
        self.assertEqual(
            "512",
            common["generated_attention_split_kv_min_tokens"]["value"],
        )
        self.assertEqual(
            "common-env",
            common["generated_attention_split_kv_min_tokens"]["source"],
        )
        self.assertEqual("2", common["generated_attention_split_kv_splits"]["value"])
        self.assertEqual("common-env", common["generated_attention_split_kv_splits"]["source"])
        self.assertEqual(
            "0",
            common["common_overrides"]["ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING"],
        )

        with mock.patch.dict(
            os.environ,
            {
                GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV: "384",
                GENERATED_ATTENTION_SPLIT_KV_SPLITS_ENV: "8",
            },
            clear=False,
        ):
            inherited = candidate_environment_metadata(attention, ())
        self.assertEqual("384", inherited["generated_attention_split_kv_min_tokens"]["value"])
        self.assertEqual("inherited-environment", inherited["generated_attention_split_kv_min_tokens"]["source"])
        self.assertEqual("8", inherited["generated_attention_split_kv_splits"]["value"])
        self.assertEqual("inherited-environment", inherited["generated_attention_split_kv_splits"]["source"])

    def test_parses_cli_token_ids(self):
        match = TOKEN_IDS_RE.search("timing\ntoken_ids: 4 -2 17\ndone\n")
        self.assertIsNotNone(match)
        self.assertEqual([4, -2, 17], [int(value) for value in match.group("ids").split()])

    def test_accepts_equal_tokens_and_expected_routes(self):
        self.assertEqual([], validate_pair(timing(generated=0), timing(generated=7), 64))

    def test_accepts_score_prework_with_dedicated_route_counter(self):
        spec = CANDIDATE_CATALOG[SCORE_PREWORK_ATTENTION_KERNEL_ID]
        self.assertEqual([], validate_pair(timing(), timing(score_prework=64), 64, spec))

    def test_rejects_token_mismatch(self):
        candidate = timing(generated=7)
        candidate["token_ids"][10] = 999
        self.assertIn("token IDs differ at index 10", validate_pair(timing(generated=0), candidate, 64))

    def test_rejects_route_and_replay_failures(self):
        candidate = timing(generated=0)
        candidate["cuda"]["graph_capture_persistent_replays"] = 0
        errors = validate_pair(timing(generated=0), candidate, 64)
        self.assertTrue(any("persistent replays" in error for error in errors))
        self.assertIn("candidate did not use generated attention", errors)

    def test_accepts_lm_head_candidate_with_exact_tokens_and_route(self):
        spec = candidate_spec(CandidateKind.Q4_0_Q8_1_LM_HEAD_ARGMAX)
        self.assertEqual([], validate_pair(timing(), timing(lm_argmax=64), 64, spec))

    def test_rejects_lm_head_baseline_hit_and_missing_candidate_hit(self):
        spec = candidate_spec(CandidateKind.Q4_0_Q8_1_LM_HEAD_ARGMAX)
        errors = validate_pair(timing(lm_argmax=1), timing(), 64, spec)
        self.assertIn("baseline unexpectedly used Q4_0 x Q8_1 LM-head argmax", errors)
        self.assertIn("candidate did not use Q4_0 x Q8_1 LM-head argmax", errors)

    def test_rejects_lm_head_candidate_fallback(self):
        spec = candidate_spec(CandidateKind.Q4_0_Q8_1_LM_HEAD_ARGMAX)
        errors = validate_pair(timing(), timing(lm_argmax=4, lm_argmax_fallbacks=1), 64, spec)
        self.assertIn("candidate reported Q4_0 x Q8_1 LM-head argmax fallbacks: 1", errors)

    def test_accepts_generated_q6_lm_head_candidate_with_exact_tokens_and_route(self):
        spec = candidate_spec(CandidateKind.Q6_K_Q8_1_LM_HEAD_ARGMAX)
        self.assertEqual([], validate_pair(timing(), timing(generated_q6_lm_argmax=64), 64, spec))

    def test_rejects_generated_q6_lm_head_baseline_hit_missing_route_and_fallback(self):
        spec = candidate_spec(CandidateKind.Q6_K_Q8_1_LM_HEAD_ARGMAX)
        errors = validate_pair(
            timing(generated_q6_lm_argmax=1),
            timing(generated_q6_lm_argmax_fallbacks=1),
            64,
            spec,
        )
        self.assertIn("baseline unexpectedly used generated Q6_K x Q8_1 LM-head argmax", errors)
        self.assertIn("candidate did not use generated Q6_K x Q8_1 LM-head argmax", errors)
        self.assertIn("candidate reported generated Q6_K x Q8_1 LM-head argmax fallbacks: 1", errors)

    def test_accepts_e2b_ffn_candidate_when_both_routes_are_covered(self):
        spec = candidate_spec(CandidateKind.Q4_0_Q8_1_E2B_FFN)
        self.assertEqual([], validate_pair(timing(), timing(ffn_pair=32, ffn_down=16), 64, spec))

    def test_rejects_e2b_ffn_baseline_hits_independently(self):
        spec = candidate_spec(CandidateKind.Q4_0_Q8_1_E2B_FFN)
        errors = validate_pair(
            timing(ffn_pair=1, ffn_down=2),
            timing(ffn_pair=32, ffn_down=16),
            64,
            spec,
        )
        self.assertIn("baseline unexpectedly used Q4_0 x Q8_1 generated E2B FFN pair route", errors)
        self.assertIn("baseline unexpectedly used Q4_0 x Q8_1 generated E2B FFN down route", errors)

    def test_rejects_e2b_ffn_partial_candidate_coverage(self):
        spec = candidate_spec(CandidateKind.Q4_0_Q8_1_E2B_FFN)
        pair_missing = validate_pair(timing(), timing(ffn_down=16), 64, spec)
        down_missing = validate_pair(timing(), timing(ffn_pair=32), 64, spec)
        self.assertIn("candidate did not use Q4_0 x Q8_1 generated E2B FFN pair route", pair_missing)
        self.assertNotIn("candidate did not use Q4_0 x Q8_1 generated E2B FFN down route", pair_missing)
        self.assertIn("candidate did not use Q4_0 x Q8_1 generated E2B FFN down route", down_missing)
        self.assertNotIn("candidate did not use Q4_0 x Q8_1 generated E2B FFN pair route", down_missing)

    def test_rejects_each_e2b_ffn_candidate_fallback(self):
        spec = candidate_spec(CandidateKind.Q4_0_Q8_1_E2B_FFN)
        errors = validate_pair(
            timing(),
            timing(ffn_pair=32, ffn_down=16, ffn_pair_fallbacks=2, ffn_down_fallbacks=3),
            64,
            spec,
        )
        self.assertIn("candidate reported Q4_0 x Q8_1 generated E2B FFN pair fallbacks: 2", errors)
        self.assertIn("candidate reported Q4_0 x Q8_1 generated E2B FFN down fallbacks: 3", errors)

    def test_accepts_exact_e2b_ffn_candidate_when_both_routes_are_covered(self):
        spec = candidate_spec(CandidateKind.Q4_0_E2B_FFN_EXACT)
        self.assertEqual([], validate_pair(timing(), timing(exact_ffn_pair=32, exact_ffn_down=16), 64, spec))

    def test_rejects_exact_e2b_ffn_baseline_hits_independently(self):
        spec = candidate_spec(CandidateKind.Q4_0_E2B_FFN_EXACT)
        errors = validate_pair(
            timing(exact_ffn_pair=1, exact_ffn_down=2),
            timing(exact_ffn_pair=32, exact_ffn_down=16),
            64,
            spec,
        )
        self.assertIn("baseline unexpectedly used exact F32 generated E2B FFN pair route", errors)
        self.assertIn("baseline unexpectedly used exact F32 generated E2B FFN down route", errors)

    def test_rejects_exact_e2b_ffn_partial_candidate_coverage(self):
        spec = candidate_spec(CandidateKind.Q4_0_E2B_FFN_EXACT)
        pair_missing = validate_pair(timing(), timing(exact_ffn_down=16), 64, spec)
        down_missing = validate_pair(timing(), timing(exact_ffn_pair=32), 64, spec)
        self.assertIn("candidate did not use exact F32 generated E2B FFN pair route", pair_missing)
        self.assertNotIn("candidate did not use exact F32 generated E2B FFN down route", pair_missing)
        self.assertIn("candidate did not use exact F32 generated E2B FFN down route", down_missing)
        self.assertNotIn("candidate did not use exact F32 generated E2B FFN pair route", down_missing)

    def test_rejects_each_exact_e2b_ffn_candidate_fallback(self):
        spec = candidate_spec(CandidateKind.Q4_0_E2B_FFN_EXACT)
        errors = validate_pair(
            timing(),
            timing(
                exact_ffn_pair=32,
                exact_ffn_down=16,
                exact_ffn_pair_fallbacks=2,
                exact_ffn_down_fallbacks=3,
            ),
            64,
            spec,
        )
        self.assertIn("candidate reported exact F32 generated E2B FFN pair fallbacks: 2", errors)
        self.assertIn("candidate reported exact F32 generated E2B FFN down fallbacks: 3", errors)

    def test_candidate_spec_requires_unique_route_counters(self):
        with self.assertRaisesRegex(ValueError, "at least one route counter"):
            CandidateSpec("cuda.test", "TEST_GATE", ())
        duplicate = RouteCounter("duplicate", "duplicate route")
        with self.assertRaisesRegex(ValueError, "must be unique"):
            CandidateSpec("cuda.test", "TEST_GATE", (duplicate, duplicate))
        with self.assertRaisesRegex(ValueError, "forbidden counters must be unique"):
            CandidateSpec(
                "cuda.test",
                "TEST_GATE",
                (RouteCounter("required", "required"),),
                (duplicate, duplicate),
            )
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            CandidateSpec("cuda.test", "TEST_GATE", (duplicate,), (duplicate,))
        with self.assertRaisesRegex(ValueError, "kernel ID"):
            CandidateSpec("Gemma 4 E4B", "TEST_GATE", (duplicate,))
        with self.assertRaisesRegex(ValueError, "environment variable"):
            CandidateSpec("cuda.test", "bad-env", (duplicate,))

    def test_candidate_metadata_preserves_primary_counter_and_lists_all(self):
        attention = candidate_metadata(candidate_spec(CandidateKind.GENERATED_ATTENTION))
        self.assertEqual(GENERATED_ATTENTION_KERNEL_ID, attention["kernel_id"])
        self.assertEqual(attention["kernel_id"], attention["catalog_id"])
        self.assertEqual("launch_attention_gqa_decode_generated", attention["route_counter"])
        self.assertEqual([attention["route_counter"]], attention["route_counters"])
        self.assertEqual(attention["route_counters"], attention["required_route_counters"])
        self.assertEqual([], attention["candidate_forbidden_counters"])
        self.assertEqual([], attention["forbidden_route_counters"])

        e2b_ffn = candidate_metadata(candidate_spec(CandidateKind.Q4_0_Q8_1_E2B_FFN))
        self.assertEqual("q4_0_generated_e2b_pair_q8_hits", e2b_ffn["route_counter"])
        self.assertEqual(
            ["q4_0_generated_e2b_pair_q8_hits", "q4_0_generated_e2b_down_q8_hits"],
            e2b_ffn["route_counters"],
        )
        self.assertEqual(
            ["q4_0_generated_e2b_pair_q8_fallbacks", "q4_0_generated_e2b_down_q8_fallbacks"],
            e2b_ffn["candidate_forbidden_counters"],
        )

        exact_e2b_ffn = candidate_metadata(candidate_spec(CandidateKind.Q4_0_E2B_FFN_EXACT))
        self.assertEqual("q4_0_generated_e2b_exact_pair_f32_hits", exact_e2b_ffn["route_counter"])
        self.assertEqual(
            ["q4_0_generated_e2b_exact_pair_f32_hits", "q4_0_generated_e2b_exact_down_f32_hits"],
            exact_e2b_ffn["route_counters"],
        )
        self.assertEqual(
            [
                "q4_0_generated_e2b_exact_pair_f32_fallbacks",
                "q4_0_generated_e2b_exact_down_f32_fallbacks",
            ],
            exact_e2b_ffn["candidate_forbidden_counters"],
        )

        q6_lm_head = candidate_metadata(candidate_spec(CandidateKind.Q6_K_Q8_1_LM_HEAD_ARGMAX))
        self.assertEqual("lm_head_argmax_generated_q6_k_q8_1_hits", q6_lm_head["route_counter"])
        self.assertTrue(q6_lm_head["requires_explicit_model"])
        self.assertEqual(
            ["lm_head_argmax_generated_q6_k_q8_1_fallbacks"],
            q6_lm_head["candidate_forbidden_counters"],
        )

    def test_candidate_environment_only_changes_selected_route(self):
        attention = CANDIDATE_SPECS[CandidateKind.GENERATED_ATTENTION]
        lm_argmax = CANDIDATE_SPECS[CandidateKind.Q4_0_Q8_1_LM_HEAD_ARGMAX]
        q6_lm_argmax = CANDIDATE_SPECS[CandidateKind.Q6_K_Q8_1_LM_HEAD_ARGMAX]
        e2b_ffn = CANDIDATE_SPECS[CandidateKind.Q4_0_Q8_1_E2B_FFN]
        exact_e2b_ffn = CANDIDATE_SPECS[CandidateKind.Q4_0_E2B_FFN_EXACT]
        env = {
            attention.environment_variable: "inherited-attention",
            lm_argmax.environment_variable: "inherited-lm",
            q6_lm_argmax.environment_variable: "inherited-q6-lm",
            e2b_ffn.environment_variable: "inherited-ffn",
            exact_e2b_ffn.environment_variable: "inherited-exact-ffn",
        }
        overrides = (("ANTFLY_TEST_OVERRIDE", "candidate-only"),)
        configure_candidate_environment(env, e2b_ffn, False, overrides)
        self.assertEqual("inherited-attention", env[attention.environment_variable])
        self.assertEqual("inherited-lm", env[lm_argmax.environment_variable])
        self.assertEqual("inherited-q6-lm", env[q6_lm_argmax.environment_variable])
        self.assertEqual("inherited-exact-ffn", env[exact_e2b_ffn.environment_variable])
        self.assertEqual("0", env[e2b_ffn.environment_variable])
        self.assertNotIn("ANTFLY_TEST_OVERRIDE", env)
        configure_candidate_environment(env, e2b_ffn, True, overrides)
        self.assertEqual("1", env[e2b_ffn.environment_variable])
        self.assertEqual("candidate-only", env["ANTFLY_TEST_OVERRIDE"])

    def test_exact_e2b_ffn_environment_only_changes_its_gate(self):
        approximate = CANDIDATE_SPECS[CandidateKind.Q4_0_Q8_1_E2B_FFN]
        exact = CANDIDATE_SPECS[CandidateKind.Q4_0_E2B_FFN_EXACT]
        env = {
            approximate.environment_variable: "inherited-approximate",
            exact.environment_variable: "inherited-exact",
        }
        configure_candidate_environment(env, exact, False)
        self.assertEqual("inherited-approximate", env[approximate.environment_variable])
        self.assertEqual("0", env[exact.environment_variable])
        configure_candidate_environment(env, exact, True)
        self.assertEqual("inherited-approximate", env[approximate.environment_variable])
        self.assertEqual("1", env[exact.environment_variable])

    def test_common_environment_applies_to_baseline_and_candidate(self):
        attention = CANDIDATE_SPECS[CandidateKind.GENERATED_ATTENTION]
        common = (("ANTFLY_TEST_COMMON", "shared"),)
        candidate_only = (("ANTFLY_TEST_CANDIDATE", "candidate"),)
        baseline = {}
        candidate = {}
        configure_candidate_environment(baseline, attention, False, candidate_only, common)
        configure_candidate_environment(candidate, attention, True, candidate_only, common)
        self.assertEqual("shared", baseline["ANTFLY_TEST_COMMON"])
        self.assertEqual("shared", candidate["ANTFLY_TEST_COMMON"])
        self.assertNotIn("ANTFLY_TEST_CANDIDATE", baseline)
        self.assertEqual("candidate", candidate["ANTFLY_TEST_CANDIDATE"])
        self.assertEqual("0", baseline[attention.environment_variable])
        self.assertEqual("1", candidate[attention.environment_variable])

    def test_run_case_applies_common_and_candidate_environment_with_explicit_capture_capacity(self):
        attention = candidate_spec(CandidateKind.GENERATED_ATTENTION)
        overrides = ((GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV, "128"),)
        common = (("ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING", "0"),)
        seen_environments = []

        with tempfile.TemporaryDirectory() as temp_dir:
            args = type(
                "Args",
                (),
                {
                    "output_dir": pathlib.Path(temp_dir),
                    "wrapper": pathlib.Path("/test-wrapper"),
                    "binary": pathlib.Path("/test-binary"),
                    "model": pathlib.Path("/test-model.gguf"),
                    "timeout_sec": 1,
                },
            )()

            def fake_run(command, **kwargs):
                seen_environments.append(kwargs["env"])
                timing_path = pathlib.Path(command[command.index("--json-timing") + 1])
                timing_path.write_text(json.dumps(timing(generated=1)), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "token_ids: 0 1 2\n")

            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "validate_gemma4_cuda_candidate.subprocess.run",
                side_effect=fake_run,
            ):
                run_case(args, "prompt", 64, False, "baseline", attention, overrides, 4096, common)
                run_case(args, "prompt", 64, True, "candidate", attention, overrides, 4096, common)

        self.assertEqual("0", seen_environments[0][attention.environment_variable])
        self.assertNotIn(GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV, seen_environments[0])
        self.assertEqual("1", seen_environments[1][attention.environment_variable])
        self.assertEqual("128", seen_environments[1][GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS_ENV])
        self.assertEqual("0", seen_environments[0]["ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING"])
        self.assertEqual("0", seen_environments[1]["ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING"])
        self.assertEqual("4096", seen_environments[0]["ANTFLY_CAPTURE_FORCE_KV_CAPACITY"])
        self.assertEqual("4096", seen_environments[1]["ANTFLY_CAPTURE_FORCE_KV_CAPACITY"])

    def test_relative_launch_paths_are_absolute_before_repo_root_subprocess(self):
        attention = candidate_spec(CandidateKind.GENERATED_ATTENTION)
        seen_commands = []
        with tempfile.TemporaryDirectory() as temp_dir:
            invocation_dir = pathlib.Path(temp_dir) / "invocation"
            invocation_dir.mkdir()
            output_dir = invocation_dir / "output"
            old_cwd = pathlib.Path.cwd()
            try:
                os.chdir(invocation_dir)
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "validate_gemma4_cuda_candidate.py",
                        "--binary",
                        "zig-out/bin/antfly-inference",
                        "--wrapper",
                        "scripts/with_gemma4_qat_cuda_tuning.sh",
                        "--model",
                        "models/e2b.gguf",
                        "--output-dir",
                        str(output_dir),
                    ],
                ):
                    args = parse_args()
            finally:
                os.chdir(old_cwd)

            self.assertEqual((invocation_dir / "zig-out/bin/antfly-inference").resolve(), args.binary)
            self.assertEqual(
                (invocation_dir / "scripts/with_gemma4_qat_cuda_tuning.sh").resolve(),
                args.wrapper,
            )
            self.assertEqual((invocation_dir / "models/e2b.gguf").resolve(), args.model)
            output_dir.mkdir()

            def fake_run(command, **_kwargs):
                seen_commands.append(command)
                timing_path = pathlib.Path(command[command.index("--json-timing") + 1])
                timing_path.write_text(json.dumps(timing(generated=1)), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "token_ids: 0 1 2\n")

            with mock.patch(
                "validate_gemma4_cuda_candidate.subprocess.run",
                side_effect=fake_run,
            ):
                run_case(args, "prompt", 64, False, "relative", attention, (), 4096)

        command = seen_commands[0]
        self.assertEqual(str(args.wrapper), command[0])
        self.assertEqual(str(args.binary), command[1])
        self.assertEqual(str(args.model), command[3])
        self.assertEqual("f32", command[command.index("--cache-dtype") + 1])
        self.assertTrue(pathlib.Path(command[0]).is_absolute())
        self.assertTrue(pathlib.Path(command[1]).is_absolute())
        self.assertTrue(pathlib.Path(command[3]).is_absolute())

    def test_exact_e2b_ffn_run_case_applies_f32_locks_to_both_arms(self):
        spec = candidate_spec(CandidateKind.Q4_0_E2B_FFN_EXACT)
        seen_environments = []
        with tempfile.TemporaryDirectory() as temp_dir:
            args = type(
                "Args",
                (),
                {
                    "output_dir": pathlib.Path(temp_dir),
                    "wrapper": pathlib.Path("/test-wrapper"),
                    "binary": pathlib.Path("/test-binary"),
                    "model": pathlib.Path("/test-model.gguf"),
                    "timeout_sec": 1,
                },
            )()

            def fake_run(command, **kwargs):
                seen_environments.append(kwargs["env"])
                timing_path = pathlib.Path(command[command.index("--json-timing") + 1])
                timing_path.write_text(json.dumps(timing(exact_ffn_pair=1, exact_ffn_down=1)), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "token_ids: 0 1 2\n")

            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "validate_gemma4_cuda_candidate.subprocess.run",
                side_effect=fake_run,
            ):
                run_case(
                    args,
                    "prompt",
                    64,
                    False,
                    "baseline",
                    spec,
                    (),
                    4096,
                    EXACT_E2B_FFN_F32_COMPARISON_ENVIRONMENT,
                )
                run_case(
                    args,
                    "prompt",
                    64,
                    True,
                    "candidate",
                    spec,
                    (),
                    4096,
                    EXACT_E2B_FFN_F32_COMPARISON_ENVIRONMENT,
                )

        for name, value in EXACT_E2B_FFN_F32_COMPARISON_ENVIRONMENT:
            self.assertEqual(value, seen_environments[0][name])
            self.assertEqual(value, seen_environments[1][name])
        self.assertEqual("0", seen_environments[0][spec.environment_variable])
        self.assertEqual("1", seen_environments[1][spec.environment_variable])

    def test_uncatalogued_e4b_kernel_uses_explicit_route_contract(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_gemma4_cuda_candidate.py",
                "--kernel-id",
                "cuda.quant.q4_0-q8_1.ffn.e4b-12b",
                "--candidate-environment-variable",
                "ANTFLY_TEST_E4B_FFN",
                "--required-route-counters",
                "q4_0_generated_pair_q8_hits=E4B pair,q4_0_generated_down_q8_hits=E4B down",
                "--forbidden-route-counter",
                "q4_0_generated_pair_q8_fallbacks=E4B pair fallback",
            ],
        ):
            spec = resolve_candidate_spec(parse_args())
        self.assertIsNone(spec.legacy_kind)
        self.assertEqual("cuda.quant.q4_0-q8_1.ffn.e4b-12b", spec.kernel_id)
        self.assertEqual("ANTFLY_TEST_E4B_FFN", spec.environment_variable)
        self.assertEqual(
            ["q4_0_generated_pair_q8_hits", "q4_0_generated_down_q8_hits"],
            [counter.name for counter in spec.required_route_counters],
        )
        self.assertEqual("E4B pair", spec.required_route_counters[0].label)
        self.assertEqual(
            ["q4_0_generated_pair_q8_fallbacks"],
            [counter.name for counter in spec.forbidden_route_counters],
        )

    def test_uncatalogued_kernel_requires_gate_and_required_counter(self):
        with mock.patch.object(
            sys,
            "argv",
            ["validate_gemma4_cuda_candidate.py", "--kernel-id", "cuda.custom"],
        ):
            with self.assertRaisesRegex(ValueError, "environment-variable"):
                resolve_candidate_spec(parse_args())

        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_gemma4_cuda_candidate.py",
                "--kernel-id",
                "cuda.custom",
                "--candidate-environment-variable",
                "ANTFLY_CUSTOM",
            ],
        ):
            with self.assertRaisesRegex(ValueError, "required-route-counter"):
                resolve_candidate_spec(parse_args())

    def test_route_counter_list_supports_repeated_and_comma_separated_values(self):
        counters = parse_route_counter_list(["first=First route,second", "third=Third route"])
        self.assertEqual(["first", "second", "third"], [counter.name for counter in counters])
        self.assertEqual(["First route", "second", "Third route"], [counter.label for counter in counters])

    def test_generic_timing_metadata_controls_counter_group_and_throughput_field(self):
        metadata = TimingMetadata(
            counter_group="device.routes",
            throughput_field="performance.generated_tokens_per_second",
            throughput_unit="generated_tokens_per_second",
            require_persistent_replay=False,
        )
        spec = CandidateSpec(
            kernel_id="cuda.quant.e4b-12b.test",
            environment_variable="ANTFLY_TEST_E4B",
            required_route_counters=(RouteCounter("candidate_hits", "candidate route"),),
            forbidden_route_counters=(RouteCounter("candidate_fallbacks", "candidate fallback"),),
        )

        baseline = {
            "token_ids": list(range(64)),
            "performance": {"generated_tokens_per_second": 80.0},
            "device": {"routes": {"candidate_hits": 0, "candidate_fallbacks": 0}},
        }
        candidate = {
            "token_ids": list(range(64)),
            "performance": {"generated_tokens_per_second": 100.0},
            "device": {"routes": {"candidate_hits": 64, "candidate_fallbacks": 0}},
        }
        self.assertEqual([], validate_pair(baseline, candidate, 64, spec, metadata))
        self.assertEqual(64, timing_counter(candidate, "candidate_hits", metadata))
        self.assertEqual(100.0, timing_throughput(candidate, metadata))
        self.assertEqual(1.25, paired_throughput(baseline, candidate, metadata)["candidate_ratio"])
        self.assertEqual(
            {
                "counter_group": "device.routes",
                "throughput_field": "performance.generated_tokens_per_second",
                "throughput_unit": "generated_tokens_per_second",
                "require_persistent_replay": False,
                "persistent_replay_counter": "graph_capture_persistent_replays",
                "graph_discard_counter": "graph_capture_discards",
                "graph_capacity_skip_counter": "graph_capture_capacity_skips",
            },
            timing_metadata(metadata),
        )

    def test_execution_order_alternates_by_repetition(self):
        self.assertEqual((False, True), execution_order(0))
        self.assertEqual((True, False), execution_order(1))
        self.assertEqual((False, True), execution_order(2))

    def test_single_repeat_preserves_filenames_and_repeats_are_distinct(self):
        self.assertEqual("case", repetition_stem("case", 0, 1))
        self.assertEqual("case-r01", repetition_stem("case", 0, 3))
        self.assertEqual("case-r02", repetition_stem("case", 1, 3))

    def test_reports_per_pair_and_median_throughput(self):
        first = paired_throughput(timing(tok_s=100.0), timing(tok_s=125.0))
        second = paired_throughput(timing(tok_s=80.0), timing(tok_s=88.0))
        self.assertEqual(1.25, first["candidate_ratio"])
        self.assertEqual(25.0, first["delta_percent"])
        summary = summarize_throughput([first, second])
        self.assertEqual(90.0, summary["baseline_median_tok_s"])
        self.assertEqual(106.5, summary["candidate_median_tok_s"])
        self.assertAlmostEqual(1.175, summary["median_candidate_ratio"])
        self.assertEqual(1.1, summary["min_candidate_ratio"])
        self.assertEqual(2, summary["pair_count"])

    def test_repeated_case_reports_medians_min_ratio_and_cvs(self):
        pairs = []
        for repetition, (baseline_tps, candidate_tps) in enumerate(
            ((100.0, 110.0), (110.0, 121.0), (90.0, 99.0)),
            start=1,
        ):
            throughput = paired_throughput(timing(tok_s=baseline_tps), timing(tok_s=candidate_tps))
            pairs.append(
                {
                    "repetition": repetition,
                    "execution_order": ["baseline", "candidate"],
                    **throughput,
                    "paired_throughput": throughput,
                    "token_ids_equal": True,
                    "errors": [],
                }
            )

        case = summarize_case("prompt", 64, pairs)
        self.assertEqual(100.0, case["baseline_tok_s"])
        self.assertEqual(110.0, case["candidate_tok_s"])
        self.assertAlmostEqual(1.1, case["candidate_ratio"])
        self.assertAlmostEqual(1.1, case["min_candidate_ratio"])
        self.assertAlmostEqual(coefficient_of_variation([100.0, 110.0, 90.0]), case["baseline_tok_s_cv"])
        self.assertAlmostEqual(coefficient_of_variation([110.0, 121.0, 99.0]), case["candidate_tok_s_cv"])
        self.assertEqual(3, case["repeats"])
        self.assertTrue(case["token_ids_equal"])

        aggregate = summarize_throughput([case])
        self.assertEqual(3, aggregate["pair_count"])
        self.assertAlmostEqual(1.1, aggregate["median_candidate_ratio"])

    def test_repeated_case_retains_each_pair_validation_error(self):
        throughput = paired_throughput(timing(tok_s=100.0), timing(tok_s=100.0))
        pairs = [
            {
                "repetition": 1,
                **throughput,
                "paired_throughput": throughput,
                "token_ids_equal": True,
                "errors": ["first failure"],
            },
            {
                "repetition": 2,
                **throughput,
                "paired_throughput": throughput,
                "token_ids_equal": False,
                "errors": ["second failure"],
            },
        ]
        case = summarize_case("prompt", 64, pairs)
        self.assertEqual(["repeat 1: first failure", "repeat 2: second failure"], case["errors"])
        self.assertFalse(case["token_ids_equal"])

    def test_promotion_gates_use_min_paired_ratio_and_both_cvs(self):
        case = {
            "min_candidate_ratio": 1.01,
            "baseline_tok_s_cv": 0.01,
            "candidate_tok_s_cv": 0.02,
        }
        self.assertEqual([], case_promotion_errors(case, 1.0, 0.02))
        self.assertTrue(any("minimum candidate ratio" in error for error in case_promotion_errors(case, 1.02, 0.02)))
        self.assertTrue(any("baseline throughput CV" in error for error in case_promotion_errors(case | {"baseline_tok_s_cv": 0.03}, 1.0, 0.02)))
        self.assertTrue(any("candidate throughput CV" in error for error in case_promotion_errors(case | {"candidate_tok_s_cv": 0.03}, 1.0, 0.02)))

    def test_non_positive_throughput_fails_pair_validation(self):
        candidate = timing(generated=1, tok_s=0.0)
        self.assertIn(
            "candidate reported non-positive decode throughput",
            validate_pair(timing(), candidate, 64),
        )


if __name__ == "__main__":
    unittest.main()
