#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("validate_gemma4_grpo_incremental_kv_parity.py")
SPEC = importlib.util.spec_from_file_location("incremental_kv_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class IncrementalKvParityGateTests(unittest.TestCase):
    def test_accepts_checkpoint_report_schema_v6(self) -> None:
        self.assertIn("antfly_inference_finetune_grpo_report/v6", gate.TRAIN_SCHEMAS)

    def test_accepts_exact_qualified_telemetry(self) -> None:
        report = {
            "incremental_kv": {
                "groups": 8,
                "prompt_prefill_forwards": 8,
                "prompt_tail_prefill_forwards": 24,
                "prompt_tail_prefill_candidates": 24,
                "max_prompt_tail_batch_size": 1,
                "decode_forwards": 88,
                "exact_logprob_rescore_forwards": 32,
                "resident_ranked_token_selections": 248,
                "host_logit_fallbacks": 0,
                "shared_prompt_tokens": 1024,
                "reused_candidate_prompt_tokens": 1024,
                "cache_page_tokens": 16,
                "cache_dtype": "f32",
            }
        }
        telemetry = gate.require_incremental_telemetry(
            report,
            expected_groups=8,
            expected_completions=32,
            expected_tokens=120,
            label="training",
        )
        self.assertEqual(1024, telemetry["reused_candidate_prompt_tokens"])

    def test_rejects_non_reusing_candidate(self) -> None:
        report = {
            "incremental_kv": {
                "groups": 8,
                "prompt_prefill_forwards": 8,
                "prompt_tail_prefill_forwards": 24,
                "prompt_tail_prefill_candidates": 24,
                "max_prompt_tail_batch_size": 1,
                "decode_forwards": 88,
                "exact_logprob_rescore_forwards": 32,
                "resident_ranked_token_selections": 120,
                "host_logit_fallbacks": 0,
                "shared_prompt_tokens": 0,
                "reused_candidate_prompt_tokens": 0,
                "cache_page_tokens": 16,
                "cache_dtype": "f32",
            }
        }
        with self.assertRaises(gate.ParityError):
            gate.require_incremental_telemetry(
                report,
                expected_groups=8,
                expected_completions=32,
                expected_tokens=120,
                label="training",
            )

    def test_semantic_field_selection_fails_closed(self) -> None:
        with self.assertRaises(gate.ParityError):
            gate.select_fields({"groups": 8}, ("groups", "loss"))

    def test_requires_real_active_candidate_amortization_when_requested(self) -> None:
        report = {
            "incremental_kv": {
                "groups": 8,
                "prompt_prefill_forwards": 8,
                "prompt_tail_prefill_forwards": 24,
                "prompt_tail_prefill_candidates": 24,
                "max_prompt_tail_batch_size": 1,
                "decode_forwards": 24,
                "decode_forward_candidates": 88,
                "max_decode_batch_size": 4,
                "active_candidate_batching": True,
                "exact_logprob_rescore_forwards": 32,
                "resident_ranked_token_selections": 248,
                "host_logit_fallbacks": 0,
                "shared_prompt_tokens": 1024,
                "reused_candidate_prompt_tokens": 1024,
                "cache_page_tokens": 16,
                "cache_dtype": "f32",
            }
        }
        telemetry = gate.require_incremental_telemetry(
            report,
            expected_groups=8,
            expected_completions=32,
            expected_tokens=120,
            label="training",
            require_active_batching=True,
        )
        self.assertEqual(4, telemetry["max_decode_batch_size"])

        report["incremental_kv"]["decode_forward_candidates"] = 24
        with self.assertRaises(gate.ParityError):
            gate.require_incremental_telemetry(
                report,
                expected_groups=8,
                expected_completions=32,
                expected_tokens=120,
                label="training",
                require_active_batching=True,
            )

    def test_requires_one_segmented_canonical_prompt_tail_and_clone_fanout_when_requested(self) -> None:
        report = {
            "incremental_kv": {
                "groups": 8,
                "prompt_prefill_forwards": 8,
                "prompt_tail_prefill_forwards": 8,
                "prompt_tail_prefill_candidates": 8,
                "prompt_tail_clone_candidates": 24,
                "prompt_tail_clone_tokens": 168,
                "prompt_tail_cloning": True,
                "decode_forwards": 24,
                "exact_logprob_rescore_forwards": 32,
                "resident_ranked_token_selections": 248,
                "host_logit_fallbacks": 0,
                "shared_prompt_tokens": 1024,
                "reused_candidate_prompt_tokens": 1024,
                "cache_page_tokens": 16,
                "cache_dtype": "f32",
            }
        }
        telemetry = gate.require_incremental_telemetry(
            report,
            expected_groups=8,
            expected_completions=32,
            expected_tokens=120,
            label="training",
            require_prompt_tail_cloning=True,
        )
        self.assertEqual(24, telemetry["prompt_tail_clone_candidates"])

        report["incremental_kv"]["prompt_tail_prefill_forwards"] = 0
        with self.assertRaises(gate.ParityError):
            gate.require_incremental_telemetry(
                report,
                expected_groups=8,
                expected_completions=32,
                expected_tokens=120,
                label="training",
                require_prompt_tail_cloning=True,
            )

    def test_accepts_page_aligned_groups_without_tail_clones(self) -> None:
        report = {
            "incremental_kv": {
                "groups": 8,
                "prompt_prefill_forwards": 8,
                "prompt_tail_prefill_forwards": 7,
                "prompt_tail_prefill_candidates": 7,
                "prompt_tail_clone_candidates": 21,
                "prompt_tail_clone_tokens": 147,
                "prompt_tail_cloning": True,
                "decode_forwards": 24,
                "exact_logprob_rescore_forwards": 32,
                "resident_ranked_token_selections": 248,
                "host_logit_fallbacks": 0,
                "shared_prompt_tokens": 1024,
                "reused_candidate_prompt_tokens": 1024,
                "cache_page_tokens": 16,
                "cache_dtype": "f32",
            }
        }
        telemetry = gate.require_incremental_telemetry(
            report,
            expected_groups=8,
            expected_completions=32,
            expected_tokens=120,
            label="evaluation",
            require_prompt_tail_cloning=True,
        )
        self.assertEqual(21, telemetry["prompt_tail_clone_candidates"])

        report["incremental_kv"]["prompt_tail_clone_candidates"] = 24
        with self.assertRaises(gate.ParityError):
            gate.require_incremental_telemetry(
                report,
                expected_groups=8,
                expected_completions=32,
                expected_tokens=120,
                label="evaluation",
                require_prompt_tail_cloning=True,
            )


if __name__ == "__main__":
    unittest.main()
