#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import run_gemma4_grpo_boolq_mlx_multitoken as campaign


class Gemma4GrpoBoolQMultiTokenTests(unittest.TestCase):
    def test_accepts_stochastic_report_schema_v7(self) -> None:
        self.assertIn(
            "antfly_inference_finetune_grpo_report/v7",
            campaign.GRPO_REPORT_SCHEMA_VERSIONS,
        )
        self.assertIn(
            "antfly_inference_finetune_grpo_evaluation/v4",
            campaign.GRPO_EVAL_SCHEMA_VERSIONS,
        )

    def test_shared_native_rollout_guard_rejects_stochastic_sampler_drift(self) -> None:
        with self.assertRaisesRegex(
            campaign.legacy.BoolQParityContractError,
            "retired deterministic ranked sampler",
        ):
            campaign.legacy.require_native_rollout_sampler_compatibility(
                {"sampling_mode": "shared-prompt-seeded-categorical"}
            )

    def test_campaign_shape_requires_a_real_multi_token_matrix(self) -> None:
        campaign.CampaignSpec("gemma-4-E2B-it", 8, 16, 4, 4).validate()
        with self.assertRaisesRegex(campaign.MultiTokenParityError, "multi-token"):
            campaign.CampaignSpec("gemma-4-E2B-it", 8, 16, 4, 1).validate()
        with self.assertRaisesRegex(campaign.MultiTokenParityError, "at least two"):
            campaign.CampaignSpec("gemma-4-E4B-it", 1, 16, 4, 4).validate()

    def test_adaptive_kl_rule_is_bounded_and_applies_to_next_group(self) -> None:
        below = campaign.adaptive_kl_update(0.04, 0.0)
        self.assertAlmostEqual(0.03992, below, places=8)
        above = campaign.adaptive_kl_update(below, 1.0)
        self.assertGreater(above, below)
        current = campaign.GRPO["max_kl_coef"]
        self.assertEqual(current, campaign.adaptive_kl_update(current, 1.0))

    def test_stable_raw_k3_is_independent_of_beta(self) -> None:
        expected = math.expm1(0.5) - 0.5
        self.assertAlmostEqual(
            expected,
            campaign.mean_k3([-1.0], [-0.5]),
            places=12,
        )
        self.assertEqual(0.0, campaign.mean_k3([-1.0], [-1.0]))

    def test_prefix_reward_matches_case_sensitive_antfly_contract(self) -> None:
        self.assertEqual(1.0, campaign.prefix_match_reward(" yes indeed\n", "yes"))
        self.assertEqual(0.0, campaign.prefix_match_reward("Yes indeed", "yes"))
        self.assertEqual(0.0, campaign.prefix_match_reward("indeed yes", "yes"))

    def test_sequence_overlap_keeps_full_sequence_and_first_token_evidence(self) -> None:
        overlap = campaign.sequence_overlap(
            [[1, 2], [3, 4]],
            [[1, 9], [3, 4]],
        )
        self.assertEqual(0.5, overlap["sequence_recall"])
        self.assertEqual(1.0, overlap["first_token_recall"])
        self.assertFalse(overlap["top_sequence_match"])
        self.assertTrue(overlap["top1_first_token_match"])

    def test_trace_loader_accepts_variable_length_eos_completions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            sequences = ([10, 11, 1], [20, 1])
            rows = []
            for index, sequence in enumerate(sequences):
                rows.append(
                    {
                        "schema_version": campaign.REWARD_TRACE_SCHEMA_VERSION,
                        "phase": "train",
                        "call_index": index,
                        "prompt_index": 0,
                        "completion_tokens": list(sequence),
                        "aggregate_reward": float(index == 0),
                    }
                )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            groups = campaign.load_trace(
                path,
                phase="train",
                expected_groups=1,
                group_size=2,
                max_completion_tokens=4,
            )
            self.assertEqual(tuple(tuple(row) for row in sequences), groups[0].sequences)
            self.assertEqual((10, 20), groups[0].first_token_ids)

    def test_import_surface_keeps_mlx_lazy(self) -> None:
        source = Path(campaign.__file__).read_text(encoding="utf-8")
        prefix = source.split("def run(args", 1)[0]
        self.assertNotIn("import mlx.core", prefix)
        self.assertNotIn("import mlx.nn", prefix)

    def test_mlx_candidate_contract_stays_batch_one_and_fail_closed(self) -> None:
        source = Path(campaign.__file__).read_text(encoding="utf-8")
        scoring = source.split("        def score_sequences(", 1)[1].split(
            "        def ranked_group(", 1
        )[0]
        rollout = source.split("        def ranked_group(", 1)[1].split(
            "        def flatten(", 1
        )[0]
        optimizer_step = source.split("            def step(", 1)[1].split(
            "            compiled_step =", 1
        )[0]
        training_loop = source.split("        def train_lane(", 1)[1].split(
            "        def evaluate_lane(", 1
        )[0]

        self.assertIn("for sequence in sequences:", scoring)
        self.assertIn("padded_sequence(row, sequence)", scoring)
        self.assertIn("for completion_index in active_indices:", rollout)
        self.assertNotIn("model(tokens).astype(mx.float32)[:, row_index", rollout)
        self.assertIn(
            "tokens[completion_index : completion_index + 1]", optimizer_step
        )
        self.assertIn("tree_map(", optimizer_step)
        self.assertIn("sampling_rescore_max_abs_error > 1.0e-4", training_loop)
        self.assertIn("raw_mean_kl > GRPO[\"train_max_kl\"]", training_loop)


if __name__ == "__main__":
    unittest.main()
