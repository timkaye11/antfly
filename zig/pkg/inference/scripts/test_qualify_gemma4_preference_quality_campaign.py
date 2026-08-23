from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import qualify_gemma4_preference_quality_campaign as campaign


class PreferenceQualityCampaignTest(unittest.TestCase):
    def test_seed_contract_requires_three_unique_u64_values(self) -> None:
        self.assertEqual(campaign._parse_seeds("17,42,991"), [17, 42, 991])
        with self.assertRaisesRegex(Exception, "three unique"):
            campaign._parse_seeds("17,17,42")
        with self.assertRaisesRegex(Exception, "fit u64"):
            campaign._parse_seeds("17,42,-1")

    def test_seeded_row_order_is_stable_distinct_and_content_preserving(self) -> None:
        rows = [
            json.dumps({"row": index}, sort_keys=True).encode()
            for index in range(8)
        ]
        first = campaign._permuted_rows(rows, 17)
        self.assertEqual(first, campaign._permuted_rows(rows, 17))
        self.assertNotEqual(first, campaign._permuted_rows(rows, 42))
        self.assertCountEqual(first.splitlines(), rows)
        self.assertEqual(
            campaign._row_multiset_sha256(first.splitlines()),
            campaign._row_multiset_sha256(rows),
        )

    def test_selected_row_set_must_support_every_requested_order_seed(self) -> None:
        campaign._require_permutation_capacity(3, 3)
        with self.assertRaisesRegex(campaign.ContractError, "enough distinct"):
            campaign._require_permutation_capacity(2, 3)

    def test_quality_metric_ranges_fail_closed(self) -> None:
        self.assertEqual(campaign._probability(0.25, "rate"), 0.25)
        self.assertEqual(campaign._nonnegative(0.0, "loss"), 0.0)
        with self.assertRaisesRegex(campaign.ContractError, r"\[0, 1\]"):
            campaign._probability(1.01, "rate")
        with self.assertRaisesRegex(campaign.ContractError, "non-negative"):
            campaign._nonnegative(-0.01, "loss")

    def test_variant_binds_horizon_order_and_direct_gguf_admission(self) -> None:
        base = {
            "recipe": "grpo",
            "model": {"path": "/model.gguf", "family": "gemma4"},
            "dataset": {"path": "/train.jsonl", "eval_path": "/eval.jsonl"},
            "optimizer": {"epochs": 2, "learning_rate": 1e-7},
            "checkpoint": {"every_epochs": 1},
            "artifacts": {"root": "/old"},
        }
        variant = campaign._variant(
            base,
            "grpo",
            991,
            Path("/seeded.jsonl"),
            Path("/run"),
            8,
            5e-8,
            True,
        )
        self.assertEqual(variant["dataset"]["path"], "/seeded.jsonl")
        self.assertEqual(variant["optimizer"]["epochs"], 8)
        self.assertEqual(variant["optimizer"]["seed"], 991)
        self.assertEqual(variant["optimizer"]["learning_rate"], 5e-8)
        self.assertTrue(variant["model"]["allow_direct_gguf_training"])
        self.assertNotIn("checkpoint", variant)
        self.assertEqual(variant["artifacts"]["root"], "/run")
        self.assertIn("checkpoint", base)

    def test_variant_rebinds_both_training_dataset_aliases(self) -> None:
        base = {
            "recipe": "dpo",
            "model": {"path": "/model", "family": "gemma4"},
            "dataset": {
                "path": "/train.jsonl",
                "train_path": "/train.jsonl",
                "eval_path": "/eval.jsonl",
            },
            "optimizer": {"epochs": 2},
        }
        variant = campaign._variant(
            base,
            "dpo",
            42,
            Path("/seeded.jsonl"),
            Path("/run"),
            8,
            None,
            False,
        )
        self.assertEqual(variant["dataset"]["path"], "/seeded.jsonl")
        self.assertEqual(variant["dataset"]["train_path"], "/seeded.jsonl")

    def test_dataset_path_uses_runtime_precedence_and_rejects_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            alternate = root / "alternate.jsonl"
            for path in (train, evaluation, alternate):
                path.write_text("{}\n", encoding="utf-8")
            recipe = {
                "dataset": {
                    "path": str(train),
                    "train_path": str(train),
                    "eval_path": str(evaluation),
                },
                "eval": {"path": str(evaluation)},
            }
            self.assertEqual(campaign._dataset_path(recipe, "train"), train.resolve())
            self.assertEqual(campaign._dataset_path(recipe, "eval"), evaluation.resolve())
            recipe["eval"]["path"] = str(alternate)
            with self.assertRaisesRegex(campaign.ContractError, "conflicting eval"):
                campaign._dataset_path(recipe, "eval")

    def test_metric_summary_reports_worst_case_and_population_spread(self) -> None:
        runs = [
            {"quality": {"metrics": {"eval_loss": value}}}
            for value in (1.0, 2.0, 3.0)
        ]
        summary = campaign._metric_summary(runs)["eval_loss"]
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["minimum"], 1.0)
        self.assertEqual(summary["maximum"], 3.0)
        self.assertGreater(summary["population_stddev"], 0.0)


if __name__ == "__main__":
    unittest.main()
