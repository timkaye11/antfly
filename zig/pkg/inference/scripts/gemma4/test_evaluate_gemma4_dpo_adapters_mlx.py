from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_gemma4_dpo_adapters_mlx as evaluator  # noqa: E402


class Gemma4DpoMlxAdapterEvaluationTest(unittest.TestCase):
    def test_candidate_parser_is_closed_and_stable(self) -> None:
        candidate = evaluator.parse_candidate("antfly=/tmp/adapter.safetensors")
        self.assertEqual("antfly", candidate.label)
        self.assertEqual(Path("/tmp/adapter.safetensors"), candidate.path)
        for malformed in ("Antfly=/tmp/a", "antfly", "1bad=/tmp/a", "antfly="):
            with self.assertRaises(argparse.ArgumentTypeError):
                evaluator.parse_candidate(malformed)

    def test_pairwise_comparison_reports_decision_and_metric_deltas(self) -> None:
        left = {
            "accuracy": 0.5,
            "mean_loss": 0.4,
            "mean_reward_margin": 1.5,
            "rows": [
                {
                    "index": 0,
                    "loss": 0.2,
                    "reward_margin": 2.0,
                    "policy_chosen_logp": -1.0,
                    "policy_rejected_logp": -3.0,
                    "preferred": True,
                },
                {
                    "index": 1,
                    "loss": 0.6,
                    "reward_margin": -1.0,
                    "policy_chosen_logp": -4.0,
                    "policy_rejected_logp": -3.0,
                    "preferred": False,
                },
            ],
        }
        right = {
            "accuracy": 1.0,
            "mean_loss": 0.2,
            "mean_reward_margin": 1.0,
            "rows": [
                {
                    "index": 0,
                    "loss": 0.1,
                    "reward_margin": 1.0,
                    "policy_chosen_logp": -2.0,
                    "policy_rejected_logp": -3.0,
                    "preferred": True,
                },
                {
                    "index": 1,
                    "loss": 0.3,
                    "reward_margin": 1.0,
                    "policy_chosen_logp": -2.0,
                    "policy_rejected_logp": -3.0,
                    "preferred": True,
                },
            ],
        }
        comparison = evaluator.compare_evaluations(left, right)
        self.assertEqual(0.5, comparison["preference_decision_agreement"])
        self.assertEqual(0.5, comparison["accuracy_abs_delta"])
        self.assertAlmostEqual(2.0, comparison["mean_loss_ratio"])
        self.assertAlmostEqual(0.2, comparison["row_loss_mae"])


if __name__ == "__main__":
    unittest.main()
