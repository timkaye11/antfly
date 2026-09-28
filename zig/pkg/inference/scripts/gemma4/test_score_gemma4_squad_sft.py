import copy
import unittest

from score_gemma4_squad_sft import answer_scores, score_pairs, score_seeds, sign_test


def prediction(qid, text, terminated=True):
    return {
        "id": qid,
        "text": text,
        "token_ids": [42, 106] if terminated else [42] * 65,
        "terminated": terminated,
    }


def case(count=12):
    references = [
        {"id": str(i), "article": str(i), "references": ["red fox"]}
        for i in range(count)
    ]
    baseline = [prediction(str(i), "green bear") for i in range(count)]
    trained = [prediction(str(i), "red fox") for i in range(count)]
    return references, baseline, trained


class SquadScoringTests(unittest.TestCase):
    def test_normalization_multiple_references_and_repeated_tokens(self):
        self.assertEqual(answer_scores("THE Red, fox!", ["red fox", "fox"]), (1.0, 1.0))
        self.assertEqual(answer_scores("fox fox", ["fox"]), (0.0, 2 / 3))
        self.assertEqual(answer_scores("red", ["blue", "red fox"]), (0.0, 2 / 3))
        self.assertEqual(answer_scores("", ["red fox"]), (0.0, 0.0))
        self.assertEqual(answer_scores("", ["the", "red fox"]), (0.0, 0.0))
        self.assertEqual(answer_scores("", ["the"]), (1.0, 1.0))

    def test_complete_improvement_passes_and_pairing_uses_identity(self):
        refs, baseline, trained = case()
        result = score_pairs(refs, baseline, list(reversed(trained)))
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["paired_article_test"]["one_sided_exact_p_value"], 1 / 4096
        )

    def test_one_article_is_not_twelve_independent_questions(self):
        refs, baseline, trained = case()
        for row in refs:
            row["article"] = "same article"
        result = score_pairs(refs, baseline, trained)
        self.assertFalse(result["passed"])
        self.assertEqual(result["paired_article_test"]["one_sided_exact_p_value"], 0.5)

    def test_truncated_correct_prefix_is_not_accepted(self):
        refs, baseline, _ = case()
        trained = [prediction(str(i), "red fox", False) for i in range(12)]
        result = score_pairs(refs, baseline, trained)
        self.assertFalse(result["passed"])
        self.assertEqual(result["evaluations"]["trained"]["token_f1"], 0)
        self.assertTrue(
            all(
                r["raw_token_f1"] == 1 for r in result["evaluations"]["trained"]["rows"]
            )
        )

    def test_termination_claim_must_match_tokens(self):
        refs, baseline, trained = case()
        trained[0]["token_ids"] = [42] * 65
        with self.assertRaisesRegex(ValueError, "termination"):
            score_pairs(refs, baseline, trained)

    def test_coverage_and_duplicate_prediction_fail_closed(self):
        refs, baseline, trained = case()
        with self.assertRaisesRegex(ValueError, "coverage"):
            score_pairs(refs, baseline, trained[:-1])
        trained[-1] = copy.deepcopy(trained[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_pairs(refs, baseline, trained)

    def test_duplicate_reference_id_rejected(self):
        refs, baseline, trained = case()
        refs[-1]["id"] = refs[0]["id"]
        with self.assertRaisesRegex(ValueError, "unique IDs"):
            score_pairs(refs, baseline, trained)

    def test_ties_cannot_establish_improvement(self):
        self.assertFalse(sign_test([0.0, 1e-14, -1e-14])["passed"])
        self.assertEqual(sign_test([0.0])["one_sided_exact_p_value"], 1.0)
        with self.assertRaises(ValueError):
            sign_test([float("nan")])

    def test_article_losses_are_not_hidden_by_many_correlated_wins(self):
        refs, baseline, trained = case(102)
        for i, row in enumerate(refs):
            if i < 100:
                row["article"] = "one winner"
            else:
                baseline[i], trained[i] = trained[i], baseline[i]
        result = score_pairs(refs, baseline, trained)
        self.assertTrue(result["checks"]["f1_improved"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["paired_article_test"]["losses"], 2)

    def test_seed_aggregation_keeps_article_count_and_requires_each_seed(self):
        refs, baseline, trained = case()
        predictions = {
            seed: {"baseline": baseline, "trained": trained} for seed in (17, 42, 991)
        }
        result = score_seeds(refs, predictions)
        self.assertTrue(result["passed"])
        self.assertEqual(result["aggregate_article_test"]["groups"], 12)
        self.assertEqual(
            result["aggregate_article_test"]["one_sided_exact_p_value"], 1 / 4096
        )
        predictions[991] = {"baseline": trained, "trained": trained}
        self.assertFalse(score_seeds(refs, predictions)["passed"])
        del predictions[991]
        with self.assertRaisesRegex(ValueError, "frozen seeds"):
            score_seeds(refs, predictions)


if __name__ == "__main__":
    unittest.main()
