from __future__ import annotations

import copy
import unittest

import evaluate_gemma4_sft_answers_mlx as quality


def row(index: int, correct: bool, following: int | None = 106) -> dict:
    target = 4443 if index % 2 else 1904
    prediction = target if correct else (1904 if target == 4443 else 4443)
    return {
        "source_group_id": f"group-{index}",
        "target_token_id": target,
        "predicted_token_id": prediction,
        "forced_choice_token_id": prediction,
        "next_token_id": following,
    }


def prepared() -> dict:
    rows = []
    for index, target in enumerate((4443, 1904)):
        prompt = [2, 800 + index]
        response = [target, 106, 107]
        rows.append(
            {
                "source_group_id": f"group-{index}",
                "prompt_input_ids": prompt,
                "response_input_ids": response,
                "input_ids": prompt + response,
                "labels": [-100] * len(prompt) + response,
                "num_supervised_tokens": 3,
                "was_truncated": False,
                "turn_count": 2,
            }
        )
    return {
        "schema_version": "gemma4_prepared/v6",
        "examples_seen": 2,
        "examples_truncated": 0,
        "examples": rows,
    }


class SftAnswerQualityTests(unittest.TestCase):
    def test_pairing_uses_identity_and_recomputes_correctness(self) -> None:
        baseline = [row(i, False) for i in range(12)]
        trained = [row(i, True) for i in reversed(range(12))]
        for item in baseline:
            item["correct"] = item["complete_answer_correct"] = True
        result = quality.score_answers(baseline, trained)
        self.assertTrue(result["passed"])
        self.assertEqual(0, result["evaluations"]["baseline"]["accuracy"])
        self.assertEqual(12, result["paired_test"]["wins"])
        self.assertEqual(2**-12, result["paired_test"]["one_sided_exact_p_value"])

    def test_rejects_improvement_without_significance_or_termination(self) -> None:
        before = [row(i, False) for i in range(4)]
        after = [row(i, True) for i in range(4)]
        self.assertFalse(quality.score_answers(before, after)["passed"])
        before = [row(i, False) for i in range(12)]
        after = [row(i, True, 107) for i in range(12)]
        result = quality.score_answers(before, after)
        self.assertTrue(result["first_token_improved"])
        self.assertEqual(0, result["paired_test"]["wins"])
        self.assertFalse(result["passed"])

    def test_forced_choice_regression_cannot_hide_behind_format_improvement(
        self,
    ) -> None:
        before = [row(i, True) for i in range(12)]
        for item in before:
            item["predicted_token_id"] = 999
            item["next_token_id"] = None
        after = [row(i, i < 11) for i in range(12)]
        result = quality.score_answers(before, after)
        self.assertTrue(result["paired_test"]["passed"])
        self.assertFalse(result["forced_choice_nonregression"])
        self.assertFalse(result["passed"])

    def test_accuracy_regression_and_ties_fail(self) -> None:
        before = [row(i, True) for i in range(12)]
        self.assertFalse(quality.score_answers(before, before)["passed"])
        result = quality.score_answers(before, [row(i, i < 6) for i in range(12)])
        self.assertFalse(result["passed"])
        self.assertEqual(6, result["paired_test"]["losses"])

    def test_rejects_missing_duplicate_or_relabelled_pairs(self) -> None:
        before = [row(i, True) for i in range(12)]
        for after in (
            before[:-1],
            before + [before[0]],
            [],
            [row(13, True)] + before[1:],
        ):
            with self.assertRaises(quality.AnswerQualityError):
                quality.score_answers(before, after)
        after = copy.deepcopy(before)
        after[0]["target_token_id"] = 4443
        with self.assertRaisesRegex(quality.AnswerQualityError, "targets differ"):
            quality.score_answers(before, after)
        after[0]["target_token_id"] = True
        with self.assertRaises(quality.AnswerQualityError):
            quality.score_answers(before, after)

    def test_prepared_examples_require_exact_supervision_and_unique_prompts(
        self,
    ) -> None:
        summary = prepared()
        self.assertEqual([1904, 4443], quality.validate_examples(summary, 2)[1])
        for field, value in (
            ("was_truncated", True),
            ("turn_count", 3),
            ("labels", [-100, -100, 4443, 106, -100]),
            ("num_supervised_tokens", 2),
            ("response_input_ids", [4443, 106, 108]),
        ):
            altered = copy.deepcopy(summary)
            altered["examples"][0][field] = value
            with (
                self.subTest(field=field),
                self.assertRaises(quality.AnswerQualityError),
            ):
                quality.validate_examples(altered, 2)
        for field in ("source_group_id", "prompt_input_ids"):
            altered = copy.deepcopy(summary)
            altered["examples"][1][field] = altered["examples"][0][field]
            with (
                self.subTest(field=field),
                self.assertRaises(quality.AnswerQualityError),
            ):
                quality.validate_examples(altered, 2)
        with self.assertRaises(quality.AnswerQualityError):
            quality.validate_examples(summary, 3)


if __name__ == "__main__":
    unittest.main()
