import unittest
import hashlib
from pathlib import Path
import tempfile

from evaluate_gemma4_squad_mlx import greedy_tokens, validate_prepared
from gemma4_oracle_contract import fingerprint_dataset_source


def fixture(source):
    refs = [{"id": "q0"}, {"id": "q1"}]
    rows = []
    for i in range(2):
        prompt, response = [2, 100 + i], [42, 43, 106, 107]
        rows.append(
            {
                "source_id": "q" + str(i),
                "source_group_id": "g" + str(i),
                "prompt_input_ids": prompt,
                "response_input_ids": response,
                "input_ids": prompt + response,
                "labels": [-100, -100] + response,
                "num_supervised_tokens": 4,
                "was_truncated": False,
                "turn_count": 2,
            }
        )
    summary = {
        "schema_version": "gemma4_prepared/v6",
        "examples_seen": 2,
        "examples_truncated": 0,
        "source_dataset_sha256": fingerprint_dataset_source(source, "eval"),
        "source_split": "eval",
        "examples": rows,
    }
    selection = {
        "splits": {
            "eval": {
                "examples": 2,
                "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        }
    }
    return summary, refs, selection


class SquadGenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dataset = Path(temporary.name) / "eval.jsonl"
        self.dataset.write_text("{}\n")

    def test_cached_decode_prefills_once_then_sends_last_token(self):
        calls = []
        answers = iter([42, 43, 106])

        def next_token(inputs):
            calls.append(inputs)
            return next(answers)

        self.assertEqual(
            greedy_tokens([2, 3], next_token, use_cache=True), [42, 43, 106]
        )
        self.assertEqual(calls, [[2, 3], [42], [43]])

    def test_uncached_decode_replays_full_prefix(self):
        calls = []
        answers = iter([42, 43, 106])

        def next_token(inputs):
            calls.append(inputs)
            return next(answers)

        greedy_tokens([2, 3], next_token, use_cache=False)
        self.assertEqual(calls, [[2, 3], [2, 3, 42], [2, 3, 42, 43]])

    def test_budget_allows_64_answer_tokens_then_end_of_turn(self):
        answers = iter([42] * 64 + [106])
        tokens = greedy_tokens([2], lambda _: next(answers), use_cache=True)
        self.assertEqual(len(tokens), 65)
        self.assertEqual(tokens[-1], 106)
        self.assertEqual(len(greedy_tokens([2], lambda _: 42, use_cache=True)), 65)

    def test_native_prepared_inputs_are_paired_by_source_identity(self):
        summary, refs, selection = fixture(self.dataset)
        summary["examples"].reverse()
        result = validate_prepared(summary, refs, selection, self.dataset)
        self.assertEqual([row["source_id"] for row in result], ["q0", "q1"])

    def test_corrupt_supervision_rejected(self):
        summary, refs, selection = fixture(self.dataset)
        summary["examples"][0]["labels"][0] = 2
        with self.assertRaisesRegex(ValueError, "supervision"):
            validate_prepared(summary, refs, selection, self.dataset)

    def test_unrelated_prepared_dataset_rejected(self):
        summary, refs, selection = fixture(self.dataset)
        summary["source_dataset_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "source dataset SHA"):
            validate_prepared(summary, refs, selection, self.dataset)

    def test_duplicate_reference_rejected_before_model_load(self):
        summary, refs, selection = fixture(self.dataset)
        refs.append(refs[0])
        with self.assertRaisesRegex(ValueError, "identities or counts"):
            validate_prepared(summary, refs, selection, self.dataset)

    def test_changed_source_bytes_rejected_even_if_native_digest_is_rebound(self):
        summary, refs, selection = fixture(self.dataset)
        self.dataset.write_text('{"changed":true}\n')
        summary["source_dataset_sha256"] = fingerprint_dataset_source(
            self.dataset, "eval"
        )
        with self.assertRaisesRegex(ValueError, "frozen selection"):
            validate_prepared(summary, refs, selection, self.dataset)


if __name__ == "__main__":
    unittest.main()
