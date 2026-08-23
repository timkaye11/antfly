from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import materialize_gemma4_dpo_hf_parity as materializer  # noqa: E402


class FakeTokenizer:
    def __init__(self, values: dict[str, list[int]]) -> None:
        self.values = values

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("parity tokenization must not add hidden special tokens")
        return self.values[text]


def example(
    *,
    row: int,
    prompt_tokens: int,
    chosen_tokens: int,
    rejected_tokens: int,
    score_margin: float = 2.0,
) -> materializer.TokenizedPreference:
    return materializer.TokenizedPreference(
        source_row_index=row,
        source_id=f"{row:064x}",
        score_chosen=8.0,
        score_rejected=8.0 - score_margin,
        prompt=f"prompt-{row}",
        rendered_prompt=f"rendered-{row}",
        prompt_token_ids=tuple(range(prompt_tokens)),
        chosen_text=f"chosen-{row}",
        chosen_token_ids=tuple(range(chosen_tokens)),
        rejected_text=f"rejected-{row}",
        rejected_token_ids=tuple(range(1, rejected_tokens + 1)),
    )


class Gemma4DpoHfMaterializerTest(unittest.TestCase):
    def test_renderer_matches_locked_gemma4_prompt(self) -> None:
        self.assertEqual(
            "<bos><|turn>user\nAnswer with one word: yes or no?"
            "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
            materializer.render_gemma4_user_prompt(
                materializer.CONTRACT_PROBE_PROMPT
            ),
        )

    def test_selection_is_stratified_then_restored_to_source_order(self) -> None:
        candidates = [
            example(row=3, prompt_tokens=20, chosen_tokens=50, rejected_tokens=50),
            example(row=4, prompt_tokens=150, chosen_tokens=100, rejected_tokens=100),
            example(row=7, prompt_tokens=100, chosen_tokens=70, rejected_tokens=70),
            example(row=8, prompt_tokens=300, chosen_tokens=70, rejected_tokens=70),
            example(row=9, prompt_tokens=60, chosen_tokens=40, rejected_tokens=40),
            example(row=11, prompt_tokens=400, chosen_tokens=50, rejected_tokens=50),
        ]
        selected = materializer.select_length_stratified(
            candidates, sequence_length=512
        )
        self.assertEqual([3, 4, 7, 8, 11], [item.source_row_index for item in selected])
        self.assertEqual([70, 250, 170, 370, 450], [item.total_tokens for item in selected])

    def test_selection_rejects_weak_or_unbalanced_preferences(self) -> None:
        self.assertFalse(
            materializer.admitted_candidate(
                example(
                    row=1,
                    prompt_tokens=40,
                    chosen_tokens=40,
                    rejected_tokens=40,
                    score_margin=0.5,
                ),
                512,
            )
        )
        self.assertFalse(
            materializer.admitted_candidate(
                example(row=2, prompt_tokens=40, chosen_tokens=160, rejected_tokens=32),
                512,
            )
        )
        self.assertFalse(
            materializer.admitted_candidate(
                example(row=3, prompt_tokens=480, chosen_tokens=32, rejected_tokens=32),
                511,
            )
        )
        identical = example(
            row=4, prompt_tokens=40, chosen_tokens=32, rejected_tokens=32
        )
        identical = materializer.TokenizedPreference(
            **{
                **identical.__dict__,
                "rejected_token_ids": identical.chosen_token_ids,
            }
        )
        self.assertFalse(materializer.admitted_candidate(identical, 512))

    def test_selection_can_materialize_disjoint_bucket_occurrence(self) -> None:
        candidates = []
        totals = [80, 160, 240, 340, 440]
        for occurrence in range(2):
            for bucket, total in enumerate(totals):
                candidates.append(
                    example(
                        row=occurrence * len(totals) + bucket,
                        prompt_tokens=40,
                        chosen_tokens=total - 40,
                        rejected_tokens=total - 40,
                    )
                )
        selected = materializer.select_length_stratified(
            candidates,
            sequence_length=512,
            admitted_occurrence=1,
        )
        self.assertEqual([5, 6, 7, 8, 9], [item.source_row_index for item in selected])
        with self.assertRaisesRegex(
            materializer.MaterializationError, "must be nonnegative"
        ):
            materializer.select_length_stratified(
                candidates,
                sequence_length=512,
                admitted_occurrence=-1,
            )

    def test_tokenizer_contract_detects_drift(self) -> None:
        payload = json.loads(
            materializer.DEFAULT_TOKENIZER_CONTRACT.read_text(encoding="utf-8")
        )
        values = {
            payload["rendered_prompt"]: payload["prompt_token_ids"],
            payload["chosen"]["text"]: payload["chosen"]["token_ids"],
            payload["rejected"]["text"]: payload["rejected"]["token_ids"],
        }
        materializer.validate_tokenizer_contract(
            FakeTokenizer(values), materializer.DEFAULT_TOKENIZER_CONTRACT
        )
        values[payload["chosen"]["text"]] = [999]
        with self.assertRaisesRegex(materializer.MaterializationError, "chosen"):
            materializer.validate_tokenizer_contract(
                FakeTokenizer(values), materializer.DEFAULT_TOKENIZER_CONTRACT
            )

    def test_source_row_requires_exact_two_message_preferences(self) -> None:
        prompt = "hello"
        rendered = materializer.render_gemma4_user_prompt(prompt)
        tokenizer = FakeTokenizer(
            {rendered: [2, 3], "preferred": [4, 5], "other": [6, 7]}
        )
        row = {
            "prompt": prompt,
            "prompt_id": "a" * 64,
            "chosen": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "preferred"},
            ],
            "rejected": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "other"},
            ],
            "score_chosen": 8.0,
            "score_rejected": 3.0,
        }
        parsed = materializer.tokenize_source_row(row, 12, tokenizer)
        self.assertEqual(12, parsed.source_row_index)
        self.assertEqual((2, 3), parsed.prompt_token_ids)
        bad = dict(row)
        bad["chosen"] = row["chosen"][:1]
        with self.assertRaisesRegex(materializer.MaterializationError, "one user"):
            materializer.tokenize_source_row(bad, 12, tokenizer)

    def test_write_outputs_refuses_overwrite_and_binds_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parquet = root / "test.parquet"
            parquet.write_bytes(b"immutable-source")
            model = root / "model"
            model.mkdir()
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            selected = [
                example(row=index, prompt_tokens=40, chosen_tokens=32, rejected_tokens=32)
                for index in range(5)
            ]
            output = root / "out"
            manifest = materializer.write_outputs(
                output_dir=output,
                parquet_path=parquet,
                source_rows=2000,
                dataset_id=materializer.EXPECTED_DATASET_ID,
                revision="a" * 40,
                split=materializer.EXPECTED_SPLIT,
                model_dir=model,
                sequence_length=512,
                examples=selected,
            )
            self.assertEqual(
                materializer.sha256_file(parquet),
                manifest["dataset"]["source_file_sha256"],
            )
            self.assertEqual("gemma-4-E2B-it", manifest["model_key"])
            self.assertEqual(5, len(json.loads((output / "mlx_case.json").read_text())["examples"]))

            e4_output = root / "e4-out"
            e4_manifest = materializer.write_outputs(
                output_dir=e4_output,
                parquet_path=parquet,
                source_rows=2000,
                dataset_id=materializer.EXPECTED_DATASET_ID,
                revision="a" * 40,
                split=materializer.EXPECTED_SPLIT,
                model_dir=model,
                sequence_length=512,
                examples=selected,
                model_key="gemma-4-E4B-it",
            )
            e4_case = json.loads((e4_output / "mlx_case.json").read_text())
            self.assertEqual("gemma-4-E4B-it", e4_manifest["model_key"])
            self.assertEqual("gemma-4-E4B-it", e4_case["model_key"])
            self.assertIn("gemma4-e4b-", e4_case["name"])
            with self.assertRaises(FileExistsError):
                materializer.write_outputs(
                    output_dir=output,
                    parquet_path=parquet,
                    source_rows=2000,
                    dataset_id=materializer.EXPECTED_DATASET_ID,
                    revision="a" * 40,
                    split=materializer.EXPECTED_SPLIT,
                    model_dir=model,
                    sequence_length=512,
                    examples=selected,
                )


if __name__ == "__main__":
    unittest.main()
