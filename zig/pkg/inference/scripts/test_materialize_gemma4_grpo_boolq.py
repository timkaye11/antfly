from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import materialize_gemma4_grpo_boolq as materializer


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        if add_special_tokens:
            raise AssertionError("hidden special tokens are forbidden")
        if text in ("yes", "no"):
            return SimpleNamespace(ids=[1])
        return SimpleNamespace(ids=list(range(max(1, len(text.split())))))


def rows(count: int = 12) -> list[dict]:
    return [
        {
            "passage": f"Short factual passage number {index}.",
            "question": f"is statement {index} supported",
            "answer": index % 2 == 0,
        }
        for index in range(count)
    ]


class BoolQMaterializerTest(unittest.TestCase):
    def test_selection_is_balanced_source_order_and_disjoint_by_split(self) -> None:
        train = materializer.select_balanced(
            rows(),
            split="train",
            tokenizer=FakeTokenizer(),
            examples=4,
            max_seq_len=128,
            max_completion_tokens=1,
        )
        evaluation = materializer.select_balanced(
            rows(),
            split="validation",
            tokenizer=FakeTokenizer(),
            examples=4,
            max_seq_len=128,
            max_completion_tokens=1,
        )
        self.assertEqual([0, 1, 2, 3], [item.source_row_index for item in train])
        self.assertTrue(train[0].prompt.startswith("<bos><|turn>user\n"))
        self.assertTrue(train[0].prompt.endswith("<|channel>final\n<channel|>"))
        self.assertEqual({"yes": 2, "no": 2}, {
            label: sum(item.target == label for item in train) for label in ("yes", "no")
        })
        self.assertFalse({item.source_id for item in train} & {item.source_id for item in evaluation})

    def test_long_prompt_is_skipped_without_truncation(self) -> None:
        source = rows(6)
        source.insert(0, {"passage": "word " * 200, "question": "long", "answer": True})
        selected = materializer.select_balanced(
            source,
            split="train",
            tokenizer=FakeTokenizer(),
            examples=4,
            max_seq_len=64,
            max_completion_tokens=1,
        )
        self.assertNotIn(0, [item.source_row_index for item in selected])

    def test_multi_token_rollout_budget_is_admitted_and_bound(self) -> None:
        materializer.validate_length_contract(128, 4)
        selected = materializer.select_balanced(
            rows(),
            split="train",
            tokenizer=FakeTokenizer(),
            examples=4,
            max_seq_len=128,
            max_completion_tokens=4,
        )
        self.assertEqual(4, len(selected))
        self.assertTrue(all(item.target_tokens == 1 for item in selected))

    def test_invalid_rollout_budgets_fail_closed(self) -> None:
        for completion_tokens in (0, materializer.MAX_COMPLETION_TOKENS + 1):
            with self.subTest(completion_tokens=completion_tokens):
                with self.assertRaisesRegex(
                    materializer.MaterializationError, "max completion tokens"
                ):
                    materializer.validate_length_contract(128, completion_tokens)
        with self.assertRaisesRegex(
            materializer.MaterializationError, "smaller than max sequence length"
        ):
            materializer.validate_length_contract(16, 16)

    def test_write_outputs_refuses_stale_directory_and_binds_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_source = root / "train.parquet"
            eval_source = root / "eval.parquet"
            train_source.write_bytes(b"train")
            eval_source.write_bytes(b"eval")
            model = root / "model"
            model.mkdir()
            (model / "tokenizer.json").write_text("{}")
            (model / "tokenizer_config.json").write_text("{}")
            train = materializer.select_balanced(
                rows(), split="train", tokenizer=FakeTokenizer(), examples=4,
                max_seq_len=128, max_completion_tokens=1,
            )
            evaluation = materializer.select_balanced(
                rows(), split="validation", tokenizer=FakeTokenizer(), examples=4,
                max_seq_len=128, max_completion_tokens=1,
            )
            output = root / "out"
            manifest = materializer.write_outputs(
                output_dir=output,
                train_parquet=train_source,
                eval_parquet=eval_source,
                model_dir=model,
                revision="a" * 40,
                train_rows=12,
                eval_rows=12,
                train_examples=train,
                eval_examples=evaluation,
                max_seq_len=128,
                max_completion_tokens=1,
                dependency_versions={"pyarrow": "test", "tokenizers": "test"},
            )
            self.assertEqual(materializer.sha256_file(train_source), manifest["dataset"]["train"]["source_file_sha256"])
            self.assertTrue((output / "manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                materializer.write_outputs(
                    output_dir=output,
                    train_parquet=train_source,
                    eval_parquet=eval_source,
                    model_dir=model,
                    revision="a" * 40,
                    train_rows=12,
                    eval_rows=12,
                    train_examples=train,
                    eval_examples=evaluation,
                    max_seq_len=128,
                    max_completion_tokens=1,
                    dependency_versions={},
                )

    def test_invalid_target_tokenization_fails_closed(self) -> None:
        class BadTokenizer(FakeTokenizer):
            def encode(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
                if text == "yes":
                    return SimpleNamespace(ids=[1, 2])
                return super().encode(text, add_special_tokens=add_special_tokens)

        with self.assertRaisesRegex(materializer.MaterializationError, "one tokenizer token"):
            materializer.select_balanced(
                rows(), split="train", tokenizer=BadTokenizer(), examples=2,
                max_seq_len=128, max_completion_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()
