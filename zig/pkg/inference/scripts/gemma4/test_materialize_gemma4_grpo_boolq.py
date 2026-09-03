from __future__ import annotations

import json
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

    def test_balanced_offset_rotates_to_a_disjoint_source_order_slice(self) -> None:
        first = materializer.select_balanced(
            rows(),
            split="validation",
            tokenizer=FakeTokenizer(),
            examples=4,
            max_seq_len=128,
            max_completion_tokens=1,
        )
        rotated = materializer.select_balanced(
            rows(),
            split="validation",
            tokenizer=FakeTokenizer(),
            examples=4,
            skip_per_label=2,
            max_seq_len=128,
            max_completion_tokens=1,
        )
        self.assertEqual([4, 5, 6, 7], [item.source_row_index for item in rotated])
        self.assertFalse(
            {item.source_id for item in first}
            & {item.source_id for item in rotated}
        )
        with self.assertRaisesRegex(
            materializer.MaterializationError, "per-label skip"
        ):
            materializer.select_balanced(
                rows(),
                split="validation",
                tokenizer=FakeTokenizer(),
                examples=4,
                skip_per_label=-1,
                max_seq_len=128,
                max_completion_tokens=1,
            )

    def test_exact_source_exclusions_survive_admission_contract_changes(self) -> None:
        first = materializer.select_balanced(
            rows(),
            split="validation",
            tokenizer=FakeTokenizer(),
            examples=4,
            max_seq_len=128,
            max_completion_tokens=1,
        )
        rotated = materializer.select_balanced(
            rows(),
            split="validation",
            tokenizer=FakeTokenizer(),
            examples=4,
            excluded_source_ids=frozenset(item.source_id for item in first),
            max_seq_len=128,
            max_completion_tokens=1,
        )
        self.assertEqual([4, 5, 6, 7], [item.source_row_index for item in rotated])
        self.assertFalse(
            {item.source_id for item in first}
            & {item.source_id for item in rotated}
        )

    def test_exclusion_manifest_is_validated_and_digest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            source_ids = ["a" * 64, "b" * 64]
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "antfly_gemma4_grpo_boolq_materialization/v1",
                        "eval_source_ids": source_ids,
                    }
                ),
                encoding="utf-8",
            )
            excluded, evidence = materializer.load_excluded_eval_source_ids([path])
            self.assertEqual(set(source_ids), set(excluded))
            self.assertEqual(materializer.sha256_file(path), evidence[0]["sha256"])
            self.assertEqual(2, evidence[0]["source_id_count"])

    def test_v2_selection_contract_rechecks_exclusion_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exclusion = root / "prior.json"
            excluded_ids = ["a" * 64, "b" * 64]
            exclusion.write_text(
                json.dumps(
                    {
                        "schema_version": materializer.SCHEMA_VERSION_V1,
                        "eval_source_ids": excluded_ids,
                    }
                ),
                encoding="utf-8",
            )
            excluded, evidence = materializer.load_excluded_eval_source_ids(
                [exclusion]
            )
            manifest = {
                "schema_version": materializer.SCHEMA_VERSION,
                "dataset": {
                    "selection_policy": {
                        "ordering": materializer.V2_SELECTION_ORDERING,
                        "train_skip_per_label": 0,
                        "evaluation_skip_per_label": 2,
                        "evaluation_excluded_source_ids": len(excluded),
                        "evaluation_exclusion_evidence_source_ids": 2,
                    }
                },
                "train_source_ids": ["c" * 64],
                "eval_source_ids": ["d" * 64],
                "evaluation_exclusion_manifests": list(evidence),
            }
            materializer.validate_materialization_selection_contract(manifest)

            manifest["eval_source_ids"] = [excluded_ids[0]]
            with self.assertRaisesRegex(
                materializer.MaterializationError, "overlap the exclusion set"
            ):
                materializer.validate_materialization_selection_contract(manifest)

    def test_exclusion_manifest_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "schema_version": materializer.SCHEMA_VERSION_V1,
                        "eval_source_ids": ["a" * 64],
                    }
                ),
                encoding="utf-8",
            )
            alias = root / "alias.json"
            alias.symlink_to(source)
            with self.assertRaisesRegex(
                materializer.MaterializationError, "cannot be a symlink"
            ):
                materializer.load_excluded_eval_source_ids([alias])

    def test_overlapping_exclusion_manifests_report_unique_and_evidence_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first.json"
            second = root / "second.json"
            for path, source_ids in (
                (first, ["a" * 64, "b" * 64]),
                (second, ["b" * 64, "c" * 64]),
            ):
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": materializer.SCHEMA_VERSION,
                            "eval_source_ids": source_ids,
                        }
                    ),
                    encoding="utf-8",
                )
            excluded, evidence = materializer.load_excluded_eval_source_ids(
                [first, second]
            )
            self.assertEqual(3, len(excluded))
            self.assertEqual(4, sum(item["source_id_count"] for item in evidence))
            with self.assertRaisesRegex(
                materializer.MaterializationError, "is repeated"
            ):
                materializer.load_excluded_eval_source_ids([first, first])

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
                train_skip_per_label=0,
                eval_skip_per_label=0,
                eval_excluded_source_id_count=0,
                eval_exclusion_evidence=(),
                max_seq_len=128,
                max_completion_tokens=1,
                dependency_versions={"pyarrow": "test", "tokenizers": "test"},
            )
            self.assertEqual(materializer.sha256_file(train_source), manifest["dataset"]["train"]["source_file_sha256"])
            self.assertEqual(
                0,
                manifest["dataset"]["selection_policy"][
                    "evaluation_excluded_source_ids"
                ],
            )
            self.assertEqual(
                0,
                manifest["dataset"]["selection_policy"][
                    "evaluation_exclusion_evidence_source_ids"
                ],
            )
            self.assertTrue((output / "manifest.json").is_file())
            materializer.validate_materialization_semantic_sha256(manifest)
            materializer.validate_materialization_selection_contract(manifest)
            manifest["dependency_versions"]["pyarrow"] = "drift"
            with self.assertRaisesRegex(
                materializer.MaterializationError, "semantic SHA-256 drifted"
            ):
                materializer.validate_materialization_semantic_sha256(manifest)
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
                    train_skip_per_label=0,
                    eval_skip_per_label=0,
                    eval_excluded_source_id_count=0,
                    eval_exclusion_evidence=(),
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
