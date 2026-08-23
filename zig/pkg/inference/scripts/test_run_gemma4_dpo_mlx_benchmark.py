from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_gemma4_dpo_mlx_benchmark as benchmark  # noqa: E402


class Gemma4DpoMlxBenchmarkTest(unittest.TestCase):
    def load_payload(self) -> dict:
        return json.loads(benchmark.DEFAULT_CASE_PATH.read_text(encoding="utf-8"))

    def write_case(self, payload: dict, root: Path) -> Path:
        path = root / "case.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def dataset_payload(self, root: Path, *, examples: int = 5) -> dict:
        legacy = self.load_payload()
        example = {
            "rendered_prompt": legacy.pop("rendered_prompt"),
            "prompt_token_ids": legacy.pop("prompt_token_ids"),
            "chosen": legacy.pop("chosen"),
            "rejected": legacy.pop("rejected"),
        }
        rows = []
        for index in range(examples):
            item = copy.deepcopy(example)
            item.update(
                {
                    "source_row_index": index,
                    "source_id": f"{index + 1:064x}",
                    "score_chosen": 8.0,
                    "score_rejected": 4.0,
                }
            )
            rows.append(item)
        jsonl = root / "preferences.jsonl"
        jsonl.write_text('{"real":"preference"}\n', encoding="utf-8")
        legacy["schema_version"] = benchmark.DATASET_CASE_SCHEMA_VERSION
        legacy["dataset"] = {
            "repo_id": "HuggingFaceH4/ultrafeedback_binarized",
            "revision": "a" * 40,
            "split": "test_prefs",
            "source_file": "data/test.parquet",
            "source_file_sha256": "b" * 64,
            "source_rows": 2000,
            "materialized_jsonl": jsonl.name,
            "materialized_jsonl_sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
            "selection_policy": {"ordering": "source-order"},
        }
        legacy["examples"] = rows
        return legacy

    def test_canonical_case_is_stable(self) -> None:
        case = benchmark.load_case(benchmark.DEFAULT_CASE_PATH)
        self.assertEqual("gemma-4-E2B-it", case.model_key)
        self.assertEqual("peft-qv", case.target_preset)
        self.assertEqual(128, case.sequence_length)
        self.assertEqual(benchmark.FIXED_PROTOCOL, case.protocol)
        self.assertEqual(
            "sha256:e765223a1c30e5e2cb889d9b19548a52f8301502340ff491eb8e8dfe02fb6f7a",
            case.semantic_sha256,
        )

    def test_padding_supervises_only_completion_tokens(self) -> None:
        case = benchmark.load_case(benchmark.DEFAULT_CASE_PATH)
        ids, labels = benchmark.padded_sequence(
            case.prompt_token_ids, case.chosen_token_ids, case.sequence_length
        )
        completion_index = len(case.prompt_token_ids)
        self.assertEqual(case.sequence_length, len(ids))
        self.assertEqual(case.sequence_length, len(labels))
        self.assertEqual(list(case.prompt_token_ids), ids[:completion_index])
        self.assertEqual(list(case.chosen_token_ids), ids[completion_index : completion_index + 1])
        self.assertTrue(all(label == -100 for label in labels[:completion_index]))
        self.assertEqual(list(case.chosen_token_ids), labels[completion_index : completion_index + 1])
        self.assertTrue(all(label == -100 for label in labels[completion_index + 1 :]))

    def test_dpo_evaluation_metrics_match_closed_form(self) -> None:
        metrics = benchmark.dpo_evaluation_metrics(
            policy_chosen=[-2.0, -1.0],
            policy_rejected=[-3.0, -4.0],
            reference_chosen=[-2.0, -2.0],
            reference_rejected=[-3.0, -3.0],
            beta=0.1,
        )
        self.assertEqual(2, metrics["examples"])
        self.assertAlmostEqual(math.log(2.0), metrics["rows"][0]["loss"])
        self.assertAlmostEqual(0.2, metrics["rows"][1]["reward_margin"])
        self.assertAlmostEqual(0.5, metrics["accuracy"])
        expected = (math.log(2.0) + math.log1p(math.exp(-0.2))) / 2.0
        self.assertAlmostEqual(expected, metrics["mean_loss"])

    def test_dpo_evaluation_rejects_vector_drift(self) -> None:
        with self.assertRaisesRegex(
            benchmark.DpoBenchmarkContractError, "non-empty and equally sized"
        ):
            benchmark.dpo_evaluation_metrics([1.0], [], [1.0], [1.0], 0.1)

    def test_case_rejects_unknown_fields(self) -> None:
        payload = self.load_payload()
        payload["typo"] = True
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                benchmark.DpoBenchmarkContractError, "fields drifted"
            ):
                benchmark.load_case(self.write_case(payload, Path(temp)))

    def test_case_rejects_protocol_drift(self) -> None:
        payload = self.load_payload()
        payload["protocol"]["measured"] = 19
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                benchmark.DpoBenchmarkContractError, "fixed"
            ):
                benchmark.load_case(self.write_case(payload, Path(temp)))

    def test_case_rejects_identical_preferences(self) -> None:
        payload = self.load_payload()
        payload["rejected"] = copy.deepcopy(payload["chosen"])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                benchmark.DpoBenchmarkContractError, "must differ"
            ):
                benchmark.load_case(self.write_case(payload, Path(temp)))

    def test_case_rejects_overlength_rows(self) -> None:
        payload = self.load_payload()
        payload["sequence_length"] = len(payload["prompt_token_ids"])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                benchmark.DpoBenchmarkContractError, "exceeds sequence_length"
            ):
                benchmark.load_case(self.write_case(payload, Path(temp)))

    def test_dataset_case_binds_examples_and_materialized_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = benchmark.load_case(
                self.write_case(self.dataset_payload(root), root)
            )
            self.assertEqual(benchmark.DATASET_CASE_SCHEMA_VERSION, case.schema_version)
            self.assertEqual(5, len(case.examples))
            self.assertEqual([0, 1, 2, 3, 4], [item.source_row_index for item in case.examples])
            self.assertEqual(
                "HuggingFaceH4/ultrafeedback_binarized", case.dataset["repo_id"]
            )

    def test_dataset_case_rejects_jsonl_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = self.dataset_payload(root)
            path = self.write_case(payload, root)
            (root / "preferences.jsonl").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                benchmark.DpoBenchmarkContractError, "JSONL SHA-256 drifted"
            ):
                benchmark.load_case(path)

    def test_dataset_case_rejects_non_divisor_example_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(
                benchmark.DpoBenchmarkContractError, "divide the fixed update count"
            ):
                benchmark.load_case(
                    self.write_case(self.dataset_payload(root, examples=3), root)
                )

    def test_dataset_case_rejects_out_of_order_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = self.dataset_payload(root)
            payload["examples"][0], payload["examples"][1] = (
                payload["examples"][1],
                payload["examples"][0],
            )
            with self.assertRaisesRegex(
                benchmark.DpoBenchmarkContractError, "source-order"
            ):
                benchmark.load_case(self.write_case(payload, root))

    def test_exclusive_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            benchmark.write_json_exclusive(path, {"run": 1})
            self.assertEqual({"run": 1}, json.loads(path.read_text(encoding="utf-8")))
            with self.assertRaisesRegex(
                benchmark.DpoBenchmarkContractError, "already exists"
            ):
                benchmark.write_json_exclusive(path, {"run": 2})

    def test_adapter_output_uses_exact_antfly_names_and_refuses_overwrite(self) -> None:
        class FakeTensor:
            @property
            def T(self):
                return self

            def astype(self, _dtype):
                return self

        class FakeMlx:
            float32 = "float32"

            @staticmethod
            def save_safetensors(path, tensors, metadata):
                Path(path).write_text(
                    json.dumps(
                        {
                            "metadata": metadata,
                            "names": sorted(tensors),
                        }
                    ),
                    encoding="utf-8",
                )

        target = "model.language_model.layers.0.self_attn.q_proj"
        canonical = benchmark.locked.canonicalize_module_name(target)
        adapter = SimpleNamespace(
            tensors={
                (canonical, "lora_A"): SimpleNamespace(
                    source_name=f"{target}.weight.lora_A.weight"
                ),
                (canonical, "lora_B"): SimpleNamespace(
                    source_name=f"{target}.weight.lora_B.weight"
                ),
            }
        )
        trainables = {
            f"{target}.lora_a": FakeTensor(),
            f"{target}.lora_b": FakeTensor(),
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "adapter.safetensors"
            digest = benchmark.write_adapter_exclusive(
                output,
                final_trainables=trainables,
                target_names=[target],
                adapter=adapter,
                mx=FakeMlx(),
            )
            self.assertTrue(output.is_file())
            self.assertEqual(
                [
                    f"{target}.weight.lora_A.weight",
                    f"{target}.weight.lora_B.weight",
                ],
                json.loads(output.read_text(encoding="utf-8"))["names"],
            )
            self.assertEqual(
                "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(),
                digest,
            )
            with self.assertRaisesRegex(
                benchmark.DpoBenchmarkContractError, "already exists"
            ):
                benchmark.write_adapter_exclusive(
                    output,
                    final_trainables=trainables,
                    target_names=[target],
                    adapter=adapter,
                    mx=FakeMlx(),
                )

    def test_first_and_final_adapter_outputs_must_differ(self) -> None:
        output = Path("same.safetensors")
        args = SimpleNamespace(
            case=benchmark.DEFAULT_CASE_PATH,
            first_update_adapter_output=output,
            adapter_output=output,
        )
        with self.assertRaisesRegex(
            benchmark.DpoBenchmarkContractError, "must be different paths"
        ):
            benchmark.run(args)

    def test_source_revision_requires_clean_pinned_checkout(self) -> None:
        expected = "a" * 40
        with mock.patch.object(
            benchmark.locked,
            "verify_source_checkout",
            return_value={"path": "/tmp/mlx", "revision": expected},
        ) as verify:
            self.assertEqual(
                expected,
                benchmark.require_source_revision(Path("/tmp/mlx"), expected, "MLX"),
            )
        verify.assert_called_once_with(
            Path("/tmp/mlx"), expected, source_name="MLX"
        )

        with mock.patch.object(
            benchmark.locked,
            "verify_source_checkout",
            side_effect=benchmark.locked.ContractError("checkout must be clean"),
        ):
            with self.assertRaisesRegex(
                benchmark.DpoBenchmarkContractError, "clean MLX source revision"
            ):
                benchmark.require_source_revision(Path("/tmp/mlx"), expected, "MLX")


if __name__ == "__main__":
    unittest.main()
