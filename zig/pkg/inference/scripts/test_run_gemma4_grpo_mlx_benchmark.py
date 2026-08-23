from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_gemma4_grpo_mlx_benchmark as benchmark  # noqa: E402


class Gemma4GrpoMlxBenchmarkTest(unittest.TestCase):
    def load_payload(self) -> dict:
        return json.loads(benchmark.DEFAULT_CASE_PATH.read_text(encoding="utf-8"))

    def write_case(self, payload: dict, root: Path) -> Path:
        path = root / "case.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_canonical_case_is_stable(self) -> None:
        case = benchmark.load_case(benchmark.DEFAULT_CASE_PATH)
        self.assertEqual("gemma-4-E2B-it", case.model_key)
        self.assertEqual("peft-qv", case.target_preset)
        self.assertEqual(128, case.sequence_length)
        self.assertEqual(2, case.group_size)
        self.assertEqual(1, case.max_completion_tokens)
        self.assertEqual((7001, 711), case.expected_initial_token_ids)
        self.assertEqual(benchmark.FIXED_PROTOCOL, case.protocol)
        self.assertEqual(
            "sha256:58b28cfb2f4f9654c803647ec9a97951dd3ecaeda2f93e9a9d436dd4b7565496",
            case.semantic_sha256,
        )

    def test_reward_and_advantage_contract_is_non_degenerate(self) -> None:
        case = benchmark.load_case(benchmark.DEFAULT_CASE_PATH)
        rewards = [
            case.reward_for_token(token_id)
            for token_id in case.expected_initial_token_ids
        ]
        advantages = benchmark.normalized_advantages(
            rewards, case.advantage_epsilon
        )
        self.assertEqual([1.0, 0.0], rewards)
        self.assertAlmostEqual(1.0, advantages[0], places=6)
        self.assertAlmostEqual(-1.0, advantages[1], places=6)
        self.assertAlmostEqual(0.0, sum(advantages), places=7)
        with self.assertRaisesRegex(
            benchmark.GrpoBenchmarkContractError, "closed reward contract"
        ):
            case.reward_for_token(42)

    def test_padding_preserves_prompt_and_one_ranked_token(self) -> None:
        case = benchmark.load_case(benchmark.DEFAULT_CASE_PATH)
        rows = benchmark.padded_group(
            case.prompt_token_ids,
            case.expected_initial_token_ids,
            case.sequence_length,
        )
        self.assertEqual(case.group_size, len(rows))
        for row, token_id in zip(rows, case.expected_initial_token_ids):
            self.assertEqual(case.sequence_length, len(row))
            self.assertEqual(
                list(case.prompt_token_ids), row[: len(case.prompt_token_ids)]
            )
            self.assertEqual(token_id, row[len(case.prompt_token_ids)])
            self.assertTrue(all(value == 0 for value in row[len(case.prompt_token_ids) + 1 :]))

    def test_package_versions_are_exact(self) -> None:
        expected = {"mlx": "1", "mlx-lm": "2", "numpy": "3"}
        self.assertEqual(
            expected,
            benchmark.require_exact_package_versions(expected, expected),
        )
        with self.assertRaisesRegex(
            benchmark.GrpoBenchmarkContractError, "versions drifted"
        ):
            benchmark.require_exact_package_versions(
                {**expected, "numpy": "4"}, expected
            )

    def test_case_rejects_unknown_fields(self) -> None:
        payload = self.load_payload()
        payload["typo"] = True
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                benchmark.GrpoBenchmarkContractError, "fields drifted"
            ):
                benchmark.load_case(self.write_case(payload, Path(temp)))

    def test_case_rejects_protocol_drift(self) -> None:
        payload = self.load_payload()
        payload["protocol"]["measured"] = 19
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                benchmark.GrpoBenchmarkContractError, "fixed"
            ):
                benchmark.load_case(self.write_case(payload, Path(temp)))

    def test_case_rejects_reward_order_drift(self) -> None:
        payload = self.load_payload()
        payload["reward"]["token_contract"] = list(
            reversed(payload["reward"]["token_contract"])
        )
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                benchmark.GrpoBenchmarkContractError, "reward token order"
            ):
                benchmark.load_case(self.write_case(payload, Path(temp)))

    def test_case_rejects_degenerate_rewards(self) -> None:
        payload = self.load_payload()
        payload["reward"]["token_contract"][1]["reward"] = 1.0
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                benchmark.GrpoBenchmarkContractError, "produce an advantage"
            ):
                benchmark.load_case(self.write_case(payload, Path(temp)))

    def test_case_rejects_overlength_prompt(self) -> None:
        payload = self.load_payload()
        payload["sequence_length"] = len(payload["prompt_token_ids"])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                benchmark.GrpoBenchmarkContractError, "exceeds sequence_length"
            ):
                benchmark.load_case(self.write_case(payload, Path(temp)))

    def test_exclusive_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            benchmark.write_json_exclusive(path, {"run": 1})
            self.assertEqual({"run": 1}, json.loads(path.read_text(encoding="utf-8")))
            with self.assertRaisesRegex(
                benchmark.GrpoBenchmarkContractError, "already exists"
            ):
                benchmark.write_json_exclusive(path, {"run": 2})

    def test_source_revision_requires_clean_pinned_checkout(self) -> None:
        expected = "a" * 40
        with mock.patch.object(
            benchmark.locked,
            "verify_source_checkout",
            return_value={"path": "/tmp/mlx", "revision": expected},
        ) as verify:
            self.assertEqual(
                expected,
                benchmark.require_source_revision(
                    Path("/tmp/mlx"), expected, "MLX"
                ),
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
                benchmark.GrpoBenchmarkContractError,
                "clean MLX source revision",
            ):
                benchmark.require_source_revision(
                    Path("/tmp/mlx"), expected, "MLX"
                )

    def test_import_surface_keeps_mlx_lazy(self) -> None:
        source = Path(benchmark.__file__).read_text(encoding="utf-8")
        prefix = source.split("def run(args", 1)[0]
        self.assertNotIn("import mlx.core", prefix)
        self.assertNotIn("import mlx.nn", prefix)


if __name__ == "__main__":
    unittest.main()
