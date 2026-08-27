#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_gemma4_grpo_boolq_mlx_parity as parity  # noqa: E402


class BoolQGrpoMlxParityTests(unittest.TestCase):
    def test_accepts_stochastic_report_schema_v7(self) -> None:
        self.assertIn(
            "antfly_inference_finetune_grpo_report/v7",
            parity.GRPO_REPORT_SCHEMA_VERSIONS,
        )
        self.assertIn(
            "antfly_inference_finetune_grpo_evaluation/v4",
            parity.GRPO_EVAL_SCHEMA_VERSIONS,
        )

    def test_native_rollout_fails_closed_for_stochastic_antfly_sampling(self) -> None:
        parity.require_native_rollout_sampler_compatibility(
            {"sampling_mode": "shared-prompt-ranked-sparse-row"}
        )
        with self.assertRaisesRegex(
            parity.BoolQParityContractError, "retired deterministic ranked sampler"
        ):
            parity.require_native_rollout_sampler_compatibility(
                {"sampling_mode": "shared-prompt-seeded-categorical-sparse-row"}
            )

    def test_exact_match_ci_is_trimmed_but_not_substring_match(self) -> None:
        self.assertEqual(1.0, parity.exact_match_ci(" Yes ", "yes"))
        self.assertEqual(1.0, parity.exact_match_ci("NO", "no"))
        self.assertEqual(0.0, parity.exact_match_ci("yesterday", "yes"))
        self.assertEqual(0.0, parity.exact_match_ci("not", "no"))

    def test_normalized_advantages_match_unbiased_variance_contract(self) -> None:
        values = parity.normalized_advantages([1.0, 1.0, 0.0, 0.0], 1.0e-4)
        self.assertAlmostEqual(0.865875, values[0], places=6)
        self.assertAlmostEqual(0.865875, values[1], places=6)
        self.assertAlmostEqual(-0.865875, values[2], places=6)
        self.assertAlmostEqual(-0.865875, values[3], places=6)
        self.assertAlmostEqual(0.0, sum(values), places=7)

    def test_candidate_overlap_distinguishes_set_order_and_top1(self) -> None:
        same_set = parity.candidate_overlap([3, 2, 1], [1, 2, 3])
        self.assertEqual(3, same_set["overlap"])
        self.assertTrue(same_set["exact_set"])
        self.assertFalse(same_set["exact_order"])
        self.assertFalse(same_set["top1_match"])
        partial = parity.candidate_overlap([1, 4, 5], [1, 2, 3])
        self.assertEqual(1, partial["overlap"])
        self.assertAlmostEqual(1.0 / 3.0, partial["recall"])
        self.assertTrue(partial["top1_match"])

    def _trace_row(
        self, *, call_index: int, prompt_index: int, token_id: int, reward: float
    ) -> dict:
        return {
            "schema_version": parity.TRACE_SCHEMA_VERSION,
            "phase": "train",
            "call_index": call_index,
            "prompt_index": prompt_index,
            "completion_tokens": [token_id],
            "aggregate_reward": reward,
        }

    def test_trace_loader_requires_contiguous_call_and_prompt_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            rows = [
                self._trace_row(
                    call_index=index,
                    prompt_index=index // 2,
                    token_id=100 + index,
                    reward=float(index % 2),
                )
                for index in range(4)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            groups = parity.load_trace(
                path, phase="train", expected_groups=2, group_size=2
            )
            self.assertEqual((100, 101), groups[0].token_ids)
            self.assertEqual((0.0, 1.0), groups[0].rewards)
            rows[2]["call_index"] = 99
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                parity.BoolQParityContractError, "trace order drifted"
            ):
                parity.load_trace(path, phase="train", expected_groups=2, group_size=2)

    def test_trace_loader_accepts_duplicate_stochastic_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            rows = [
                self._trace_row(
                    call_index=index,
                    prompt_index=0,
                    token_id=7,
                    reward=float(index),
                )
                for index in range(2)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            groups = parity.load_trace(
                path, phase="train", expected_groups=1, group_size=2
            )
            self.assertEqual((7, 7), groups[0].token_ids)

    def _materialization(self, root: Path) -> Path:
        train = root / "train.jsonl"
        evaluation = root / "eval.jsonl"
        train.write_text("train\n", encoding="utf-8")
        evaluation.write_text("eval\n", encoding="utf-8")
        train_ids = [f"{index:064x}" for index in range(1, 9)]
        eval_ids = [f"{index:064x}" for index in range(100, 164)]
        payload = {
            "schema_version": parity.MATERIALIZATION_SCHEMA_VERSION,
            "dataset": {
                "repo_id": "google/boolq",
                "revision": "a" * 40,
                "selection_policy": {
                    "dataset_format": "rendered-text-grpo",
                    "max_seq_len": parity.FIXED_SEQUENCE_LENGTH,
                    "max_completion_tokens": 1,
                    "target_tokens": 1,
                    "rendered_prompt_truncation": "forbidden",
                    "response_channel": "final",
                },
                "train": {
                    "materialized_jsonl_sha256": hashlib.sha256(
                        train.read_bytes()
                    ).hexdigest()
                },
                "evaluation": {
                    "materialized_jsonl_sha256": hashlib.sha256(
                        evaluation.read_bytes()
                    ).hexdigest()
                },
            },
            "train_jsonl": str(train),
            "eval_jsonl": str(evaluation),
            "train_source_ids": train_ids,
            "eval_source_ids": eval_ids,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_materialization_binds_jsonl_digests_and_disjoint_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self._materialization(root)
            manifest = parity.load_materialization(path)
            self.assertEqual("google/boolq", manifest["dataset"]["repo_id"])
            (root / "eval.jsonl").write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(
                parity.BoolQParityContractError, "evaluation JSONL SHA-256 drifted"
            ):
                parity.load_materialization(path)

    def test_exclusive_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            parity.write_json_exclusive(path, {"run": 1})
            with self.assertRaisesRegex(
                parity.BoolQParityContractError, "already exists"
            ):
                parity.write_json_exclusive(path, {"run": 2})

    def test_import_surface_keeps_mlx_lazy(self) -> None:
        source = Path(parity.__file__).read_text(encoding="utf-8")
        prefix = source.split("def run(args", 1)[0]
        self.assertNotIn("import mlx.core", prefix)
        self.assertNotIn("import mlx.nn", prefix)

    def test_batched_antfly_reference_requires_phase_separated_evaluation(self) -> None:
        train_report = {"reference_mode": parity.ANTFLY_BATCHED_REFERENCE_MODE}
        eval_report = {
            "reference_mode": parity.ANTFLY_BATCHED_REFERENCE_MODE,
            "execution_order": parity.ANTFLY_PHASED_EVALUATION_ORDER,
        }
        parity.require_antfly_reference_contract(train_report, eval_report)
        with self.assertRaisesRegex(
            parity.BoolQParityContractError, "did not phase-separate"
        ):
            parity.require_antfly_reference_contract(
                train_report,
                {
                    "reference_mode": parity.ANTFLY_BATCHED_REFERENCE_MODE,
                    "execution_order": "interleaved-policy-reference",
                },
            )

    def test_legacy_antfly_reference_remains_loadable(self) -> None:
        parity.require_antfly_reference_contract(
            {"reference_mode": parity.ANTFLY_LEGACY_REFERENCE_MODE}, {}
        )

    def test_v4_kl_control_trace_is_digest_bound_and_fully_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            trace_path = root / "grpo_kl_control_trace.jsonl"
            rows = [
                {
                    "schema_version": parity.GRPO_KL_TRACE_SCHEMA_VERSION,
                    "group_index": index,
                    "optimizer_steps_before": index,
                    "status": "admitted",
                    "budget_policy": "skip_group",
                    "mean_kl": 0.001,
                    "train_max_kl": 0.1,
                }
                for index in range(parity.FIXED_TRAIN_GROUPS)
            ]

            def write_trace() -> str:
                trace_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                return "sha256:" + hashlib.sha256(trace_path.read_bytes()).hexdigest()

            report = {
                "schema_version": "antfly_inference_finetune_grpo_report/v4",
                "mean_kl": 0.001,
                "kl_control": {
                    "trace_path": str(trace_path),
                    "trace_digest": write_trace(),
                },
            }
            parity.require_v4_kl_control(root, report)

            rows[-1]["status"] = "budget-exceeded"
            report["kl_control"]["trace_digest"] = write_trace()
            with self.assertRaisesRegex(
                parity.BoolQParityContractError, "admission trace drifted"
            ):
                parity.require_v4_kl_control(root, report)

    def test_wheel_runtime_attestation_binds_extracted_native_members(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "runtime"
            core_member = "mlx/core.cpython-312-darwin.so"
            metal_members = {
                "mlx/lib/libjaccl.dylib": b"jaccl",
                "mlx/lib/libmlx.dylib": b"mlx",
                "mlx/lib/mlx.metallib": b"metal",
            }

            def write_wheel(path: Path, package: str, members: dict[str, bytes]) -> None:
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(
                        f"{package.replace('-', '_')}-0.31.2.dist-info/METADATA",
                        f"Name: {package}\nVersion: 0.31.2\n",
                    )
                    for name, value in members.items():
                        archive.writestr(name, value)
                        extracted = runtime / name
                        extracted.parent.mkdir(parents=True, exist_ok=True)
                        extracted.write_bytes(value)

            mlx_wheel = root / "mlx.whl"
            metal_wheel = root / "mlx_metal.whl"
            write_wheel(mlx_wheel, "mlx", {core_member: b"core"})
            write_wheel(metal_wheel, "mlx-metal", metal_members)
            attestation = parity.attest_wheel_runtime(
                runtime_root=runtime,
                wheel_path=mlx_wheel,
                metal_wheel_path=metal_wheel,
                expected_version="0.31.2",
            )
            self.assertEqual("versioned-wheel-archive", attestation["mode"])
            self.assertIn(core_member, attestation["wheels"]["mlx"]["members"])
            (runtime / "mlx/lib/mlx.metallib").write_bytes(b"drift")
            with self.assertRaisesRegex(
                parity.BoolQParityContractError, "differs from archive"
            ):
                parity.attest_wheel_runtime(
                    runtime_root=runtime,
                    wheel_path=mlx_wheel,
                    metal_wheel_path=metal_wheel,
                    expected_version="0.31.2",
                )

    def test_parity_assessment_does_not_hide_update_drift(self) -> None:
        antfly = {
            "mean_reward": 0.36328125,
            "top_rank_mean_reward": 0.703125,
            "kl_loss": 5.8e-7,
        }
        overlap = {"mean_recall": 0.982421875, "top1_match_rate": 0.984375}
        baseline = {
            "mean_reward": 0.36328125,
            "top_rank_mean_reward": 0.6875,
            "candidate_overlap_with_antfly": overlap,
        }
        native = {
            "mean_reward": 0.357421875,
            "top_rank_mean_reward": 0.6875,
            "kl_loss": 5.947e-4,
            "candidate_overlap_with_antfly": overlap,
        }
        trace_training = {
            "candidate_overlap_with_antfly": {"exact_set_rate": 1.0}
        }
        adapter = {
            "delta_cosine_similarity": 0.572,
            "delta_l2_relative_difference": 0.0146,
            "delta_max_abs_difference": 1.59e-6,
        }
        assessment = parity.assess_parity(
            antfly_evaluation=antfly,
            baseline_evaluation=baseline,
            native_evaluation=native,
            trace_training=trace_training,
            trace_adapter=adapter,
            native_quality_passed=True,
        )
        self.assertTrue(assessment["behavioral"]["passed"])
        self.assertFalse(assessment["numerical"]["passed"])
        self.assertEqual(
            "behavioral-parity-with-numerical-drift",
            assessment["classification"],
        )


if __name__ == "__main__":
    unittest.main()
