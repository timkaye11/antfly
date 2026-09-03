from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import qualify_gemma4_preference_quality_campaign as campaign


class PreferenceQualityCampaignTest(unittest.TestCase):
    def test_seed_contract_requires_three_unique_u64_values(self) -> None:
        self.assertEqual(campaign._parse_seeds("17,42,991"), [17, 42, 991])
        with self.assertRaisesRegex(Exception, "three unique"):
            campaign._parse_seeds("17,17,42")
        with self.assertRaisesRegex(Exception, "fit u64"):
            campaign._parse_seeds("17,42,-1")

    def test_seeded_row_order_is_stable_distinct_and_content_preserving(self) -> None:
        rows = [
            json.dumps({"row": index}, sort_keys=True).encode()
            for index in range(8)
        ]
        first = campaign._permuted_rows(rows, 17)
        self.assertEqual(first, campaign._permuted_rows(rows, 17))
        self.assertNotEqual(first, campaign._permuted_rows(rows, 42))
        self.assertCountEqual(first.splitlines(), rows)
        self.assertEqual(
            campaign._row_multiset_sha256(first.splitlines()),
            campaign._row_multiset_sha256(rows),
        )

    def test_selected_row_set_must_support_every_requested_order_seed(self) -> None:
        campaign._require_permutation_capacity(3, 3)
        with self.assertRaisesRegex(campaign.ContractError, "enough distinct"):
            campaign._require_permutation_capacity(2, 3)

    def test_quality_metric_ranges_fail_closed(self) -> None:
        self.assertEqual(campaign._probability(0.25, "rate"), 0.25)
        self.assertEqual(campaign._nonnegative(0.0, "loss"), 0.0)
        with self.assertRaisesRegex(campaign.ContractError, r"\[0, 1\]"):
            campaign._probability(1.01, "rate")
        with self.assertRaisesRegex(campaign.ContractError, "non-negative"):
            campaign._nonnegative(-0.01, "loss")

    def test_grpo_paired_prompt_gate_uses_exact_clustered_sign_test(self) -> None:
        self.assertAlmostEqual(
            campaign._one_sided_exact_sign_test_p_value(34, 6),
            4.182292286714073e-06,
        )
        baseline = [0.0, 0.0] * 8
        evaluation = [1.0, 0.0] * 6 + [0.0, 0.0] * 2
        result = campaign._paired_prompt_reward_test(
            baseline,
            evaluation,
            group_size=2,
            maximum_p_value=0.05,
        )
        self.assertEqual(result["wins"], 6)
        self.assertEqual(result["losses"], 0)
        self.assertEqual(result["ties"], 2)
        self.assertEqual(result["one_sided_exact_p_value"], 0.015625)
        self.assertTrue(result["directional_passed"])
        self.assertTrue(result["significance_passed"])
        self.assertTrue(result["passed"])

        with self.assertRaisesRegex(campaign.ContractError, "production maximum"):
            campaign._paired_prompt_reward_test(
                baseline,
                evaluation,
                group_size=2,
                maximum_p_value=0.051,
            )
        with self.assertRaisesRegex(campaign.ContractError, "must be positive"):
            campaign._paired_prompt_reward_test(
                baseline,
                evaluation,
                group_size=2,
                maximum_p_value=0.0,
            )

    def test_grpo_paired_prompt_gate_rejects_nondirectional_changes(self) -> None:
        baseline = [0.0, 0.0] * 3 + [1.0, 0.0] * 3
        evaluation = [1.0, 0.0] * 3 + [0.0, 0.0] * 3
        result = campaign._paired_prompt_reward_test(
            baseline,
            evaluation,
            group_size=2,
            maximum_p_value=0.05,
        )
        self.assertEqual((result["wins"], result["losses"]), (3, 3))
        self.assertFalse(result["directional_passed"])
        self.assertFalse(result["passed"])

    def test_grpo_per_seed_direction_does_not_require_independent_significance(self) -> None:
        result = campaign._paired_prompt_reward_test(
            [0.0] * 36,
            [1.0] * 23 + [-1.0] * 13,
            group_size=1,
            maximum_p_value=0.05,
        )
        self.assertEqual((result["wins"], result["losses"]), (23, 13))
        self.assertAlmostEqual(
            result["one_sided_exact_p_value"], 0.06624908198136836
        )
        self.assertTrue(result["directional_passed"])
        self.assertFalse(result["significance_passed"])
        self.assertFalse(result["passed"])

    def test_grpo_multi_seed_gate_counts_each_prompt_once(self) -> None:
        # Pooling these 15 prompt-seed signs would report p < 0.05, but there
        # are only five independent evaluation prompts. The prompt-averaged
        # test correctly retains five observations and does not pass.
        deltas = [1.0, 1.0, 1.0, 1.0, -1.0]
        result = campaign._multi_seed_paired_prompt_reward_test(
            {17: deltas, 42: deltas, 991: deltas},
            group_size=2,
            maximum_p_value=0.05,
        )
        self.assertLessEqual(
            campaign._one_sided_exact_sign_test_p_value(12, 3), 0.05
        )
        self.assertEqual(result["groups"], 5)
        self.assertEqual((result["wins"], result["losses"]), (4, 1))
        self.assertEqual(result["one_sided_exact_p_value"], 0.1875)
        self.assertTrue(result["all_seeds_directional"])
        self.assertFalse(result["passed"])

    def test_grpo_multi_seed_gate_requires_directional_seeds_and_significance(self) -> None:
        passing = campaign._multi_seed_paired_prompt_reward_test(
            {
                17: [1.0] * 6 + [0.0] * 2,
                42: [2.0] * 6 + [0.0] * 2,
                991: [3.0] * 6 + [0.0] * 2,
            },
            group_size=2,
            maximum_p_value=0.05,
        )
        self.assertEqual((passing["wins"], passing["losses"]), (6, 0))
        self.assertEqual(passing["one_sided_exact_p_value"], 0.015625)
        self.assertTrue(passing["all_seeds_directional"])
        self.assertTrue(passing["passed"])

        nondirectional = campaign._multi_seed_paired_prompt_reward_test(
            {
                17: [1.0] * 6 + [0.0] * 2,
                42: [1.0] * 6 + [0.0] * 2,
                991: [-1.0] * 5 + [1.0] * 3,
            },
            group_size=2,
            maximum_p_value=0.05,
        )
        self.assertFalse(nondirectional["all_seeds_directional"])
        self.assertFalse(nondirectional["passed"])

        with self.assertRaisesRegex(campaign.ContractError, "at least three"):
            campaign._multi_seed_paired_prompt_reward_test(
                {17: [1.0], 42: [1.0]},
                group_size=2,
                maximum_p_value=0.05,
            )
        with self.assertRaisesRegex(campaign.ContractError, "prompt counts"):
            campaign._multi_seed_paired_prompt_reward_test(
                {17: [1.0], 42: [1.0], 991: [1.0, 1.0]},
                group_size=2,
                maximum_p_value=0.05,
            )

    def test_grpo_reward_trace_loader_binds_call_and_prompt_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            rows = [
                {
                    "schema_version": "antfly_inference_grpo_reward_trace/v1",
                    "phase": "evaluation",
                    "call_index": index,
                    "prompt_index": index // 2,
                    "aggregate_reward": float(index % 2),
                }
                for index in range(4)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in reversed(rows)),
                encoding="utf-8",
            )
            self.assertEqual(
                campaign._load_grpo_evaluation_rewards(
                    path, groups=2, group_size=2, where="test trace"
                ),
                [0.0, 1.0, 0.0, 1.0],
            )
            rows[0]["prompt_index"] = 1
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(campaign.ContractError, "group contract"):
                campaign._load_grpo_evaluation_rewards(
                    path, groups=2, group_size=2, where="test trace"
                )

    def test_grpo_multi_seed_evidence_reloads_digest_bound_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            baseline_paths = []

            def write_trace(path: Path, rewards: list[float]) -> None:
                path.write_text(
                    "".join(
                        json.dumps(
                            {
                                "schema_version": "antfly_inference_grpo_reward_trace/v1",
                                "phase": "evaluation",
                                "call_index": index,
                                "prompt_index": index,
                                "aggregate_reward": reward,
                            }
                        )
                        + "\n"
                        for index, reward in enumerate(rewards)
                    ),
                    encoding="utf-8",
                )

            for seed in (17, 42, 991):
                baseline_path = root / f"baseline-{seed}.jsonl"
                evaluation_path = root / f"evaluation-{seed}.jsonl"
                write_trace(baseline_path, [0.0] * 8)
                write_trace(evaluation_path, [1.0] * 6 + [0.0] * 2)
                baseline_paths.append(baseline_path)
                runs.append(
                    {
                        "seed": seed,
                        "quality": {
                            "paired_evaluation": {
                                "groups": 8,
                                "completions": 8,
                                "wins": 6,
                                "losses": 0,
                                "ties": 2,
                                "per_seed_gate_passed": True,
                                "baseline_trace_path": str(baseline_path),
                                "baseline_trace_sha256": campaign.resume_qualifier._sha256(
                                    baseline_path
                                ),
                                "evaluation_trace_path": str(evaluation_path),
                                "evaluation_trace_sha256": campaign.resume_qualifier._sha256(
                                    evaluation_path
                                ),
                            }
                        },
                    }
                )

            evidence = campaign._multi_seed_grpo_evaluation_evidence(
                runs,
                expected_groups=8,
                maximum_p_value=0.05,
            )
            self.assertTrue(evidence["passed"])
            self.assertEqual(evidence["one_sided_exact_p_value"], 0.015625)
            self.assertEqual(len(evidence["trace_pairs"]), 3)

            baseline_paths[0].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(campaign.ContractError, "changed"):
                campaign._multi_seed_grpo_evaluation_evidence(
                    runs,
                    expected_groups=8,
                    maximum_p_value=0.05,
                )

    def test_grpo_quality_defaults_use_paired_gain_and_one_group_noninferiority(self) -> None:
        args = campaign.parse_args(
            [
                "--binary",
                "/antfly",
                "--recipe",
                "/recipe.json",
                "--output-dir",
                "/out",
            ]
        )
        self.assertEqual(args.min_grpo_eval_mean_reward_improvement, 1e-6)
        self.assertEqual(args.min_grpo_eval_top_rank_mean_reward_improvement, 0.0)
        self.assertIsNone(
            args.min_grpo_eval_positive_reward_group_rate_improvement
        )
        self.assertIsNone(args.max_grpo_eval_positive_reward_group_regressions)
        self.assertEqual(
            args.max_grpo_paired_prompt_reward_sign_test_p_value, 0.05
        )

    def test_grpo_positive_group_noninferiority_is_count_bounded(self) -> None:
        self.assertEqual(
            campaign._grpo_positive_group_requirement(256, None, None),
            (-1 / 256, 1),
        )
        self.assertEqual(
            campaign._grpo_positive_group_requirement(256, 0.0, None),
            (0.0, 0),
        )
        with self.assertRaisesRegex(campaign.ContractError, "choose either"):
            campaign._grpo_positive_group_requirement(256, 0.0, 1)
        with self.assertRaisesRegex(campaign.ContractError, "production maximum"):
            campaign._grpo_positive_group_requirement(256, None, 2)

        one_regression = campaign._grpo_positive_group_noninferiority(
            216 / 256,
            215 / 256,
            groups=256,
            maximum_regressions=1,
        )
        self.assertTrue(one_regression["passed"])
        self.assertEqual(one_regression["change"], -1)
        two_regressions = campaign._grpo_positive_group_noninferiority(
            216 / 256,
            214 / 256,
            groups=256,
            maximum_regressions=1,
        )
        self.assertFalse(two_regressions["passed"])
        as_f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
        non_power_of_two_groups = campaign._grpo_positive_group_noninferiority(
            as_f32(214 / 254),
            as_f32(213 / 254),
            groups=254,
            maximum_regressions=1,
        )
        self.assertTrue(non_power_of_two_groups["passed"])
        self.assertEqual(non_power_of_two_groups["change"], -1)
        with self.assertRaisesRegex(campaign.ContractError, "group-count rate"):
            campaign._rate_count(0.5, 3, "test rate")

    def test_long_horizon_is_bound_to_training_units_not_prompt_repetition(self) -> None:
        self.assertEqual(campaign._long_horizon_units("dpo", 5, 8), (40, 40))
        self.assertEqual(campaign._long_horizon_units("grpo", 512, 1), (512, 512))
        self.assertEqual(campaign._long_horizon_units("grpo", 128, 4), (512, 512))
        with self.assertRaisesRegex(campaign.ContractError, "at least 512"):
            campaign._long_horizon_units("grpo", 128, 3)
        with self.assertRaisesRegex(campaign.ContractError, "must be positive"):
            campaign._long_horizon_units("dpo", 40, 0)

    def test_grpo_training_coverage_rejects_nominal_horizon_without_signal(self) -> None:
        sparse = {
            "optimizer_groups": 7,
            "zero_reward_std_groups": 249,
            "all_truncated_groups": 0,
            "kl_rejected_groups": 0,
            "frac_reward_zero_std": 249 / 256,
            "frac_kl_rejected": 0.0,
        }
        with self.assertRaisesRegex(
            campaign.ContractError, "optimizer-group rate is below"
        ):
            campaign._grpo_training_coverage(sparse, 256, 0.25)

        admitted = dict(sparse)
        admitted.update(
            {
                "optimizer_groups": 96,
                "zero_reward_std_groups": 160,
                "frac_reward_zero_std": 160 / 256,
            }
        )
        coverage = campaign._grpo_training_coverage(admitted, 256, 0.25)
        self.assertEqual(coverage["optimizer_groups"], 96)
        self.assertEqual(coverage["optimizer_group_rate"], 0.375)

    def test_grpo_training_coverage_requires_exact_horizon_accounting(self) -> None:
        malformed = {
            "optimizer_groups": 64,
            "zero_reward_std_groups": 191,
            "all_truncated_groups": 0,
            "kl_rejected_groups": 0,
            "frac_reward_zero_std": 191 / 256,
            "frac_kl_rejected": 0.0,
        }
        with self.assertRaisesRegex(campaign.ContractError, "cover the horizon"):
            campaign._grpo_training_coverage(malformed, 256, 0.25)

    def test_grpo_production_signal_and_kl_skip_bounds_cannot_be_relaxed(self) -> None:
        admitted = {
            "optimizer_groups": 253,
            "zero_reward_std_groups": 0,
            "all_truncated_groups": 0,
            "kl_rejected_groups": 3,
            "frac_reward_zero_std": 0.0,
            "frac_kl_rejected": 3 / 256,
        }
        with self.assertRaisesRegex(campaign.ContractError, "production minimum"):
            campaign._grpo_training_coverage(admitted, 256, 0.10)
        with self.assertRaisesRegex(campaign.ContractError, "production maximum"):
            campaign._grpo_training_coverage(admitted, 256, 0.25, 0.02)
        with self.assertRaisesRegex(campaign.ContractError, "campaign ceiling"):
            campaign._grpo_training_coverage(admitted, 256, 0.25, 0.01)

    def test_grpo_kl_admissions_match_realized_optimizer_groups(self) -> None:
        campaign._require_grpo_kl_admissions({"admitted_groups": 96}, 96)
        with self.assertRaisesRegex(campaign.ContractError, "optimizer groups"):
            campaign._require_grpo_kl_admissions({"admitted_groups": 256}, 96)

    def test_bounded_improvement_requirements_preserve_saturated_metrics(self) -> None:
        self.assertEqual(
            (0.0, True),
            campaign._bounded_increase_requirement(1.0, 1e-6, 1.0),
        )
        self.assertEqual(
            (1e-6, False),
            campaign._bounded_increase_requirement(0.75, 1e-6, 1.0),
        )
        self.assertEqual(
            (0.0, True),
            campaign._bounded_decrease_requirement(0.0, 1e-6, 0.0),
        )

    def test_variant_binds_horizon_order_and_direct_gguf_admission(self) -> None:
        base = {
            "recipe": "grpo",
            "model": {"path": "/model.gguf", "family": "gemma4"},
            "adapter": {"path": "/template-adapter"},
            "dataset": {"path": "/train.jsonl", "eval_path": "/eval.jsonl"},
            "optimizer": {"epochs": 2, "learning_rate": 1e-7},
            "checkpoint": {"every_epochs": 1},
            "artifacts": {"root": "/old"},
        }
        variant = campaign._variant(
            base,
            "grpo",
            991,
            Path("/adapter-991"),
            Path("/seeded.jsonl"),
            Path("/run"),
            8,
            5e-8,
            True,
        )
        self.assertEqual(variant["dataset"]["path"], "/seeded.jsonl")
        self.assertEqual(variant["optimizer"]["epochs"], 8)
        self.assertEqual(variant["optimizer"]["seed"], 991)
        self.assertEqual(variant["optimizer"]["learning_rate"], 5e-8)
        self.assertEqual(variant["adapter"]["path"], "/adapter-991")
        self.assertEqual(variant["adapter"]["initialization_seed"], 991)
        self.assertTrue(variant["model"]["allow_direct_gguf_training"])
        self.assertNotIn("checkpoint", variant)
        self.assertEqual(variant["artifacts"]["root"], "/run")
        self.assertIn("checkpoint", base)

    def test_variant_rebinds_both_training_dataset_aliases(self) -> None:
        base = {
            "recipe": "dpo",
            "model": {"path": "/model", "family": "gemma4"},
            "adapter": {"path": "/template-adapter"},
            "dataset": {
                "path": "/train.jsonl",
                "train_path": "/train.jsonl",
                "eval_path": "/eval.jsonl",
            },
            "optimizer": {"epochs": 2},
        }
        variant = campaign._variant(
            base,
            "dpo",
            42,
            Path("/adapter-42"),
            Path("/seeded.jsonl"),
            Path("/run"),
            8,
            None,
            False,
        )
        self.assertEqual(variant["dataset"]["path"], "/seeded.jsonl")
        self.assertEqual(variant["dataset"]["train_path"], "/seeded.jsonl")

    def test_compiled_sampling_cli_forces_incremental_runtime_out_of_variants(self) -> None:
        args = campaign.parse_args(
            [
                "--binary",
                "/antfly",
                "--recipe",
                "/recipe.json",
                "--output-dir",
                "/out",
                "--compiled-sampling",
            ]
        )
        self.assertTrue(args.compiled_sampling)
        base = {
            "recipe": "grpo",
            "model": {"path": "/model", "family": "gemma4"},
            "adapter": {"path": "/template-adapter"},
            "dataset": {"path": "/train.jsonl", "eval_path": "/eval.jsonl"},
            "optimizer": {"epochs": 2},
            "runtime": {
                "grpo_incremental_kv": True,
                "grpo_incremental_kv_batch_active": True,
                "grpo_incremental_kv_clone_prompt_tail": True,
                "grpo_incremental_kv_shadow_exact": True,
            },
        }
        campaign.resume_qualifier._apply_compiled_sampling_recipe_contract(
            base, "grpo", True
        )
        variant = campaign._variant(
            base,
            "grpo",
            17,
            Path("/adapter-17"),
            Path("/seeded.jsonl"),
            Path("/run"),
            8,
            None,
            False,
        )
        self.assertFalse(variant["runtime"]["grpo_incremental_kv"])
        for field in (
            "grpo_incremental_kv_batch_active",
            "grpo_incremental_kv_clone_prompt_tail",
            "grpo_incremental_kv_shadow_exact",
        ):
            self.assertNotIn(field, variant["runtime"])

    def test_template_adapter_bootstrap_spec_is_strict_and_peft_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            config = {
                "r": 8,
                "lora_alpha": 16.0,
                "target_modules": ["model.layers.0.self_attn.q_proj"],
                "init_lora_weights": True,
                "use_dora": False,
            }
            (adapter / "adapter_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            self.assertEqual(
                campaign._adapter_bootstrap_spec(adapter),
                {
                    "rank": 8,
                    "alpha": 16.0,
                    "target_modules": ["model.layers.0.self_attn.q_proj"],
                },
            )
            config["init_lora_weights"] = "eva"
            (adapter / "adapter_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            with self.assertRaisesRegex(campaign.ContractError, "standard LoRA"):
                campaign._adapter_bootstrap_spec(adapter)

    def test_seed_bootstrap_accepts_v3_manifest_through_shared_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter-seed-17"
            target_modules = ["model.layers.0.self_attn.q_proj"]
            payload = b"seeded-adapter"

            def fake_run(command, _env, _log_root, _timeout):
                self.assertEqual(command[command.index("--initialization-seed") + 1], "17")
                adapter.mkdir()
                (adapter / "adapter_model.safetensors").write_bytes(payload)
                (adapter / "adapter_config.json").write_text(
                    json.dumps(
                        {
                            "r": 8,
                            "lora_alpha": 16.0,
                            "target_modules": target_modules,
                        }
                    ),
                    encoding="utf-8",
                )
                (adapter / "antfly_finetune_manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "antfly_gemma4_finetune/v3",
                            "status": "complete",
                            "adapter_checkpoint_sha256": hashlib.sha256(
                                payload
                            ).hexdigest(),
                            "adapter_checkpoint_size_bytes": len(payload),
                            "initialization_seed": 17,
                        }
                    ),
                    encoding="utf-8",
                )
                return {"command": command, "returncode": 0}

            with mock.patch.object(campaign, "_run", side_effect=fake_run):
                evidence = campaign._bootstrap_seed_adapter(
                    Path("/fake/antfly"),
                    Path("/model"),
                    {"rank": 8, "alpha": 16.0, "target_modules": target_modules},
                    17,
                    adapter,
                    {},
                    root / "bootstrap-seed-17",
                    1.0,
                )

            self.assertEqual(evidence["initialization_seed"], 17)
            self.assertEqual(
                evidence["adapter_model_sha256"],
                "sha256:" + hashlib.sha256(payload).hexdigest(),
            )

    def test_dataset_path_uses_runtime_precedence_and_rejects_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            alternate = root / "alternate.jsonl"
            for path in (train, evaluation, alternate):
                path.write_text("{}\n", encoding="utf-8")
            recipe = {
                "dataset": {
                    "path": str(train),
                    "train_path": str(train),
                    "eval_path": str(evaluation),
                },
                "eval": {"path": str(evaluation)},
            }
            self.assertEqual(campaign._dataset_path(recipe, "train"), train.resolve())
            self.assertEqual(campaign._dataset_path(recipe, "eval"), evaluation.resolve())
            recipe["eval"]["path"] = str(alternate)
            with self.assertRaisesRegex(campaign.ContractError, "conflicting eval"):
                campaign._dataset_path(recipe, "eval")

    def test_failed_run_evidence_surfaces_bounded_metrics_and_artifact_digests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "seed-42"
            run_root.mkdir()
            report = {
                "schema_version": "antfly_inference_finetune_grpo_report/v8",
                "execution_mode": "train",
                "groups": 512,
                "optimizer_groups": 200,
                "baseline_relative": {
                    "top_rank_mean_reward_improvement": 0.0,
                    "passed": False,
                },
                "trained_adapter_dir": None,
                "unbounded_detail": "must-not-be-copied",
            }
            report_path = run_root / "grpo_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            evaluation_path = run_root / "grpo-evaluation.json"
            evaluation_path.write_text(
                json.dumps(
                    {
                        "schema_version": "antfly_inference_finetune_grpo_evaluation/v4",
                        "status": "passed",
                        "groups": 64,
                        "mean_reward": 0.5,
                        "top_rank_mean_reward": 0.25,
                        "positive_reward_group_rate": 0.75,
                    }
                ),
                encoding="utf-8",
            )
            trace_path = run_root / "grpo_kl_control_trace.jsonl"
            trace_path.write_text("{}\n", encoding="utf-8")
            baseline_reward_path = (
                run_root / "grpo_baseline_evaluation_reward_trace.jsonl"
            )
            baseline_reward_path.write_text("{\"reward\":1}\n", encoding="utf-8")
            stderr_path = run_root.with_suffix(".stderr.log")
            stderr_path.write_text("quality gate failed\n", encoding="utf-8")

            evidence = campaign._failed_run_evidence("grpo", 42, run_root)

            self.assertEqual(evidence["seed"], 42)
            artifacts = evidence["artifacts"]
            self.assertEqual(
                artifacts["task_report"]["summary"]["baseline_relative"][
                    "passed"
                ],
                False,
            )
            self.assertIsNone(
                artifacts["task_report"]["summary"]["trained_adapter_dir"]
            )
            self.assertNotIn(
                "unbounded_detail", artifacts["task_report"]["summary"]
            )
            self.assertEqual(
                artifacts["evaluation_report"]["summary"]["mean_reward"], 0.5
            )
            self.assertEqual(
                artifacts["kl_control_trace"]["sha256"],
                "sha256:" + hashlib.sha256(b"{}\n").hexdigest(),
            )
            self.assertEqual(
                artifacts["baseline_evaluation_reward_trace"]["sha256"],
                "sha256:"
                + hashlib.sha256(b'{"reward":1}\n').hexdigest(),
            )
            self.assertEqual(
                artifacts["stderr_log"]["sha256"],
                "sha256:"
                + hashlib.sha256(b"quality gate failed\n").hexdigest(),
            )
            self.assertNotIn("baseline_evaluation_report", artifacts)

    def test_failure_summary_rejects_oversized_json_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "grpo_report.json"
            report_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(campaign, "MAX_FAILURE_SUMMARY_BYTES", 1):
                evidence = campaign._available_failure_artifact(
                    report_path, ("schema_version",)
                )

            self.assertIsNotNone(evidence)
            assert evidence is not None
            self.assertEqual(evidence["size_bytes"], 2)
            self.assertIn("exceeds failure-summary input limit", evidence["summary_error"])
            self.assertNotIn("summary", evidence)

    def test_metric_summary_reports_worst_case_and_population_spread(self) -> None:
        runs = [
            {"quality": {"metrics": {"eval_loss": value}}}
            for value in (1.0, 2.0, 3.0)
        ]
        summary = campaign._metric_summary(runs)["eval_loss"]
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["minimum"], 1.0)
        self.assertEqual(summary["maximum"], 3.0)
        self.assertGreater(summary["population_stddev"], 0.0)


if __name__ == "__main__":
    unittest.main()
