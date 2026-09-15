#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import ast
import json
import math
import tempfile
import unittest
import copy
from pathlib import Path
from types import SimpleNamespace

import run_gemma4_grpo_boolq_mlx_multitoken as campaign


class Gemma4GrpoBoolQMultiTokenTests(unittest.TestCase):
    def test_trace_adapter_output_is_exclusive_and_preserves_source_names(self) -> None:
        class Value:
            @property
            def T(self):
                return self

            def astype(self, _dtype):
                return self

        class FakeMx:
            float32 = object()

            @staticmethod
            def save_safetensors(path, tensors, metadata):
                Path(path).write_text(json.dumps({
                    "names": sorted(tensors), "metadata": metadata,
                }))

        target = "language_model.model.layers.0.self_attn.q_proj"
        module = "model.layers.0.self_attn.q_proj"
        adapter = SimpleNamespace(tensors={
            (module, "lora_A"): SimpleNamespace(source_name="q.weight.lora_A"),
            (module, "lora_B"): SimpleNamespace(source_name="q.weight.lora_B"),
        })
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "adapter.safetensors"
            result = campaign.write_adapter_exclusive(
                output,
                final_trainables={
                    f"{target}.lora_a": Value(),
                    f"{target}.lora_b": Value(),
                },
                target_names=[target],
                adapter=adapter,
                mx=FakeMx,
            )
            self.assertEqual(2, result["tensor_count"])
            self.assertEqual(campaign.sha256_file(output), result["sha256"])
            self.assertEqual(
                ["q.weight.lora_A", "q.weight.lora_B"],
                json.loads(output.read_text())["names"],
            )
            with self.assertRaises(campaign.MultiTokenParityError):
                campaign.write_adapter_exclusive(
                    output,
                    final_trainables={
                        f"{target}.lora_a": Value(),
                        f"{target}.lora_b": Value(),
                    },
                    target_names=[target],
                    adapter=adapter,
                    mx=FakeMx,
                )

    def test_predictor_capture_keeps_initial_row_across_completion_steps(self) -> None:
        tree = ast.parse(Path(campaign.__file__).read_text(encoding="utf-8"))
        rollout = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "rollout_group")
        code = compile(ast.fix_missing_locations(ast.Module(body=[rollout], type_ignores=[])),
                       str(campaign.__file__), "exec")
        calls = []

        class Tensor:
            def astype(self, dtype):
                return self

            def __getitem__(self, indices):
                return self

            def tolist(self):
                return [float(len(calls) + i) for i in range(4)]

        def forward(model, tokens):
            calls.append(tokens)
            return Tensor()

        env = {"BoolQRow": campaign.BoolQRow, "Any": object, "Sequence": list,
               "sampling_contract": (42, campaign.SamplingPolicy(top_k=1)),
               "spec": SimpleNamespace(profile=SimpleNamespace(sequence_length=16),
                                       group_size=2, max_completion_tokens=2),
               "model": object(), "forward": forward, "eos_token_id": 0,
               "categorical_rollout_group": campaign.categorical_rollout_group,
               "mx": SimpleNamespace(array=lambda value, **kw: value, int32="i32", float32="f32",
                                     eval=lambda *v: None, synchronize=lambda: None)}
        exec(code, env)
        row = campaign.BoolQRow("prompt", "yes", (1, 2), "train", 14, "source")
        captured = {}
        sequences, _ = env["rollout_group"](row, 14, initial_prediction=captured)
        self.assertEqual([[3, 3], [3, 3]], sequences)
        self.assertGreater(len(calls), 1)
        self.assertEqual([1., 2., 3., 4.], captured["logits"])
        self.assertEqual([1, 2], captured["prompt_token_ids"])
        self.assertEqual(1, captured["predictor_position"])
        self.assertEqual(16, captured["physical_sequence_length"])

    def test_predictor_capture_requires_diagnostic_mode_before_loading(self) -> None:
        args = SimpleNamespace(
            execution_lane="both", categorical_diagnostic=False,
            model_key="gemma-4-E4B-it", train_groups=2, eval_groups=2,
            group_size=4, max_completion_tokens=4, recipe_profile="qv-multitoken",
            activation_mode="stock-bf16", capture_initial_training_logits=True,
        )
        with self.assertRaisesRegex(campaign.MultiTokenParityError, "predictor capture"):
            campaign.run(args)

    def test_individual_lanes_cannot_issue_nondiagnostic_results(self) -> None:
        self.assertEqual(("trace_replay", "native_rollout"),
                         campaign.execution_lanes("both", categorical=False))
        for selection, expected in (("trace-replay", "trace_replay"),
                                    ("native-rollout", "native_rollout")):
            self.assertEqual((expected,), campaign.execution_lanes(selection, categorical=True))
            with self.assertRaises(campaign.MultiTokenParityError):
                campaign.execution_lanes(selection, categorical=False)
        with self.assertRaises(campaign.MultiTokenParityError):
            campaign.execution_lanes("typo", categorical=True)

    def test_prefix_and_skipped_evaluation_are_diagnostic_only(self) -> None:
        spec = campaign.CampaignSpec(
            "gemma-4-E4B-it", 1960, 256, 16, 1, "all-linear-single-token"
        )
        self.assertEqual(
            (260, True),
            campaign.diagnostic_execution_shape(
                spec,
                train_prefix_groups=260,
                skip_evaluation=True,
                categorical=True,
            ),
        )
        self.assertEqual(
            (1960, False),
            campaign.diagnostic_execution_shape(
                spec,
                train_prefix_groups=None,
                skip_evaluation=False,
                categorical=False,
            ),
        )
        for prefix in (True, 1, 1961):
            with self.subTest(prefix=prefix), self.assertRaises(
                campaign.MultiTokenParityError
            ):
                campaign.diagnostic_execution_shape(
                    spec,
                    train_prefix_groups=prefix,
                    skip_evaluation=False,
                    categorical=True,
                )
        with self.assertRaisesRegex(campaign.MultiTokenParityError, "diagnostic-only"):
            campaign.diagnostic_execution_shape(
                spec,
                train_prefix_groups=260,
                skip_evaluation=False,
                categorical=False,
            )
        with self.assertRaisesRegex(campaign.MultiTokenParityError, "diagnostic-only"):
            campaign.diagnostic_execution_shape(
                spec,
                train_prefix_groups=None,
                skip_evaluation=True,
                categorical=False,
            )

    def test_unmeasured_lane_is_null_in_result(self) -> None:
        tree = ast.parse(Path(campaign.__file__).read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        result = next(node.value for node in run.body if isinstance(node, ast.Return))
        mlx = next(value for key, value in zip(result.keys, result.values) if key.value == "mlx")
        for lane, training in (("trace_replay", "trace_training"), ("native_rollout", "native_training")):
            expression = next(value for key, value in zip(mlx.keys, mlx.values) if key.value == lane)
            code = compile(ast.Expression(expression), str(campaign.__file__), "eval")
            self.assertIsNone(eval(code, {training: None}))

    def test_activation_staging_preserves_stock_and_routes_f32_inputs(self) -> None:
        tree = ast.parse(Path(campaign.__file__).read_text(encoding="utf-8"))
        forward = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "forward")
        code = compile(ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[])),
                       str(campaign.__file__), "exec")

        class Tensor:
            def __init__(self, dtype):
                self.dtype = dtype

            def astype(self, dtype):
                return Tensor(dtype)

        class Text:
            hidden_size_per_layer_input = 1

            def embed_tokens(self, tokens):
                return Tensor("bf16")

            def _get_per_layer_inputs(self, tokens, embeddings):
                self.received_dtype = embeddings.dtype
                return Tensor("bf16")

        class Model:
            def __init__(self):
                self.language_model = SimpleNamespace(model=Text())

            def __call__(self, tokens, **kwargs):
                return kwargs

        model = Model()
        env = {"Any": object, "mx": SimpleNamespace(float32="f32"),
               "args": SimpleNamespace(activation_mode="stock-bf16")}
        exec(code, env)
        self.assertEqual({}, env["forward"](model, [1, 2]))
        env["args"].activation_mode = "aligned-f32"
        inputs = env["forward"](model, [1, 2])
        self.assertEqual("f32", inputs["input_embeddings"].dtype)
        self.assertEqual("f32", inputs["per_layer_inputs"].dtype)
        self.assertEqual("f32", model.language_model.model.received_dtype)
        model.language_model.model.hidden_size_per_layer_input = 0
        self.assertIsNone(env["forward"](model, [1, 2])["per_layer_inputs"])

    def test_result_serialization_handles_absent_accepted_adapter(self) -> None:
        # Exercise the actual final artifact expression without loading a model.
        tree = ast.parse(Path(campaign.__file__).read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        result = next(node.value for node in run.body if isinstance(node, ast.Return))
        antfly = next(value for key, value in zip(result.keys, result.values) if key.value == "antfly")
        checkpoint = next(value for key, value in zip(antfly.keys, antfly.values)
                          if key.value == "trained_adapter_checkpoint_sha256")
        code = compile(ast.Expression(checkpoint), str(campaign.__file__), "eval")

        def serialize(adapter_dir):
            return eval(code, {"sha256_file": campaign.sha256_file,
                               "acceptance": SimpleNamespace(trained_adapter_dir=adapter_dir)})

        self.assertIsNone(serialize(None))
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            # An advertised accepted adapter must still have its checkpoint.
            with self.assertRaises(FileNotFoundError):
                serialize(adapter)
            (adapter / "adapter_model.safetensors").write_bytes(b"checkpoint fixture")
            self.assertEqual(hashlib.sha256(b"checkpoint fixture").hexdigest(), serialize(adapter))

    def categorical_reports(self):
        policy = {"temperature": 2., "top_p": .95, "top_k": 32}
        config = {"recipe": {"optimizer": {"seed": 991}, "grpo": {"sampling": policy}}}
        train = {"schema_version": "antfly_inference_finetune_grpo_report/v8",
                 "training_order": campaign.GRPO_TRAINING_ORDER,
                 "training_seed": 991, "sampling_mode": "shared-prompt-seeded-categorical",
                 "sampling": {**policy, "first_completion_greedy": False}}
        evaluation = {"schema_version": "antfly_inference_finetune_grpo_evaluation/v4",
                      "sampling": {**policy, "first_completion_greedy": True}}
        for report in (train, evaluation):
            report['sampling'].update(algorithm='seeded-categorical-temperature-top-k-top-p',
                stream_derivation='run-seed-domain-epoch-dataset-prompt-index-completion/v2')
        return config, train, evaluation

    def test_categorical_contract_binds_seed_policy_and_phase(self):
        reports = self.categorical_reports()
        seed, policy = campaign.categorical_contract(*reports)
        self.assertEqual(seed, 991)
        self.assertEqual(policy.top_k, 32)
        mutations = (
            (1, "training_seed", 42), (1, "training_order", {}),
            (1, "sampling_mode", "unknown-seeded-categorical"),
            (2, "schema_version", "antfly_inference_finetune_grpo_evaluation/v3"),
        )
        for index, key, value in mutations:
            changed = copy.deepcopy(reports); changed[index][key] = value
            with self.subTest(key=key), self.assertRaises(campaign.MultiTokenParityError):
                campaign.categorical_contract(*changed)
        for field, value in (("first_completion_greedy", False), ("top_p", .7),
                             ("temperature", True), ("top_k", True)):
            changed = copy.deepcopy(reports); changed[2]["sampling"][field] = value
            with self.subTest(field=field), self.assertRaises(campaign.MultiTokenParityError):
                campaign.categorical_contract(*changed)

    def test_categorical_rejects_unknown_algorithm_and_stream_derivation(self):
        for index in (1, 2):
            for field in ('algorithm', 'stream_derivation'):
                reports = self.categorical_reports()
                reports[index]['sampling'][field] = 'unknown'
                with self.subTest(report=index, field=field), self.assertRaises(campaign.MultiTokenParityError):
                    campaign.categorical_contract(*reports)

    def test_categorical_cannot_claim_parity_even_when_legacy_bounds_pass(self):
        self.assertEqual("categorical-diagnostic-only", campaign.campaign_classification(
            True, True, categorical=True))
        self.assertEqual("bounded-behavior-and-update-parity", campaign.campaign_classification(
            True, True, categorical=False))

    def test_loader_admits_failed_quality_only_as_explicit_categorical_diagnostic(self):
        for profile_name, group_size, max_tokens, preset, length, lr, eps, min_kl, max_kl in (
            ('qv-multitoken', 2, 2, 'peft-qv', 128, 1e-7, 1e-4, .001, 1.),
            ('all-linear-single-token', 16, 1, 'text-all-linear', 160, 5e-8, 1e-8, .001, 1.),
            ('all-linear-single-token-quality', 16, 1, 'text-all-linear', 160, 1e-8, 1e-8, .04, 4.),
        ):
            with self.subTest(profile=profile_name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve(); model = root / 'model'; adapter = root / 'seed'
                trained = root / 'trained'; trained.mkdir()
                config, train, evaluation = self.categorical_reports()
                if profile_name in campaign.ALL_LINEAR_SINGLE_TOKEN_PROFILES:
                    config['recipe']['grpo']['sampling']['top_p'] = 1.
                    train['sampling']['top_p'] = evaluation['sampling']['top_p'] = 1.
                train_jsonl = root / 'train.jsonl'; eval_jsonl = root / 'eval.jsonl'
                train_jsonl.write_text(''.join(json.dumps({'metadata': {
                    'source_id': source_id, 'source_row_index': source_index}}) + '\n'
                    for source_id, source_index in (('source-a', 10), ('source-b', 11))))
                eval_jsonl.write_text('evaluation fixture\n')
                manifest = {
                    'train_jsonl': str(train_jsonl), 'eval_jsonl': str(eval_jsonl),
                    'train_source_ids': ['source-a', 'source-b'],
                    'train_source_row_indices': [10, 11],
                }
                config['metadata'] = {'dataset_fingerprints': [{
                    'label': 'dataset', 'path': str(train_jsonl),
                    'digest': 'sha256:' + campaign.sha256_file(train_jsonl),
                    'size_bytes': train_jsonl.stat().st_size,
                }]}
                recipe = config['recipe']
                recipe.update(model={'family': 'gemma4', 'path': str(model)},
                    adapter={'path': str(adapter), 'rank': 16, 'alpha': 32, 'target_preset': preset},
                    dataset={'path': manifest['train_jsonl'], 'max_examples': 2, 'max_seq_len': length},
                    eval={'path': manifest['eval_jsonl'], 'max_examples': 2})
                recipe['optimizer'].update(learning_rate=lr, epochs=1, gradient_accumulation_steps=1, max_grad_norm=1.)
                recipe['grpo'].update(group_size=group_size, max_completion_tokens=max_tokens, advantage_eps=eps, adaptive_kl=True,
                    normalize_advantage=None if profile_name in campaign.ALL_LINEAR_SINGLE_TOKEN_PROFILES else True, train_max_kl_policy='skip_group',
                    clip_epsilon=.2, kl_coef=.04, train_max_kl=.1, target_kl=.01,
                    kl_horizon=100., min_kl_coef=min_kl, max_kl_coef=max_kl)
                train.update(execution_mode='train', dataset_format='rendered-text-grpo', groups=2,
                    completions=2*group_size, optimizer_steps=0, policy_backend='metal', mean_kl=0.,
                    optimizer_groups=0, zero_reward_std_groups=2, all_truncated_groups=0,
                    kl_rejected_groups=0, frac_reward_zero_std=1., frac_kl_rejected=0.,
                    loss_type='bnpo', scale_rewards='group', epsilon_low=campaign._f32(.2), epsilon_high=campaign._f32(.2),
                    max_completion_tokens=max_tokens, mask_truncated_completions=False, num_iterations=1,
                    truncated_completions=2*group_size, frac_completions_truncated=1., trained_adapter_dir=str(trained))
                evaluation.update(status='failed-quality-gate', groups=2, mean_kl=0., mask_truncated_completions=False)
                path = root / 'grpo_kl_control_trace.jsonl'; path.write_text('')
                kl_control = {k: campaign.GRPO[k] for k in (
                    'train_max_kl', 'target_kl', 'kl_horizon', 'initial_kl_coef')}
                kl_control.update(min_kl_coef=min_kl, max_kl_coef=max_kl)
                train['kl_control'] = dict(mode='adaptive', budget_policy='skip_group', admitted_groups=0,
                    rejected_groups=0, trace_path=str(path), trace_digest='sha256:' + campaign.sha256_file(path),
                    **kl_control)
                for phase, filename, report in (('train', 'grpo_reward_trace.jsonl', train),
                        ('evaluation', 'grpo_evaluation_reward_trace.jsonl', evaluation)):
                    path = root / filename
                    path.write_text(''.join(json.dumps(dict(schema_version=campaign.REWARD_TRACE_SCHEMA_VERSION,
                        phase=phase, call_index=i, prompt_index=1-i//group_size, completion_tokens=[10, 11][:max_tokens],
                        aggregate_reward=1.)) + '\n' for i in range(2*group_size)))
                    report['reward_pipeline'] = {'trace_digest': 'sha256:' + campaign.sha256_file(path)}
                def save():
                    for name, payload in (('training_config.json', config), ('grpo_report.json', train),
                                          ('grpo_evaluation_report.json', evaluation)):
                        (root / name).write_text(json.dumps(payload))
                save(); spec = campaign.CampaignSpec('gemma-4-E2B-it', 2, 2, group_size, max_tokens, profile_name)
                result = campaign.load_acceptance(root, manifest, spec, model, adapter, categorical_diagnostic=True)
                self.assertEqual([g.prompt_index for g in result.train_trace], [1, 0])
                self.assertEqual(result.eval_report['status'], 'failed-quality-gate')
                with self.assertRaises(campaign.MultiTokenParityError):
                    campaign.load_acceptance(root, manifest, spec, model, adapter)
                published = train['trained_adapter_dir']; train['trained_adapter_dir'] = None
                train['evaluation'] = {'passed': False}; save()
                rejected = campaign.load_acceptance(root, manifest, spec, model, adapter, categorical_diagnostic=True)
                self.assertIsNone(rejected.trained_adapter_dir)
                train['evaluation']['passed'] = True; save()
                with self.assertRaises(campaign.MultiTokenParityError):
                    campaign.load_acceptance(root, manifest, spec, model, adapter, categorical_diagnostic=True)
                train['trained_adapter_dir'] = published; del train['evaluation']
                for section, key, wrong in (
                    ('adapter', 'target_preset', 'text-all-linear' if preset == 'peft-qv' else 'peft-qv'),
                    ('dataset', 'max_seq_len', length - 1),
                    ('optimizer', 'learning_rate', lr * 2),
                    ('grpo', 'advantage_eps', eps * 2),
                ):
                    original = recipe[section][key]; recipe[section][key] = wrong; save()
                    with self.assertRaises(campaign.MultiTokenParityError):
                        campaign.load_acceptance(root, manifest, spec, model, adapter, categorical_diagnostic=True)
                    recipe[section][key] = original
                recipe['grpo']['normalize_advantage'] = False; save()
                with self.assertRaises(campaign.MultiTokenParityError):
                    campaign.load_acceptance(root, manifest, spec, model, adapter, categorical_diagnostic=True)
                recipe['grpo']['normalize_advantage'] = None
                original = train['epsilon_low']; train['epsilon_low'] = .20000001; save()
                with self.assertRaises(campaign.MultiTokenParityError):
                    campaign.load_acceptance(root, manifest, spec, model, adapter, categorical_diagnostic=True)
                train['epsilon_low'] = original
                train['optimizer_steps'] = 1; save()
                with self.assertRaisesRegex(campaign.MultiTokenParityError, 'counts drifted'):
                    campaign.load_acceptance(root, manifest, spec, model, adapter, categorical_diagnostic=True)


    def test_all_linear_profile_requires_its_pinned_group_and_token_counts(self):
        from dataclasses import replace
        spec = campaign.CampaignSpec('gemma-4-E4B-it', 1960, 256, 16, 1, 'all-linear-single-token')
        spec.validate()
        for fields in ({'group_size': 8}, {'max_completion_tokens': 2}, {'recipe_profile': 'unknown'}):
            with self.subTest(fields=fields), self.assertRaises(campaign.MultiTokenParityError):
                replace(spec, **fields).validate()

    def test_full_profile_row_length_reserves_the_completion_token(self):
        from types import SimpleNamespace
        class Tokenizer:
            def encode(self, value, **kwargs):
                return SimpleNamespace(ids=[1] if value == 'yes' else [2] * len(value))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rows.jsonl'
            def write(length):
                path.write_text(json.dumps({'prompt': 'x' * length, 'target': 'yes',
                    'metadata': {'prompt_tokens': length, 'target_tokens': 1,
                        'source_split': 'train', 'source_row_index': 0, 'source_id': 'id'}}) + '\n')
            args = dict(expected_count=1, expected_ids=['id'], expected_indices=[0],
                tokenizer=Tokenizer(), max_completion_tokens=1, sequence_length=160)
            write(159)
            self.assertEqual(159, len(campaign.load_rows(path, **args)[0].prompt_token_ids))
            write(160)
            with self.assertRaisesRegex(campaign.MultiTokenParityError, 'length contract'):
                campaign.load_rows(path, **args)

    def test_group_admission_matches_zero_reward_and_raw_kl_order(self):
        self.assertEqual("zero-reward-std-skipped", campaign.group_admission([1., 1.], 1.))
        self.assertEqual("budget-exceeded-skipped", campaign.group_admission([0., 1.], .11))
        self.assertEqual("admitted", campaign.group_admission([0., 1.], .1))
        self.assertEqual("admitted", campaign.group_admission([0., 1.], 0.10000000149011612))
        self.assertEqual("budget-exceeded-skipped", campaign.group_admission([0., 1.], .10000002))
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(campaign.MultiTokenParityError):
                campaign.group_admission([0., 1.], value)

    def test_categorical_overlap_counts_repeated_completions(self):
        same = campaign.sequence_overlap([[1], [1]], [[1], [1]], with_replacement=True)
        self.assertEqual(same['sequence_recall'], 1.)
        self.assertEqual(same['first_token_recall'], 1.)
        different = campaign.sequence_overlap([[1], [1], [2]], [[1], [2], [2]], with_replacement=True)
        self.assertEqual(different['sequence_recall'], 2 / 3)
        self.assertTrue(different['exact_sequence_set'])
        self.assertFalse(different['exact_sequence_multiset'])
        with self.assertRaises(campaign.MultiTokenParityError):
            campaign.sequence_overlap([[1], [1]], [[1], [1]])

    def test_categorical_kl_trace_accounts_for_skips_without_advancing_adam(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / 'grpo_kl_control_trace.jsonl'
            groups = tuple(campaign.TraceGroup(i, tuple(
                campaign.TraceCompletion((10,), reward) for reward in rewards
            )) for i, rewards in ((2, (1., 1.)), (0, (0., 1.)), (3, (0., 1.)), (1, (0., 1.))))
            coefficient = .04; records = []
            for index, prompt, steps, observed, status in (
                (1, 0, 0, .0, 'admitted'), (2, 3, 1, .2, 'budget-exceeded-skipped'),
                (3, 1, 1, .0, 'admitted'),
            ):
                after = campaign.adaptive_kl_update(coefficient, observed, 2)
                records.append(dict(schema_version=campaign.KL_TRACE_SCHEMA_VERSION,
                    group_index=index, epoch_index=0, prompt_index=prompt, optimizer_steps_before=steps,
                    observed_completions=2,
                    status=status, budget_policy='skip_group', mean_kl=observed,
                    weighted_kl_loss=coefficient*observed, train_max_kl=.1, target_kl=.01,
                    kl_coef_before=coefficient, kl_coef_after=after,
                    objective_kl_coef=coefficient))
                coefficient = after
            report = dict(optimizer_steps=2, optimizer_groups=2, zero_reward_std_groups=1,
                          all_truncated_groups=0, kl_rejected_groups=1, frac_reward_zero_std=.25,
                          frac_kl_rejected=.25, kl_control=dict(mode='adaptive', budget_policy='skip_group',
                          admitted_groups=2, rejected_groups=1, trace_path=str(path), **{
                              k: campaign.GRPO[k] for k in ('train_max_kl', 'target_kl', 'kl_horizon',
                                  'initial_kl_coef', 'min_kl_coef', 'max_kl_coef')}))
            spec = campaign.CampaignSpec('gemma-4-E2B-it', 4, 2, 2, 2)
            def validate(rows, payload):
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                payload['kl_control']['trace_digest'] = 'sha256:' + campaign.sha256_file(path)
                return campaign.validate_categorical_groups(root, payload, spec, groups)
            self.assertEqual(campaign.KL_TRACE_SCHEMA_VERSION, validate(records, report))
            legacy = copy.deepcopy(records); coefficient = .04
            for row in legacy:
                row['schema_version'] = 'antfly_inference_grpo_kl_control_trace/v2'
                row.pop('observed_completions')
                row.pop('objective_kl_coef')
                row['kl_coef_before'] = coefficient
                row['weighted_kl_loss'] = coefficient * row['mean_kl']
                if row['status'] == 'admitted':
                    coefficient = campaign.adaptive_kl_update(coefficient, row['mean_kl'])
                row['kl_coef_after'] = coefficient
            self.assertEqual(
                'antfly_inference_grpo_kl_control_trace/v2',
                validate(legacy, copy.deepcopy(report)),
            )
            v3 = copy.deepcopy(records); coefficient = .04
            for row in v3:
                row['schema_version'] = 'antfly_inference_grpo_kl_control_trace/v3'
                row.pop('observed_completions')
                row.pop('objective_kl_coef')
                row['kl_coef_before'] = coefficient
                row['weighted_kl_loss'] = coefficient * row['mean_kl']
                coefficient = campaign.adaptive_kl_update(coefficient, row['mean_kl'])
                row['kl_coef_after'] = coefficient
            self.assertEqual(
                'antfly_inference_grpo_kl_control_trace/v3',
                validate(v3, copy.deepcopy(report)),
            )
            for index, key, value in ((1, 'optimizer_steps_before', 2), (1, 'kl_coef_after', .05),
                                      (2, 'group_index', 2), (1, 'prompt_index', 2)):
                changed = copy.deepcopy(records); changed[index][key] = value
                with self.subTest(key=key), self.assertRaises(campaign.MultiTokenParityError):
                    validate(changed, copy.deepcopy(report))
            changed = copy.deepcopy(report); changed['optimizer_steps'] = 3
            with self.assertRaises(campaign.MultiTokenParityError):
                validate(records, changed)

    def test_accepts_epoch_shuffled_report_schema_v8(self) -> None:
        self.assertIn(
            "antfly_inference_finetune_grpo_report/v8",
            campaign.GRPO_REPORT_SCHEMA_VERSIONS,
        )
        self.assertIn(
            "antfly_inference_finetune_grpo_evaluation/v4",
            campaign.GRPO_EVAL_SCHEMA_VERSIONS,
        )

    def test_shared_native_rollout_guard_rejects_stochastic_sampler_drift(self) -> None:
        with self.assertRaisesRegex(
            campaign.legacy.BoolQParityContractError,
            "retired deterministic ranked sampler",
        ):
            campaign.legacy.require_native_rollout_sampler_compatibility(
                {"sampling_mode": "shared-prompt-seeded-categorical"}
            )

    def test_multitoken_update_bounds_require_vector_distance(self) -> None:
        metrics = {
            "delta_cosine_similarity": 0.98,
            "delta_l2_relative_difference": 0.0,
            "delta_vector_l2_relative_error": 0.2,
        }
        checks = campaign.legacy.adapter_update_checks(
            metrics, min_cosine=0.95, max_relative_error=0.1
        )
        self.assertTrue(checks["adapter_delta_direction"])
        self.assertTrue(checks["adapter_delta_norm"])
        self.assertFalse(checks["adapter_delta_vector"])
        for value in (None, True, -1.0, float("inf"), float("nan")):
            metrics["delta_vector_l2_relative_error"] = value
            self.assertFalse(all(campaign.legacy.adapter_update_checks(
                metrics, min_cosine=0.95, max_relative_error=0.1
            ).values()))

    def test_campaign_shape_requires_a_real_multi_token_matrix(self) -> None:
        campaign.CampaignSpec("gemma-4-E2B-it", 8, 16, 4, 4).validate()
        with self.assertRaisesRegex(campaign.MultiTokenParityError, "multi-token"):
            campaign.CampaignSpec("gemma-4-E2B-it", 8, 16, 4, 1).validate()
        with self.assertRaisesRegex(campaign.MultiTokenParityError, "at least two"):
            campaign.CampaignSpec("gemma-4-E4B-it", 1, 16, 4, 4).validate()

    def test_adaptive_kl_rule_is_bounded_and_applies_to_next_group(self) -> None:
        below = campaign.adaptive_kl_update(0.04, 0.0)
        self.assertAlmostEqual(0.03992, below, places=8)
        below_group = campaign.adaptive_kl_update(0.04, 0.0, 16)
        self.assertAlmostEqual(0.03872, below_group, places=8)
        above = campaign.adaptive_kl_update(below, 1.0)
        self.assertGreater(above, below)
        current = campaign.GRPO["max_kl_coef"]
        self.assertEqual(current, campaign.adaptive_kl_update(current, 1.0))
        quality_floor = campaign.adaptive_kl_update(
            0.04, 0.0, 16, min_kl_coef=0.04, max_kl_coef=4.0,
        )
        self.assertEqual(campaign._f32(0.04), quality_floor)
        quality_ceiling = campaign.adaptive_kl_update(
            4.0, 1.0, 16, min_kl_coef=0.04, max_kl_coef=4.0,
        )
        self.assertEqual(4.0, quality_ceiling)

    def test_stable_raw_k3_is_independent_of_beta(self) -> None:
        expected = math.expm1(0.5) - 0.5
        self.assertAlmostEqual(
            expected,
            campaign.mean_k3([-1.0], [-0.5]),
            places=12,
        )
        self.assertEqual(0.0, campaign.mean_k3([-1.0], [-1.0]))

    def test_prefix_reward_matches_case_sensitive_antfly_contract(self) -> None:
        self.assertEqual(1.0, campaign.prefix_match_reward(" yes indeed\n", "yes"))
        self.assertEqual(0.0, campaign.prefix_match_reward("Yes indeed", "yes"))
        self.assertEqual(0.0, campaign.prefix_match_reward("indeed yes", "yes"))

    def test_sequence_overlap_keeps_full_sequence_and_first_token_evidence(self) -> None:
        overlap = campaign.sequence_overlap(
            [[1, 2], [3, 4]],
            [[1, 9], [3, 4]],
        )
        self.assertEqual(0.5, overlap["sequence_recall"])
        self.assertEqual(1.0, overlap["first_token_recall"])
        self.assertFalse(overlap["top_sequence_match"])
        self.assertTrue(overlap["top1_first_token_match"])

    def test_trace_loader_accepts_variable_length_eos_completions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            sequences = ([10, 11, 1], [20, 1])
            rows = []
            for index, sequence in enumerate(sequences):
                rows.append(
                    {
                        "schema_version": campaign.REWARD_TRACE_SCHEMA_VERSION,
                        "phase": "train",
                        "call_index": index,
                        "prompt_index": 0,
                        "completion_tokens": list(sequence),
                        "aggregate_reward": float(index == 0),
                    }
                )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            groups = campaign.load_trace(
                path,
                phase="train",
                expected_groups=1,
                group_size=2,
                max_completion_tokens=4,
            )
            self.assertEqual(tuple(tuple(row) for row in sequences), groups[0].sequences)
            self.assertEqual((10, 20), groups[0].first_token_ids)

    def test_trace_preserves_shuffled_updates_and_rejects_interleaving(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            for prompts in ((1, 1, 0, 0), (0, 1, 0, 1)):
                rows = [
                    {
                        "schema_version": campaign.REWARD_TRACE_SCHEMA_VERSION,
                        "phase": "train",
                        "call_index": index,
                        "prompt_index": prompt,
                        "completion_tokens": [10 + prompt, 1],
                        "aggregate_reward": 1.0,
                    }
                    for index, prompt in enumerate(prompts)
                ]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                if prompts == (0, 1, 0, 1):
                    with self.assertRaisesRegex(campaign.MultiTokenParityError, "interleaved"):
                        campaign.load_trace(path, phase="train", expected_groups=2, group_size=2, max_completion_tokens=4)
                else:
                    groups = campaign.load_trace(path, phase="train", expected_groups=2, group_size=2, max_completion_tokens=4)
                    self.assertEqual([1, 0], [group.prompt_index for group in groups])
                    self.assertEqual(((11, 1), (11, 1)), groups[0].sequences)

    def test_materialization_loader_accepts_current_v2_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (model / name).write_text("{}", encoding="utf-8")
            train = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            train.write_text("train\n", encoding="utf-8")
            evaluation.write_text("eval\n", encoding="utf-8")
            policy = {
                "ordering": campaign.boolq_materializer.V2_SELECTION_ORDERING,
                "train_skip_per_label": 0,
                "evaluation_skip_per_label": 0,
                "evaluation_excluded_source_ids": 0,
                "evaluation_exclusion_evidence_source_ids": 0,
                "dataset_format": "rendered-text-grpo",
                "max_seq_len": campaign.SEQUENCE_LENGTH,
                "max_completion_tokens": 4,
                "target_tokens": 1,
                "rendered_prompt_truncation": "forbidden",
                "response_channel": "final",
            }
            manifest = {
                "schema_version": campaign.MATERIALIZATION_SCHEMA_VERSION,
                "dataset": {
                    "repo_id": "google/boolq",
                    "revision": "a" * 40,
                    "selection_policy": policy,
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
                "train_source_ids": ["1" * 64, "2" * 64],
                "eval_source_ids": ["3" * 64, "4" * 64],
                "evaluation_exclusion_manifests": [],
                "tokenizer_files": {
                    name: campaign.sha256_file(model / name)
                    for name in ("tokenizer.json", "tokenizer_config.json")
                },
            }
            manifest["semantic_sha256"] = (
                campaign.boolq_materializer.canonical_sha256(manifest)
            )
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = campaign.load_materialization(
                path,
                campaign.CampaignSpec("gemma-4-E2B-it", 2, 2, 2, 4),
                model,
            )
            self.assertEqual(campaign.MATERIALIZATION_SCHEMA_VERSION, loaded["schema_version"])
            # Independently admitted training/evaluation sets retain both source manifests.
            full = campaign.CampaignSpec('gemma-4-E2B-it', 2, 2, 16, 1, 'all-linear-single-token')
            def write_manifest(destination, value):
                unsigned = {key: item for key, item in value.items() if key != 'semantic_sha256'}
                value['semantic_sha256'] = campaign.boolq_materializer.canonical_sha256(unsigned)
                destination.write_text(json.dumps(value))
            manifest['dataset']['selection_policy']['max_completion_tokens'] = 1
            write_manifest(path, manifest)
            fresh = copy.deepcopy(manifest)
            fresh_path = root / 'fresh-manifest.json'; fresh_jsonl = root / 'fresh.jsonl'
            fresh_jsonl.write_text('fresh evaluation\n')
            fresh['eval_jsonl'] = str(fresh_jsonl)
            fresh['train_source_ids'] = ['7' * 64, '8' * 64]
            fresh['eval_source_ids'] = ['5' * 64, '6' * 64]
            fresh['dataset']['selection_policy']['max_seq_len'] = 160
            fresh['dataset']['evaluation']['materialized_jsonl_sha256'] = campaign.sha256_file(fresh_jsonl)
            write_manifest(fresh_path, fresh)
            view = campaign.load_campaign_materialization(path, fresh_path, full, model)
            self.assertEqual(str(train), view['train_jsonl'])
            self.assertEqual(str(fresh_jsonl), view['eval_jsonl'])
            self.assertNotIn('semantic_sha256', view)
            self.assertEqual(128, view['dataset']['train_selection_policy']['max_seq_len'])
            self.assertEqual(160, view['dataset']['evaluation_selection_policy']['max_seq_len'])
            self.assertEqual(campaign.sha256_file(fresh_path), view['campaign_manifest_bindings']['evaluation']['sha256'])
            fresh['eval_source_ids'][0] = '1' * 64; write_manifest(fresh_path, fresh)
            with self.assertRaisesRegex(campaign.MultiTokenParityError, 'identities overlap'):
                campaign.load_campaign_materialization(path, fresh_path, full, model)
            fresh['eval_source_ids'][0] = '5' * 64
            fresh['dataset']['revision'] = 'b' * 40; write_manifest(fresh_path, fresh)
            with self.assertRaisesRegex(campaign.MultiTokenParityError, 'revisions differ'):
                campaign.load_campaign_materialization(path, fresh_path, full, model)
            fresh['dataset']['revision'] = 'a' * 40
            fresh['dataset']['selection_policy']['max_seq_len'] = 161; write_manifest(fresh_path, fresh)
            with self.assertRaisesRegex(campaign.MultiTokenParityError, 'admission length'):
                campaign.load_campaign_materialization(path, fresh_path, full, model)


    def test_import_surface_keeps_mlx_lazy(self) -> None:
        source = Path(campaign.__file__).read_text(encoding="utf-8")
        prefix = source.split("def run(args", 1)[0]
        self.assertNotIn("import mlx.core", prefix)
        self.assertNotIn("import mlx.nn", prefix)

    def test_mlx_candidate_contract_stays_batch_one_and_fail_closed(self) -> None:
        source = Path(campaign.__file__).read_text(encoding="utf-8")
        scoring = source.split("        def score_sequences(", 1)[1].split(
            "        def ranked_group(", 1
        )[0]
        rollout = source.split("        def ranked_group(", 1)[1].split(
            "        def flatten(", 1
        )[0]
        optimizer_step = source.split("            def step(", 1)[1].split(
            "            compiled_step =", 1
        )[0]
        training_loop = source.split("        def train_lane(", 1)[1].split(
            "        def evaluate_lane(", 1
        )[0]

        self.assertIn("for sequence in sequences:", scoring)
        self.assertIn("padded_sequence(row, sequence)", scoring)
        self.assertIn("for completion_index in active_indices:", rollout)
        self.assertNotIn("model(tokens).astype(mx.float32)[:, row_index", rollout)
        self.assertIn(
            "tokens[completion_index : completion_index + 1]", optimizer_step
        )
        self.assertIn("tree_map(", optimizer_step)
        self.assertIn("sampling_rescore_max_abs_error > 1.0e-4", training_loop)
        self.assertIn("raw_mean_kl > GRPO[\"train_max_kl\"]", training_loop)
        self.assertIn(
            "executed_train_groups == spec.train_groups", training_loop
        )
        self.assertIn(
            '"unavailable-antfly-adapter-is-full-horizon"', training_loop
        )

    def test_quality_profile_serializes_its_kl_bounds(self) -> None:
        source = Path(campaign.__file__).read_text(encoding="utf-8")
        result_contract = source.split('"contract": {', 1)[1].split(
            '"dataset": {', 1
        )[0]
        self.assertIn('"min_kl_coef": spec.profile.min_kl_coef', result_contract)
        self.assertIn('"max_kl_coef": spec.profile.max_kl_coef', result_contract)


if __name__ == "__main__":
    unittest.main()
