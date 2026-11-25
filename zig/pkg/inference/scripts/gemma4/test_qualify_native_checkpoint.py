# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import qualify_native_checkpoint as collector
from test_native_whole_head_campaign import ResourceContract
from qualify_native_checkpoint import ERRORS, validate_fault

"""Synthetic freeze and worker orchestration; never load models or fresh gold."""


SCRIPTS = Path(__file__).resolve().parent

if not (SCRIPTS / 'testdata/native_whole_head_quality_v2.json').exists():
    SCRIPTS = Path(__file__).resolve().parents[3] / 'zig/pkg/inference/scripts/gemma4'

class FrozenCollection(unittest.TestCase):
    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + '\n')
        return {'path': name, 'sha256': collector.sha(path)}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = {'head': 'a' * 40, 'source_snapshot_sha256': 'b' * 64}
        self.binary_path = self.root / 'context-tests'
        self.binary_path.write_bytes(b'consumed synthetic executable, never run')
        self.binary = {'path': str(self.binary_path), 'sha256': collector.sha(self.binary_path)}
        self.build = {'binary_path': self.binary_path, 'binary_sha256': self.binary['sha256'],
                      'head': self.source['head'], 'source': {'synthetic': True}}
        self.configuration = json.loads((SCRIPTS / 'testdata/native_whole_head_quality_v2.json').read_bytes())['report']['frozen']
        self.configuration = {key: self.configuration[key] for key in ('observed', 'execution', 'configuration_sha256')}
        self.configuration['implementation_sha256'] = [37] * 32
        self.environment = {collector.FLAGS[key]: '1' if self.configuration['execution'][key] else '0'
                            for key in collector.EXECUTION_FIELDS}
        self.probe = {'passed': True, 'source': self.source, 'binary': self.binary,
                      'configuration': self.configuration, 'runtime_environment': self.environment}
        probe_ref = self.put('probe.json', self.probe)
        self.audit = {'passed': True, 'source': self.source,
                      'probe_observations_sha256': probe_ref['sha256'],
                      'backend_environment_sha256': bytes(self.configuration['observed']['backend_environment_sha256']).hex()}
        archive = self.root / 'preparation.zip'
        archive.write_bytes(b'synthetic preparation source archive')
        corpus = 'c' * 64
        self.preparation = {'schema': 'antfly.native-quality-preparation.audit.v2', 'passed': True,
                            'prepared_cases': 2000, 'model_quality_outputs_observed': False,
                            'model_quality_qualified': False, 'release_eligible': False,
                            'protected_prefix_rows': 128, 'protected_recent_rows': 2048,
                            'corpus_fingerprint_sha256': corpus, 'source': self.source, 'binary': self.binary,
                            'cells': [{'model': model, 'context_tokens': context, 'bucket': bucket, 'cases': 100}
                                      for model in collector.contract.MODELS for context in (8192, 32768)
                                      for bucket in sorted(collector.BUCKETS)]}
        self.preparation_seal = {'schema': 'antfly.native-quality-preparation.source-seal.v1',
                                 'preparation_only': True, 'release_eligible': False,
                                 'source': self.source, 'producer_binary': self.binary,
                                 'archive': {'path': str(archive), 'sha256': collector.sha(archive),
                                             'bytes': archive.stat().st_size}}
        self.index = {'schema': 'antfly.native-quality-input-preparation.v1', 'preparation_only': True,
                      'model_quality_outputs_observed': False, 'corpus_fingerprint_sha256': corpus,
                      'scoring_cases': self.put('scoring.json', []),
                      'preparation_audit': self.put('preparation-audit.json', self.preparation),
                      'preparation_source_seal': self.put('preparation-seal.json', self.preparation_seal),
                      'batches': []}
        model_specs = {}
        for model, geometry in collector.contract.MODELS.items():
            identity = {key: [11 + len(model)] * 32 for key in ('weights', 'tokenizer', 'template', 'layout')}
            model_specs[model] = {'loaded_identity': {key: bytes(value).hex() for key, value in identity.items()},
                                  'artifact_sha256': 'd' * 64}
            for encoding in sorted(collector.contract.ENCODINGS):
                for context in (8192, 32768):
                    for bucket in sorted(collector.BUCKETS):
                        manifest = {'schema': 'antfly.native-checkpoint-quality.input.v1', 'model': geometry[0],
                                    'native_identity': identity, 'main_gguf_sha256': 'd' * 64,
                                    'cases': [{'id': f'synthetic-{i}', 'bucket': bucket, 'prompt_count': context - 256,
                                               'request': {'encoding': encoding, 'output_tokens': 256,
                                                           'minimum_savings_ratio': 1.5}} for i in range(100)]}
                        index = len(self.index['batches'])
                        self.index['batches'].append({'model': model, 'encoding': encoding, 'context_tokens': context,
                                                      'bucket': bucket, 'cases': 100,
                                                      'input': self.put(f'manifest-{index}.json', manifest)})
        self.frozen = {'schema': 'antfly.native-whole-head.campaign-freeze.v2', 'runtime_frozen': True,
                       'release_authority': False, 'build': self.put('build.json', {'synthetic': True}),
                       'source': self.source, 'binary': self.binary, 'configuration': probe_ref,
                       'configuration_audit': self.put('probe-audit.json', self.audit),
                       'implementation_sha256': bytes(self.configuration['implementation_sha256']).hex(),
                       'runtime_environment': self.environment, 'corpus_fingerprint_sha256': corpus,
                       'input_index': self.put('index.json', self.index), 'models': model_specs, 'performance': []}
        self.job = {'source': self.source, 'binary': self.binary, 'runtime_environment': self.environment,
                    'producers': [self.put('producer.json', {'consumed': True})]}
        job = self.put('performance-job.json', self.job)
        resource = self.put('resource.json', ResourceContract().report())
        report = self.put('performance.json', {'synthetic': True})
        log = self.put('worker-log.json', {'synthetic': True})
        for model in collector.contract.MODELS:
            for encoding in sorted(collector.contract.ENCODINGS):
                for context in (8192, 32768):
                    self.frozen['performance'].append({'model': model, 'encoding': encoding, 'context_tokens': context,
                                                        'request': {'path': 'synthetic', 'sha256': 'e' * 64},
                                                        'observations': report, 'resource': resource,
                                                        'worker_log': log, 'job': job})
        self.freeze_ref = self.put('freeze.json', self.frozen)
        for target, value in [('builds.validate', self.build), ('source_snapshot', (self.source['head'], self.build['source'])),
                              ('git_provenance', self.source),
                              ('contract.performance_contract', {'performance_gate_passed': True})]:
            mock = patch.object(collector, target, return_value=value) if '.' not in target else patch(
                'qualify_native_checkpoint.' + target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def refresh(self):
        self.frozen['input_index'] = self.put('index.json', self.index)
        self.freeze_ref = self.put('freeze.json', self.frozen)

    def use_encoding_scope(self, encodings=('f16', 'fp8')):
        self.frozen['qualification_encodings'] = list(encodings)
        self.index['qualification_encodings'] = list(encodings)
        self.index['batches'] = [row for row in self.index['batches'] if row['encoding'] in encodings]
        self.frozen['performance'] = [row for row in self.frozen['performance'] if row['encoding'] in encodings]
        self.refresh()

    def validate(self):
        self.refresh()
        return collector.validate_freeze(self.root, self.root, self.freeze_ref)

    def use_current_execution(self):
        self.configuration['execution']['resident_gated_ffn_fusion'] = False
        self.environment['ANTFLY_METAL_ATTENTION_MATCH_GATED_FFN_FUSION'] = '0'
        self.frozen['configuration'] = self.put('probe.json', self.probe)
        self.audit['probe_observations_sha256'] = self.frozen['configuration']['sha256']
        self.frozen['configuration_audit'] = self.put('probe-audit.json', self.audit)
        job = self.put('performance-job.json', self.job)
        for cell in self.frozen['performance']:
            cell['job'] = job

    def test_current_probe_binds_the_actual_fusion_environment_key(self):
        self.use_current_execution()
        self.validate()
        self.assertEqual(collector.FLAGS['resident_gated_ffn_fusion'],
                         'ANTFLY_METAL_ATTENTION_MATCH_GATED_FFN_FUSION')
        self.environment['ANTFLY_METAL_ATTENTION_MATCH_GATED_FFN_FUSION'] = '1'
        self.frozen['configuration'] = self.put('probe.json', self.probe)
        with self.assertRaisesRegex(ValueError, 'execution flags'):
            self.validate()

    def test_complete_frozen_matrix_passes_without_reading_scoring_answers(self):
        # Invalid JSON is intentional: the preflight hashes scoring bytes only.
        (self.root / 'scoring.json').write_bytes(b'not decoded before model generation')
        self.index['scoring_cases']['sha256'] = collector.sha(self.root / 'scoring.json')
        frozen, build, index, configuration = self.validate()
        self.assertEqual(len(index['batches']), 60)
        self.assertEqual(len(frozen['performance']), 12)
        self.assertEqual(configuration, self.configuration)

    def test_explicit_f16_fp8_scope_requires_all_forty_batches_and_eight_performance_cells(self):
        self.use_encoding_scope()
        frozen, _, index, _ = self.validate()
        self.assertEqual(frozen['qualification_encodings'], ['f16', 'fp8'])
        self.assertEqual(len(index['batches']), 40)
        self.assertEqual(len(frozen['performance']), 8)
        self.frozen['performance'].pop()
        with self.assertRaisesRegex(ValueError, 'prerequisite matrix is incomplete'):
            self.validate()

    def test_narrow_scope_rejects_missing_batch_and_undeclared_extra_codec(self):
        extra = copy.deepcopy(next(row for row in self.index['batches'] if row['encoding'] == 'rotated4'))
        self.use_encoding_scope()
        saved = self.index['batches'].pop()
        with self.assertRaisesRegex(ValueError, 'incomplete frozen native cohort'):
            self.validate()
        self.index['batches'].append(saved)
        self.index['batches'][0] = extra
        with self.assertRaisesRegex(ValueError, 'duplicate, missing or undersized'):
            self.validate()

    def test_scope_cannot_be_inferred_from_missing_rows_or_changed_in_one_artifact(self):
        self.use_encoding_scope()
        del self.frozen['qualification_encodings']
        with self.assertRaisesRegex(ValueError, 'qualification encodings differ'):
            self.validate()
        self.frozen['qualification_encodings'] = ['f16']
        with self.assertRaisesRegex(ValueError, 'qualification encodings differ'):
            self.validate()

    def test_invalid_declared_scopes_are_rejected(self):
        for values in (None, [], 'fp8', ['f16', 'f16'], ['fp16'], [True], ['fp8', 'unknown']):
            with self.subTest(values=values):
                self.frozen['qualification_encodings'] = values
                with self.assertRaisesRegex(ValueError, 'invalid qualification encodings'):
                    self.validate()

    def test_runtime_source_build_and_compiled_identity_are_bound(self):
        with patch.object(collector, 'git_provenance', return_value={}):
            with self.assertRaisesRegex(ValueError, 'runtime source changed'):
                self.validate()
        with patch.object(collector.builds, 'validate', side_effect=ValueError('scoped build')):
            with self.assertRaisesRegex(ValueError, 'scoped build'):
                self.validate()
        self.frozen['implementation_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'compiled implementation identity'):
            self.validate()

    def test_configuration_environment_and_independent_audit_are_required(self):
        self.frozen['runtime_environment'] = dict(self.environment, UNTRACKED='1')
        with self.assertRaisesRegex(ValueError, 'execution flags changed'):
            self.validate()
        self.frozen['runtime_environment'] = self.environment
        self.audit['probe_observations_sha256'] = 'f' * 64
        self.frozen['configuration_audit'] = self.put('probe-audit.json', self.audit)
        with self.assertRaisesRegex(ValueError, 'independently audited'):
            self.validate()

    def test_preparation_must_remain_unexposed_complete_and_archived(self):
        self.preparation['model_quality_outputs_observed'] = True
        self.index['preparation_audit'] = self.put('preparation-audit.json', self.preparation)
        with self.assertRaisesRegex(ValueError, 'exposed preparation'):
            self.validate()
        self.preparation['model_quality_outputs_observed'] = False
        self.preparation['cells'].pop()
        self.index['preparation_audit'] = self.put('preparation-audit.json', self.preparation)
        with self.assertRaisesRegex(ValueError, 'audited preparation cohort'):
            self.validate()
        self.preparation['cells'] = copy.deepcopy(self.preparation['cells']) + [self.preparation['cells'][0]]
        self.index['preparation_audit'] = self.put('preparation-audit.json', self.preparation)
        (self.root / 'preparation.zip').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'archive changed'):
            self.validate()

    def test_model_identity_and_case_reservation_cannot_be_substituted(self):
        path = self.root / self.index['batches'][0]['input']['path']
        manifest = json.loads(path.read_bytes())
        original = copy.deepcopy(manifest)
        for mutate, reason in [(lambda x: x['native_identity']['layout'].__setitem__(0, 0), 'model identity'),
                               (lambda x: x['cases'][0].update(prompt_count=8192), 'reservation'),
                               (lambda x: x['cases'][0]['request'].update(output_tokens=64), 'reservation')]:
            manifest = copy.deepcopy(original)
            mutate(manifest)
            self.index['batches'][0]['input'] = self.put(path.name, manifest)
            with self.assertRaisesRegex(ValueError, reason):
                self.validate()

    def test_performance_requires_all_twelve_passing_cells_and_runtime_job(self):
        removed = self.frozen['performance'].pop()
        with self.assertRaisesRegex(ValueError, 'matrix is incomplete'):
            self.validate()
        self.frozen['performance'].append(removed)
        with patch.object(collector.contract, 'performance_contract', return_value={'performance_gate_passed': False}):
            with self.assertRaisesRegex(ValueError, 'prerequisite failed'):
                self.validate()
        self.job['binary'] = dict(self.binary, sha256='f' * 64)
        changed = self.put('performance-job.json', self.job)
        for entry in self.frozen['performance']:
            entry['job'] = changed
        with self.assertRaisesRegex(ValueError, 'job configuration differs'):
            self.validate()

    def args(self, name='batch-output'):
        self.refresh()
        return SimpleNamespace(root=self.root, campaign_root=self.root, freeze=Path(self.freeze_ref['path']),
                               freeze_sha256=self.freeze_ref['sha256'], batch=0, output=self.root / name)

    def simulated_guard(self, command, output, **limits):
        self.assertEqual(limits['max_rss_mib'], 8192)
        self.assertEqual(limits['max_swap_growth_mib'], 0)
        self.assertIn('--worker-spec-sha256', command)
        self.assertEqual(command[2], 'quality')
        worker = collector.parser().parse_args(command[3:])
        self.assertTrue(worker.worker)
        self.assertEqual(worker.freeze_sha256, self.freeze_ref['sha256'])
        self.assertEqual(worker.output, Path(output).parent)
        for index in range(100):
            (Path(output).parent / 'outputs' / f'case-{index:04}.json').write_text('{}\n')
        report = ResourceContract().report()
        Path(output).write_text(json.dumps(report))
        return report

    def test_parent_seals_complete_batch_and_binds_worker_spec(self):
        args = self.args()
        with patch.object(collector, 'model_artifact', return_value=(self.root, self.root)), \
             patch.object(collector, 'run_guarded_combined', side_effect=self.simulated_guard), \
             patch.object(collector.contract, 'audit_batch', return_value={'synthetic': True}) as audit:
            self.assertEqual(collector.collect(args), 0)
        seal = json.loads((args.output / 'seal.json').read_bytes())
        self.assertTrue(seal['collection_complete'])
        self.assertFalse(seal['release_authority'])
        self.assertEqual(len(seal['outputs']), 100)
        audit.assert_called_once()

    def test_failed_guard_does_not_audit_gold_and_remains_nonqualified(self):
        args = self.args()
        def fail(command, output, **limits):
            report = self.simulated_guard(command, output, **limits)
            report['swapout_growth_mib'] = 1
            Path(output).write_text(json.dumps(report))
            return report
        with patch.object(collector, 'model_artifact', return_value=(self.root, self.root)), \
             patch.object(collector, 'run_guarded_combined', side_effect=fail), \
             patch.object(collector.contract, 'audit_batch') as audit:
            self.assertEqual(collector.collect(args), 1)
        audit.assert_not_called()
        seal = json.loads((args.output / 'seal.json').read_bytes())
        self.assertFalse(seal['passed'])
        self.assertFalse(seal['collection_complete'])

    def test_worker_spec_tamper_prevents_model_preflight(self):
        args = self.args()
        args.output.mkdir()
        (args.output / 'spec.json').write_text('{}\n')
        args.worker_spec_sha256 = 'f' * 64
        with patch.object(collector.models, 'headroom') as headroom:
            with self.assertRaisesRegex(ValueError, 'worker spec'):
                collector.worker(args)
        headroom.assert_not_called()


class FreezePublication(unittest.TestCase):
    def setUp(self):
        self.fixture = FrozenCollection()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.output = self.root / 'published-freeze.json'
        self.spec = {'schema': 'antfly.native-whole-head.freeze-input.v2',
                     **{key: self.fixture.frozen[key] for key in ('build', 'configuration', 'configuration_audit',
                                                                 'input_index', 'models', 'performance')}}
        self.spec_ref = self.fixture.put('freeze-spec.json', self.spec)

    def freeze(self):
        return collector.freeze(self.root, self.root, self.spec_ref, self.output)

    def test_publish_exact_validated_bytes_and_fixed_statistical_seed(self):
        with patch.object(collector, 'model_artifact', return_value=(self.root, self.root)) as artifacts:
            ref = self.freeze()
        self.assertEqual(ref['sha256'], collector.sha(self.output))
        value = json.loads(self.output.read_bytes())
        self.assertTrue(value['runtime_frozen'])
        self.assertFalse(value['release_authority'])
        self.assertEqual(artifacts.call_count, 2)
        self.assertEqual(len(value['statistical_seed']), 64)
        self.assertFalse(self.output.with_name(self.output.name + '.pending').exists())

    def test_current_probe_freezes_fusion_flag_without_losing_legacy_bindings(self):
        self.fixture.use_current_execution()
        self.spec.update({key: self.fixture.frozen[key] for key in ('configuration', 'configuration_audit')})
        self.spec_ref = self.fixture.put('freeze-spec.json', self.spec)
        with patch.object(collector, 'model_artifact', return_value=(self.root, self.root)):
            self.freeze()
        value = json.loads(self.output.read_bytes())
        self.assertEqual(value['runtime_environment'], self.fixture.environment)
        self.assertEqual(value['runtime_environment']['ANTFLY_METAL_ATTENTION_MATCH_GATED_FFN_FUSION'], '0')

    def test_failed_prerequisite_retains_pending_bytes_without_publication(self):
        with patch.object(collector.contract, 'performance_contract', return_value={'performance_gate_passed': False}), \
             patch.object(collector, 'model_artifact') as artifacts:
            with self.assertRaisesRegex(ValueError, 'prerequisite failed'):
                self.freeze()
        self.assertFalse(self.output.exists())
        self.assertTrue(self.output.with_name(self.output.name + '.pending').exists())
        artifacts.assert_not_called()

    def test_published_narrow_scope_is_explicit_and_still_rejects_a_failed_required_cell(self):
        self.fixture.use_encoding_scope()
        self.spec.update({key: self.fixture.frozen[key] for key in ('input_index', 'performance', 'qualification_encodings')})
        self.spec_ref = self.fixture.put('freeze-spec.json', self.spec)
        with patch.object(collector, 'model_artifact', return_value=(self.root, self.root)):
            self.freeze()
        value = json.loads(self.output.read_bytes())
        self.assertEqual(value['qualification_encodings'], ['f16', 'fp8'])
        self.assertEqual(len(value['performance']), 8)
        self.assertFalse(value['release_authority'])
        self.output.unlink()
        with patch.object(collector.contract, 'performance_contract', return_value={'performance_gate_passed': False}):
            with self.assertRaisesRegex(ValueError, 'prerequisite failed'):
                self.freeze()

    def test_existing_freeze_and_changed_inputs_are_not_overwritten(self):
        self.output.write_bytes(b'immutable existing freeze')
        with self.assertRaisesRegex(ValueError, 'existing campaign freeze'):
            self.freeze()
        self.assertEqual(self.output.read_bytes(), b'immutable existing freeze')
        self.output.unlink()
        (self.root / self.spec['configuration']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'configuration probe'):
            self.freeze()
        self.assertFalse(self.output.exists())


class SoakCollection(unittest.TestCase):
    def setUp(self):
        self.fixture = FrozenCollection()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        checkpoint = self.root / 'consumed.kv'
        checkpoint.write_bytes(b'consumed synthetic checkpoint, never imported')
        self.checkpoint = {'path': checkpoint.name, 'sha256': collector.sha(checkpoint)}
        report = {'worker': {'requests': [{'kind': 'prepare', 'result': {}},
                                           {'kind': 'prepare', 'result': {'file_sha256': self.checkpoint['sha256']}}]}}
        report_ref = self.fixture.put('soak-performance.json', report)
        for entry in self.fixture.frozen['performance']:
            entry['checkpoint'] = self.checkpoint
            entry['observations'] = report_ref
        self.fixture.refresh()
        self.args = SimpleNamespace(root=self.root, campaign_root=self.root,
                                    freeze=Path(self.fixture.freeze_ref['path']),
                                    freeze_sha256=self.fixture.freeze_ref['sha256'], model='gemma4-e2b', encoding='f16',
                                    duration_seconds=1, diagnostic=True, output=self.root / 'soak-output')
        model = patch.object(collector, 'model_artifact', return_value=(self.root, self.root))
        model.start()
        self.addCleanup(model.stop)

    def guard(self, command, path, **limits):
        self.assertEqual(limits['max_rss_mib'], 8192)
        self.assertEqual(limits['max_swap_growth_mib'], 0)
        self.assertIn('--worker-spec-sha256', command)
        self.assertEqual(command[2], 'soak')
        self.assertIn('--diagnostic', command)
        self.assertEqual(command[command.index('--encoding') + 1], self.args.encoding)
        output = Path(path).parent
        (output / 'report.json').write_text('{}\n')
        (output / 'journal.jsonl').write_text('{}\n')
        receipt = ResourceContract().report()
        Path(path).write_text(json.dumps(receipt))
        return receipt

    def test_diagnostic_collection_seals_raw_evidence_without_release_authority(self):
        with patch.object(collector, 'run_guarded_combined', side_effect=self.guard), \
             patch.object(collector.evidence, 'audit', return_value={'persistent_owner_soak_qualified': False}) as audit:
            self.assertEqual(collector.soak_collect(self.args), 0)
        self.assertFalse(audit.call_args.kwargs['formal'])
        seal = json.loads((self.args.output / 'seal.json').read_bytes())
        self.assertTrue(seal['collection_complete'])
        self.assertFalse(seal['release_authority'])
        self.assertFalse(seal['production_ready'])

    def test_short_unmarked_run_and_changed_checkpoint_prevent_launch(self):
        with patch.object(collector, 'run_guarded_combined') as guard:
            self.args.diagnostic = False
            with self.assertRaisesRegex(ValueError, 'diagnostic scope'):
                collector.soak_collect(self.args)
            self.args.diagnostic = True
            (self.root / self.checkpoint['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                collector.soak_collect(self.args)
        guard.assert_not_called()

    def test_narrow_campaign_cannot_collect_a_rotated4_soak(self):
        self.fixture.use_encoding_scope()
        self.args.freeze_sha256 = self.fixture.freeze_ref['sha256']
        self.args.encoding = 'rotated4'
        with patch.object(collector, 'run_guarded_combined') as guard:
            with self.assertRaisesRegex(ValueError, 'outside frozen qualification scope'):
                collector.soak_collect(self.args)
        guard.assert_not_called()

    def test_checkpoint_must_come_from_the_fresh_public_preparation(self):
        wrong = self.fixture.put('wrong-preparation.json', {'worker': {'requests': [
            {'kind': 'prepare', 'result': {}}, {'kind': 'prepare', 'result': {'file_sha256': 'f' * 64}}]}})
        for entry in self.fixture.frozen['performance']:
            entry['observations'] = wrong
        self.fixture.refresh()
        self.args.freeze_sha256 = self.fixture.freeze_ref['sha256']
        with patch.object(collector, 'run_guarded_combined') as guard:
            with self.assertRaisesRegex(ValueError, 'fresh public preparation'):
                collector.soak_collect(self.args)
        guard.assert_not_called()

    def test_postrun_checkpoint_mutation_remains_failed_and_skips_audit(self):
        def changed(command, path, **limits):
            receipt = self.guard(command, path, **limits)
            (self.root / self.checkpoint['path']).write_bytes(b'changed after execution')
            return receipt
        with patch.object(collector, 'run_guarded_combined', side_effect=changed), \
             patch.object(collector.evidence, 'audit') as audit:
            self.assertEqual(collector.soak_collect(self.args), 1)
        audit.assert_not_called()
        seal = json.loads((self.args.output / 'seal.json').read_bytes())
        self.assertFalse(seal['passed'])
        self.assertIn('checkpoint changed', seal['error'])


class CollectorBinding(unittest.TestCase):
    def test_text_model_selection_allows_a_vision_projector_in_the_same_bundle(self):
        model = {"path": "gemma-4-E2B_q4_0-it.gguf", "sha256": "a" * 64}
        projector = {"path": "mmproj-gemma-4-E2B.gguf", "sha256": "b" * 64}
        self.assertEqual(collector.select_artifact([projector, model], model["path"]), model)

    def test_artifact_selection_rejects_missing_duplicate_and_non_gguf(self):
        model = {"path": "gemma-4-E2B_q4_0-it.gguf", "sha256": "a" * 64}
        for artifacts, path in (([], model["path"]), ([model, model], model["path"]),
                                ([model], "tokenizer.json")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                collector.select_artifact(artifacts, path)

    def test_source_must_match_the_selected_executable_before_model_load(self):
        build = {"head": "a" * 40, "source": {"files": {"a.zig": "b" * 64}}}
        with patch.object(collector, "source_snapshot", return_value=(build["head"], build["source"])), \
                patch.object(collector, "git_provenance", return_value={"identity": "frozen"}):
            self.assertEqual(collector.validate_source_build(Path("/repo"), build), {"identity": "frozen"})
        for head, source in (("c" * 40, build["source"]),
                             (build["head"], {"files": {"a.zig": "d" * 64}})):
            with self.subTest(head=head, source=source), \
                    patch.object(collector, "source_snapshot", return_value=(head, source)), \
                    patch.object(collector, "git_provenance") as provenance:
                with self.assertRaisesRegex(ValueError, "does not cover current source"):
                    collector.validate_source_build(Path("/repo"), build)
                provenance.assert_not_called()

    def test_full_8k_and_32k_trajectories_leave_query_tail(self):
        value = {"schema": "antfly.metal-checkpoint-cycles.input.v1",
                 "model": "google/gemma-4-E2B-it-qat-q4_0-gguf",
                 "expected_prompt_tokens": 8128, "messages": [{"role": "user", "content": "frozen"}],
                 "request": {"encoding": "f16", "cycles": 20, "append_tokens": 512,
                             "answer_tokens": 4}}
        collector.validate_input(value)
        value["expected_prompt_tokens"] = 32096
        value["request"]["append_tokens"] = 4
        collector.validate_input(value)
        value["expected_prompt_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "query-tail reserve"):
            collector.validate_input(value)

    def test_invalid_inputs_fail_before_child_launch(self):
        base = {"schema": "antfly.metal-checkpoint-cycles.input.v1",
                "model": "google/gemma-4-E2B-it-qat-q4_0-gguf",
                "expected_prompt_tokens": 8128, "messages": [{"role": "user", "content": "frozen"}],
                "request": {"encoding": "f16", "cycles": 20, "append_tokens": 512,
                            "answer_tokens": 4}}
        for key, invalid in (("cycles", True), ("cycles", 21), ("cycles", 0),
                             ("append_tokens", 2049), ("answer_tokens", 33),
                             ("encoding", "f32")):
            with self.subTest(key=key):
                value = copy.deepcopy(base); value["request"][key] = invalid
                with self.assertRaises(ValueError): collector.validate_input(value)
        for key, invalid in (("model", "wrong/model"), ("expected_prompt_tokens", True),
                             ("messages", []), ("schema", "old")):
            with self.subTest(key=key):
                value = copy.deepcopy(base); value[key] = invalid
                with self.assertRaises(ValueError): collector.validate_input(value)


def journal(kind):
    return dict(schema="antfly.native-checkpoint-fault.v1", kind=kind,
                diagnostic_only=True, quality_credit=False, release_authority=False,
                observation=dict(injected=True, completed_gpu_batches=2,
                                 engine_error=ERRORS[kind], live_allocations=0, live_bytes=0,
                                 source_preserved=True, source_before=[1]*32, source_after=[1]*32,
                                 revision_before=7, revision_after=7,
                                 allocator_failure_observed=kind == "allocation",
                                 finite_check_observed=kind == "nonfinite",
                                 poisoned_input=kind == "nonfinite",
                                 simulated_completion_error=kind == "gpu_completion"))

class NativeCheckpointFaultEvidenceTests(unittest.TestCase):
    def test_accepts_only_matching_actual_operation(self):
        for kind in ERRORS:
            value = journal(kind)
            validate_fault(value, kind)
            trigger = {"allocation": "allocator_failure_observed", "nonfinite": "finite_check_observed",
                       "gpu_completion": "simulated_completion_error"}[kind]
            value["observation"][trigger] = False
            with self.assertRaises(AssertionError):
                validate_fault(value, kind)

    def test_rejects_missing_work_wrong_error_and_retained_candidate(self):
        for key, bad in [("completed_gpu_batches", 0), ("completed_gpu_batches", True),
                         ("engine_error", "Cancelled"), ("live_allocations", 1),
                         ("live_bytes", 4), ("live_bytes", False)]:
            value = journal("allocation")
            value["observation"][key] = bad
            with self.subTest(key=key, value=bad), self.assertRaises(AssertionError):
                validate_fault(value, "allocation")

    def test_rejects_mutated_or_unbound_source(self):
        for key, bad in [("source_preserved", False), ("source_after", [2]*32),
                         ("source_before", [0]*32), ("revision_after", 8)]:
            value = journal("nonfinite")
            value["observation"][key] = bad
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_fault(value, "nonfinite")

    def test_fault_journal_cannot_grant_quality_or_release_credit(self):
        for key in ["quality_credit", "release_authority"]:
            value = journal("gpu_completion")
            value[key] = True
            with self.assertRaises(AssertionError):
                validate_fault(value, "gpu_completion")

    def test_nonfinite_requires_private_poison_and_finite_detector(self):
        value = journal("nonfinite")
        value["observation"]["poisoned_input"] = False
        with self.assertRaises(AssertionError):
            validate_fault(value, "nonfinite")

    def test_valid_utf8_byte_digest_uses_the_same_exact_bytes(self):
        value = journal("allocation")
        value["observation"]["source_before"] = "a"*32
        value["observation"]["source_after"] = "a"*32
        validate_fault(value, "allocation")
        value["observation"]["source_after"] = "b"*32
        with self.assertRaises(AssertionError):
            validate_fault(value, "allocation")

class PhaseDispatch(unittest.TestCase):
    def test_every_phase_exposes_help_without_loading_a_model(self):
        import contextlib
        import io
        for phase in ('freeze', 'smoke', 'quality', 'performance', 'cycles', 'fault', 'soak'):
            with self.subTest(phase=phase), contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaises(SystemExit) as stopped:
                collector.main([phase, '--help'])
            self.assertEqual(stopped.exception.code, 0)

    def test_unknown_phase_fails_before_execution(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            collector.main(['unrecognized'])
        self.assertEqual(stopped.exception.code, 2)

    def test_current_cli_does_not_import_legacy_release_or_calibration_tools(self):
        import subprocess
        import sys
        script = ('import qualify_native_checkpoint, sys; '
                  'assert "release_attention_matching" not in sys.modules; '
                  'assert "capture_attention_matching" not in sys.modules; '
                  'assert "attention_matching_calibrate" not in sys.modules')
        subprocess.run([sys.executable, '-c', script], cwd=collector.SCRIPTS,
                       check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
