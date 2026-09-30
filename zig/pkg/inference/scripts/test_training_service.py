#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc. SPDX-License-Identifier: Apache-2.0
"""Contracts for Antfarm's node-local training manager; no model downloads."""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training_peer import contained, execute, validate_dataset, check_input
from training_service import Manager, RequestError, request
from training_discovery import parse_txt

SCRIPTS = Path(__file__).resolve().parent


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError('operation did not finish')


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='aft-', dir='/tmp')
        self.root = Path(self.temporary.name)
        for name in ('state', 'tools', 'models', 'data', 'runs'):
            (self.root / name).mkdir()
        self.config = {'state_dir': str(self.root / 'state'), 'toolchain_dir': str(self.root / 'tools'),
                       'output_root': str(self.root / 'runs'), 'models_dir': str(self.root / 'models'),
                       'input_roots': [str(self.root / 'models'), str(self.root / 'data')], 'discovery': False}
        self.manager = Manager(self.config)
        self.manager.peers['mini'] = {'id': 'mini', 'name': 'Mini', 'ssh_destination': 'fixture', 'status': 'connected'}
        self.spec = {'request_id': 'request-0001', 'peer_id': 'mini', 'coordinator': '192.0.2.1:32132', 'kind': 'transport'}

    def tearDown(self):
        self.manager.close()
        self.temporary.cleanup()

    def gliner_spec(self):
        (self.root / 'data/train.jsonl').write_text('{}\n{}\n')
        config = {'version': 1, 'source_dir': str(self.root / 'models'), 'train_file': str(self.root / 'data/train.jsonl'),
                  'output_dir': '/ignored/template/output', 'run': {'mode': 'lora'}, 'peft': {'kind': 'lora', 'mode': 'train', 'rank': 8}}
        path = self.root / 'data/job.json'
        path.write_text(json.dumps(config))
        return {**self.spec, 'family': 'gliner25', 'gliner25_config': str(path), 'kind': 'training'}


class ManagerTests(Fixture):
    def local_spec(self):
        spec = self.gliner_spec()
        del spec['peer_id'], spec['coordinator']
        return {**spec, 'execution_mode': 'local'}

    def form_spec(self):
        spec = self.local_spec()
        Path(spec.pop('gliner25_config')).unlink()
        return {**spec, 'base_model': str(self.root / 'models'), 'train_file': str(self.root / 'data/train.jsonl')}

    def test_form_configuration_generates_the_same_snapshot_for_each_participating_mac(self):
        spec = self.form_spec()
        spec['gliner25_options'] = {'execution': 'native', 'mode': 'dora', 'rank': 4, 'alpha': 8,
            'batch_size': 1, 'accumulation': 2, 'epochs': 3, 'task_lr': 0.0002, 'scheduler': 'cosine',
            'warmup_ratio': 0.2, 'weight_decay': 0, 'seed': 7, 'shuffle': False, 'dropout': 0.1,
            'targets': ['encoder.query', 'encoder.value'], 'max_sequence_tokens': 256,
            'max_text_words': 64, 'max_queries': 16, 'checkpoint_every_microbatches': 5,
            'memory_total_gib': 12, 'memory_host_gib': 5.875, 'memory_backend_gib': 4,
            'dataset_memory_mib': 32, 'source_auxiliary_mib': 128}
        for mode in ('local', 'two_mac'):
            current = {**spec, 'execution_mode': mode}
            if mode == 'two_mac':
                current.update(peer_id='mini', coordinator='192.0.2.1:32132')
            tasks = []
            def run_peer(host, task, cancel):
                if task['action'] == 'prepare':
                    tasks.append(copy.deepcopy(task))
                return {'tcp': True, 'metal': True, 'memory_bytes': 16 * 1024**3}
            with self.subTest(mode=mode), patch.object(self.manager, 'run_peer', side_effect=run_peer):
                validated = self.manager.validate_spec(current)
                record = {'id': 'form-' + mode, 'kind': 'preflight', 'spec': validated}
                command = self.manager.prepare(record, threading.Event())
                self.assertEqual(len(tasks), 1 if mode == 'local' else 2)
                self.assertTrue(all(task['job'] == record['configuration'] for task in tasks))
                self.assertEqual(tasks[0]['inputs'], [spec['base_model'], spec['train_file']])
                self.assertIn(str(self.root / 'runs' / record['id'] / 'job.json'), command)
                job = record['configuration']
                self.assertEqual(job['run']['mode'], 'dora')
                self.assertEqual(job['peft']['kind'], 'dora')
                self.assertEqual(job['run']['batch_size'], 1)
                self.assertEqual(job['run']['accumulation'], 2)
                self.assertEqual(job['run']['seed'], 7)
                self.assertFalse(job['run']['shuffle'])
                self.assertEqual(job['tokenization']['max_sequence_tokens'], 256)
                self.assertEqual(job['checkpoint_every_microbatches'], 5)
                self.assertEqual(job['memory']['host_bytes'], int(5.875 * 1024**3))
                self.assertEqual(job['dataset_limits']['max_host_bytes'], 32 * 1024**2)
                self.assertEqual(job['source_limits']['max_auxiliary_bytes'], 128 * 1024**2)
                self.assertEqual(job['output_dir'], str(self.root / 'runs' / record['id'] / 'output'))
                self.assertFalse((self.root / 'data/job.json').exists())

    def test_form_configuration_requires_data_and_rejects_unsafe_or_invalid_settings(self):
        spec = self.form_spec()
        self.manager.validate_spec(spec)
        with self.assertRaisesRegex(RequestError, 'training dataset'):
            self.manager.validate_spec({key: value for key, value in spec.items() if key != 'train_file'})
        with self.assertRaisesRegex(RequestError, 'not both'):
            self.manager.validate_spec({**spec, 'gliner25_config': '/outside/job.json'})
        for field in ('base_model', 'train_file'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.manager.validate_spec({**spec, field: '/etc/hosts'})
        invalid = [{'batch_size': 9}, {'rank': True}, {'alpha': 0}, {'dropout': 1}, {'targets': []},
                   {'targets': ['encoder', 'encoder']}, {'targets': [{}]}, {'shuffle': 'false'},
                   {'scheduler': 'unknown'}, {'max_sequence_tokens': 0}, {'checkpoint_every_microbatches': 0},
                   {'seed': -1}, {'memory_total_gib': float('nan')}, {'task_lr': float('inf')},
                   {'dataset_memory_mib': 0}, {'source_auxiliary_mib': 1025},
                   {'output_dir': '/tmp/injected'}, {'resume_from': '/tmp/injected'}, {'training_limits': {}}]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(RequestError):
                self.manager.validate_spec({**spec, 'gliner25_options': options})
        identity = 'd' * 32
        self.manager.datasets.records[identity] = {'id': identity, 'family': 'gliner25', 'status': 'ready', 'path': spec['train_file']}
        imported = {key: value for key, value in spec.items() if key != 'train_file'}
        imported['dataset_id'] = identity
        self.manager.validate_spec(imported)
        self.assertEqual(self.manager.gliner_job(imported)['train_file'], spec['train_file'])

    @unittest.skipUnless(os.environ.get('ANTFLY_TRAINING_TEST_BINARY'), 'requires built native CLI')
    def test_form_configuration_passes_native_readiness_without_a_job_file(self):
        spec = self.form_spec()
        (self.root / 'tools/bin').mkdir()
        (self.root / 'tools/bin/antfly-inference').symlink_to(os.environ['ANTFLY_TRAINING_TEST_BINARY'])
        row = (SCRIPTS.parent / 'testdata/gliner25/training_job_small_v1/train.jsonl').read_text().splitlines()[0]
        Path(spec['train_file']).write_text(row + '\n')
        spec['gliner25_options'] = {'execution': 'native', 'batch_size': 1, 'accumulation': 2,
            'scheduler': 'cosine', 'dropout': 0.1, 'targets': ['encoder'], 'max_sequence_tokens': 256,
            'memory_total_gib': 12, 'memory_host_gib': 5.875, 'memory_backend_gib': 4,
            'dataset_memory_mib': 32, 'source_auxiliary_mib': 128}
        record = self.manager.start(spec, True)
        wait_for(lambda: not self.manager.workers, 30)
        result = self.manager.jobs[record['id']]
        self.assertEqual(result['status'], 'complete', result.get('error'))
        self.assertFalse((self.root / 'data/job.json').exists())
        snapshot = json.loads((self.root / 'runs' / record['id'] / 'job.json').read_text())
        self.assertEqual(snapshot, result['configuration'])
        self.assertEqual(snapshot['peft']['rank'], 8)
        self.assertEqual(snapshot['run']['epochs'], 1)
        self.assertEqual(snapshot['run']['accumulation'], 2)

    def test_local_validation_needs_no_peer_and_accepts_odd_datasets(self):
        spec = self.local_spec()
        self.manager.peers.clear()
        self.manager.validate_spec(spec)
        implicit = {key: value for key, value in spec.items() if key != 'execution_mode'}
        self.manager.validate_spec(implicit)
        for count in (1, 3):
            path = self.root / 'data/train.jsonl'
            path.write_text('{}\n' * count)
            validate_dataset(path, distributed=False)
            with self.assertRaisesRegex(ValueError, 'even'):
                validate_dataset(path, distributed=True)
        path.write_text('')
        with self.assertRaisesRegex(ValueError, 'at least one'):
            validate_dataset(path, distributed=False)
        for changes in ({'execution_mode': 'invalid'}, {'execution_mode': 'two_mac'},
                        {'peer_id': 'mini'}, {'coordinator': '192.0.2.1:123'}, {'kind': 'transport'}):
            with self.subTest(changes=changes), self.assertRaises(RequestError):
                self.manager.validate_spec({**spec, **changes}, preflight=True)
        gemma = {**spec, 'family': 'gemma4', 'base_model': str(self.root / 'models'),
                 'adapter': str(self.root / 'models'), 'prepared_inputs': str(path)}
        for count in (1, 3):
            self.manager.validate_spec({**gemma, 'max_examples': count})

    def test_local_readiness_skips_peer_staging_bridge_and_collectives(self):
        spec = self.local_spec()
        self.manager.peers.clear()
        hosts = []
        def peer(host, task, cancel):
            hosts.append((host, task))
            return {'tcp': False, 'metal': True, 'memory_bytes': 16 * 1024**3}
        with patch.object(self.manager, 'run_peer', side_effect=peer), \
                patch.object(self.manager.datasets, 'stage', side_effect=AssertionError('remote staging')), \
                patch.object(self.manager, 'managed_run', return_value={'status': 'preflight_complete'}) as run:
            record = self.manager.start(spec, True)
            wait_for(lambda: not self.manager.workers)
        self.assertEqual(self.manager.jobs[record['id']]['status'], 'complete')
        self.assertEqual([task['action'] for _, task in hosts], ['inspect', 'prepare'])
        self.assertTrue(all(host is None for host, _ in hosts))
        self.assertFalse(hosts[-1][1]['distributed'])
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[2]
        self.assertEqual(command[command.index('--transport') + 1], 'local')
        self.assertNotIn('--remote', command)
        self.assertNotIn('--library', command)
        self.assertIn('--preflight-only', command)

    def test_local_cancel_requires_one_cleanup_receipt(self):
        spec = self.local_spec()
        def run(record, control, command, report):
            control['cancel'].set()
            return {'status': 'failed', 'rank_cleanup': [{'cleanup_complete': True}]}
        with patch.object(self.manager, 'prepare', return_value=['local']), patch.object(self.manager, 'managed_run', side_effect=run):
            record = self.manager.start(spec)
            wait_for(lambda: not self.manager.workers)
        self.assertEqual(self.manager.jobs[record['id']]['status'], 'cancelled')

    def test_launcher_failure_and_timeout_do_not_claim_rank_cleanup(self):
        local = self.local_spec()
        distributed = self.gliner_spec()
        for mode, spec in (('local', local), ('two_mac', distributed)):
            for status in ('running', 'failed', 'timeout'):
                with self.subTest(mode=mode, status=status), \
                        patch.object(self.manager, 'prepare', return_value=['launcher', '--report', 'unused', '--', '/bin/false']), \
                        patch.object(self.manager, 'managed_run', return_value={'status': status, 'rank_logs': ['rank0.log']}):
                    record = self.manager.start({**spec, 'request_id': f'cleanup-{mode}-{status}'})
                    wait_for(lambda: not self.manager.workers)
                    self.assertEqual(self.manager.jobs[record['id']]['status'], 'cleanup_unconfirmed')

    def test_launcher_exception_cannot_reuse_transport_cleanup_receipts(self):
        for mode, spec in (('local', self.local_spec()), ('two_mac', self.gliner_spec())):
            for error in (RuntimeError('launcher report could not be read'), TimeoutError('launcher timed out')):
                reports = []
                if mode == 'two_mac':
                    reports.append({'status': 'complete', 'rank_logs': ['rank0.log', 'rank1.log'],
                                    'rank_cleanup': [{'cleanup_complete': True}, {'cleanup_complete': True}]})
                reports.append(error)
                with self.subTest(mode=mode, error=type(error).__name__), \
                        patch.object(self.manager, 'prepare', return_value=['launcher', '--report', 'unused', '--', '/bin/false']), \
                        patch.object(self.manager, 'managed_run', side_effect=reports):
                    record = self.manager.start({**spec, 'request_id': f'exception-{mode}-{type(error).__name__}'})
                    wait_for(lambda: not self.manager.workers)
                    self.assertEqual(self.manager.jobs[record['id']]['status'], 'cleanup_unconfirmed')

    @unittest.skipUnless(os.environ.get('ANTFLY_TRAINING_TEST_BINARY'), 'requires built native CLI')
    def test_local_native_readiness_needs_no_peer_or_tcp_bridge(self):
        spec = self.local_spec()
        self.manager.peers.clear()
        (self.root / 'tools/bin').mkdir()
        (self.root / 'tools/bin/antfly-inference').symlink_to(os.environ['ANTFLY_TRAINING_TEST_BINARY'])
        row = (SCRIPTS.parent / 'testdata/gliner25/training_job_small_v1/train.jsonl').read_text().splitlines()[0]
        (self.root / 'data/train.jsonl').write_text(row + '\n')
        record = self.manager.start(spec, True)
        wait_for(lambda: not self.manager.workers, 30)
        result = self.manager.jobs[record['id']]
        self.assertEqual(result['status'], 'complete', result.get('error'))
        self.assertEqual(result['report']['transport'], 'local')
        self.assertNotIn('transport_report', result)
        self.assertNotIn('remote_preflight', result['report'])
        self.assertFalse((self.root / 'runs' / record['id'] / 'output').exists())

    def test_local_pause_and_resume_use_local_checkpoint_and_pin_state(self):
        spec = self.local_spec()
        def run(record, control, command, report):
            directory = self.root / 'runs' / record['id'] / 'output'
            directory.mkdir(parents=True)
            (directory / 'latest.safetensors').write_bytes(b'checkpoint fixture')
            (directory / 'result.json').write_text(json.dumps({'identity': {'microbatch_step': 2}, 'state_sha256': '\x01' * 32}))
            return {'status': 'paused', 'rank_cleanup': [{'cleanup_complete': True}]}
        with patch.object(self.manager, 'prepare', return_value=['local']), patch.object(self.manager, 'managed_run', side_effect=run):
            old = self.manager.start(spec)
            wait_for(lambda: not self.manager.workers)
        old = self.manager.jobs[old['id']]
        self.assertEqual(old['status'], 'paused')
        self.assertEqual(old['checkpoint']['state_sha256'], [1] * 32)
        record = {'id': 'resumed', 'kind': 'training', 'spec': {**spec, 'resume_job_id': old['id']}}
        with patch.object(self.manager, 'run_peer', return_value={'metal': True}):
            command = self.manager.prepare(record, threading.Event())
        self.assertEqual(record['configuration']['expected_restore_state_sha256'], [1] * 32)
        self.assertEqual(record['configuration']['resume_from'], str(self.root / 'runs' / old['id'] / 'output/latest.safetensors'))
        self.assertNotIn('--remote', command)
        checkpoint = self.root / 'runs' / old['id'] / 'output/latest.safetensors'
        checkpoint.unlink()
        checkpoint.symlink_to('/etc/hosts')
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.manager.local_checkpoint(old)

    def test_path_escape_and_shell_options_are_rejected(self):
        (self.root / 'data/escape').symlink_to('/etc')
        with self.assertRaises(ValueError):
            contained(str(self.root / 'data/escape/hosts'), self.config['input_roots'])
        for destination in ('-oProxyCommand=evil', 'mini; echo injected', 'mini\nwhoami', 'mini$(id)'):
            with self.subTest(destination=destination), self.assertRaises(RequestError):
                self.manager.request('POST', 'peers', {'ssh_destination': destination})
        with self.assertRaises(RequestError):
            self.manager.validate_spec({**self.spec, 'command': ['/bin/sh']}, True)

    def test_job_validation_rejects_unsupported_training_before_spawn(self):
        spec = self.gliner_spec()
        self.manager.validate_spec(spec)
        (self.root / 'data/train.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'even'):
            validate_dataset(self.root / 'data/train.jsonl')
        (self.root / 'data/train.jsonl').write_text('{}\n{}\n')
        value = json.loads(Path(spec['gliner25_config']).read_text())
        value['execution'] = 'resident_cuda'
        Path(spec['gliner25_config']).write_text(json.dumps(value))
        with self.assertRaisesRegex(RequestError, 'CPU or Metal'):
            self.manager.validate_spec(spec)
        for key in ('base_model', 'adapter', 'prepared_inputs'):
            spec[key] = str(self.root / 'models')
        spec.update(family='gemma4', max_examples=3)
        with self.assertRaisesRegex(RequestError, 'even'):
            self.manager.validate_spec(spec)

    def test_idempotency_conflict_and_single_active_operation(self):
        gate = threading.Event()
        def work(record, control):
            gate.wait(3)
            self.manager.workers.pop(record['id'], None)
        with patch.object(self.manager, 'work', work):
            first = self.manager.start(self.spec, True)
            second = self.manager.start(self.spec, True)
            self.assertEqual(first['id'], second['id'])
            with self.assertRaisesRegex(RequestError, 'another specification'):
                self.manager.start({**self.spec, 'coordinator': '192.0.2.2:32132'}, True)
            with self.assertRaisesRegex(RequestError, 'active'):
                self.manager.start({**self.spec, 'request_id': 'request-0002'}, True)
            gate.set()
            wait_for(lambda: not self.manager.workers)

    def test_second_manager_cannot_reconcile_live_jobs(self):
        record = {'id': 'fixture', 'status': 'running'}
        self.manager.save(record)
        with self.assertRaises(BlockingIOError):
            Manager(self.config)
        self.assertEqual(json.loads((self.root / 'state/jobs/fixture.json').read_text())['status'], 'running')

    def test_disk_failures_do_not_leave_phantom_active_jobs(self):
        with patch.object(self.manager, 'save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.manager.start(self.spec, True)
        self.assertFalse(self.manager.jobs)
        self.assertFalse(self.manager.workers)
        self.manager.state = self.root / 'missing-state-directory'
        job = self.manager.start(self.spec, True)
        wait_for(lambda: not self.manager.workers)
        self.assertEqual(self.manager.jobs[job['id']]['status'], 'failed')

    def test_restart_never_claims_unobserved_cleanup(self):
        self.manager.jobs['fixture'] = {'id': 'fixture', 'status': 'running'}
        self.manager.save(self.manager.jobs['fixture'])
        self.manager.close()
        self.manager = Manager(self.config)
        self.assertEqual(self.manager.jobs['fixture']['status'], 'cleanup_unconfirmed')

    def test_restart_refreshes_peer_inventory_without_persisting_large_caches(self):
        self.manager.peers['mini'].update(inventory={'models': ['example'] * 40000}, checked_at=time.time())
        self.manager.save_peers()
        self.manager.close()
        self.manager = Manager(self.config)
        self.assertEqual(self.manager.peers['mini']['status'], 'configured')
        self.assertNotIn('inventory', self.manager.peers['mini'])
        self.assertNotIn('checked_at', self.manager.peers['mini'])

    def test_log_cursor_is_bounded_and_rejects_path_selection(self):
        self.manager.jobs['job'] = {'id': 'job'}
        directory = self.root / 'state/job'
        directory.mkdir()
        (directory / 'launcher.log').write_bytes(b'a' * 70000)
        _, first = self.manager.request('GET', 'jobs/job/logs', {}, 'rank=launcher')
        self.assertEqual(first['cursor'], 65536)
        _, second = self.manager.request('GET', 'jobs/job/logs', {}, 'rank=launcher&cursor=65536')
        self.assertEqual(len(second['text']), 4464)
        with self.assertRaises(RequestError):
            self.manager.request('GET', 'jobs/job/logs', {}, 'rank=../../secret')

    def test_peer_preparation_is_fresh_and_honors_machine_lock(self):
        task = {'action': 'prepare', 'config': self.config, 'inputs': [],
                'run_dir': str(self.root / 'runs/test'), 'disk_headroom_bytes': 0}
        execute(task)
        with self.assertRaises(FileExistsError):
            execute(task)
        with (self.root / 'runs/.training.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                execute({**task, 'run_dir': str(self.root / 'runs/other')})

    def test_cancel_interrupts_peer_subprocess(self):
        cancel = threading.Event()
        threading.Timer(0.15, cancel.set).start()
        started = time.monotonic()
        with self.assertRaisesRegex(RequestError, 'cancelled'):
            self.manager.capture([sys.executable, '-c', 'import time;time.sleep(30)'], cancel)
        self.assertLess(time.monotonic() - started, 3)

    def test_oversized_response_and_nested_symlink_are_bounded(self):
        with self.assertRaisesRegex(ValueError, 'exceeds limit'):
            self.manager.capture([sys.executable, '-c', 'import sys; sys.stdout.write("x" * 1000000)'])
        (self.root / 'models/secret').symlink_to('/etc/hosts')
        with self.assertRaisesRegex(ValueError, 'outside'):
            check_input(str(self.root / 'models'), self.config['input_roots'])
        self.assertEqual(parse_txt(b'\x03v=1\x0bcontrol=ssh'), {'v': '1', 'control': 'ssh'})
        self.assertEqual(parse_txt(b'\xffshort'), {})

    def test_gliner_overrides_are_snapshotted_and_recovery_cannot_be_injected(self):
        from training_service import gliner_config
        spec = self.gliner_spec()
        original = Path(spec['gliner25_config']).read_bytes()
        spec['gliner25_options'] = {'mode': 'dora', 'rank': 16, 'alpha': 32, 'epochs': 3, 'encoder_lr': 0.001, 'execution': 'resident_metal'}
        self.manager.validate_spec(spec)
        job = gliner_config(spec)
        self.assertEqual(job['run']['mode'], 'dora')
        self.assertEqual(job['peft']['kind'], 'dora')
        self.assertEqual(job['peft']['rank'], 16)
        self.assertEqual(job['run']['epochs'], 3)
        self.assertEqual(Path(spec['gliner25_config']).read_bytes(), original)
        with self.assertRaisesRegex(RequestError, 'Resume endpoint'):
            self.manager.validate_spec({**spec, 'resume_job_id': 'untrusted'})
        spec['gliner25_options']['rank'] = 0
        with self.assertRaises(RequestError):
            self.manager.validate_spec(spec)

    def test_readiness_requires_collectives_before_hash_validation(self):
        spec = self.gliner_spec()
        command = ['python3', 'launcher', '--report', 'unused', '--timeout-seconds', '30', '--check', spec['gliner25_config'], '--preflight-only', '--', '/tool/antfly-inference']
        phases = []
        def run(record, control, argv, report):
            phases.append(argv)
            return {'status': 'complete' if len(phases) == 1 else 'preflight_complete'}
        with patch.object(self.manager, 'prepare', return_value=command), patch.object(self.manager, 'managed_run', run):
            job = self.manager.start(spec, True)
            wait_for(lambda: not self.manager.workers)
        self.assertEqual(self.manager.jobs[job['id']]['status'], 'complete')
        self.assertTrue(phases[0][-1].endswith('/jaccl_smoke.py'))
        self.assertNotIn('--preflight-only', phases[0])
        self.assertIn('--preflight-only', phases[1])

    def test_failed_collectives_never_launch_training(self):
        spec = self.gliner_spec()
        command = ['python3', 'launcher', '--report', 'unused', '--timeout-seconds', '30', '--', '/tool/antfly-inference']
        with patch.object(self.manager, 'prepare', return_value=command), patch.object(self.manager, 'managed_run', return_value={'status': 'failed'}) as run:
            job = self.manager.start(spec)
            wait_for(lambda: not self.manager.workers)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.manager.jobs[job['id']]['status'], 'failed')

    def test_cancel_and_pause_capabilities(self):
        spec = self.gliner_spec()
        record = {'id': 'job', 'spec': spec, 'kind': 'training', 'status': 'running'}
        self.manager.jobs['job'] = record
        self.manager.controls['job'] = {'cancel': threading.Event(), 'pause': threading.Event()}
        self.manager.request('POST', 'jobs/job/pause', {})
        self.assertTrue(self.manager.controls['job']['pause'].is_set())
        record['status'] = 'running'
        record['spec']['family'] = 'gemma4'
        with self.assertRaises(RequestError):
            self.manager.request('POST', 'jobs/job/pause', {})
        self.manager.request('POST', 'jobs/job/cancel', {})
        self.assertEqual(record['status'], 'cancelling')


@unittest.skipUnless(os.environ.get('ANTFLY_TRAINING_TEST_BINARY'),
                     'set ANTFLY_TRAINING_TEST_BINARY to test the built native CLI')
class NativeCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='aft-cli-', dir='/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = str(Path(os.environ['ANTFLY_TRAINING_TEST_BINARY']).resolve(strict=True))

    def run_cli(self, arguments, environment=None):
        return subprocess.run([self.binary, 'finetune', 'train', *arguments],
                              env=environment, capture_output=True, text=True, timeout=15)

    def gliner_job(self):
        job = {'version': 1, 'source_dir': str(self.root / 'missing-model'),
               'train_file': str(self.root / 'missing-data.jsonl'),
               'output_dir': str(self.root / 'output'), 'run': {'mode': 'lora'},
               'peft': {'kind': 'lora', 'mode': 'train', 'rank': 8}}
        path = self.root / 'job.json'
        path.write_text(json.dumps(job))
        return path, job

    def test_gliner_validation_does_not_spawn_a_training_worker(self):
        path, _ = self.gliner_job()
        # Validation must not reconstruct a supervised invocation or enter the
        # worker path, even when an inherited worker environment is incomplete.
        for environment in (None, {**os.environ, 'ANTFLY_TRAINING_COMMAND_WORKER_V1': '1'}):
            result = self.run_cli(['gliner25', str(path), '--validate-only'], environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['event'], 'validated')
            self.assertFalse((self.root / 'output').exists())

    def test_gliner_validation_rejects_inconsistent_adapter_mode(self):
        path, job = self.gliner_job()
        job['peft']['kind'] = 'dora'
        path.write_text(json.dumps(job))
        result = self.run_cli(['gliner25', str(path), '--validate-only'])
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('InvalidTrainingInvocation', result.stderr)
        self.assertFalse((self.root / 'output').exists())

    def test_gemma_validation_checks_distributed_input_contract_without_weights(self):
        example = {'mode': 'instruction', 'prompt_input_ids': [1], 'response_input_ids': [2],
                   'num_prompt_tokens': 1, 'num_response_tokens': 1, 'input_ids': [1, 2],
                   'labels': [-100, 2], 'num_input_tokens': 2, 'num_supervised_tokens': 1}
        summary = {'artifact_family_version': 'gemma4/v1', 'model_dir': str(self.root / 'missing-model'),
                   'max_examples': 2, 'examples_seen': 2, 'max_seq_len': 2,
                   'examples': [example, example]}
        path = self.root / 'prepared.json'
        arguments = ['run', 'gemma4-lora', summary['model_dir'], str(self.root / 'missing-adapter'),
                     str(path), str(self.root / 'output'), '--trainer', 'autodiff',
                     '--max-examples', '2', '--validate-only']
        path.write_text(json.dumps({'summary': summary}))
        result = self.run_cli(arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        for invalid in ({**summary, 'examples': [example]},
                        {**summary, 'examples_with_images': 1},
                        {**summary, 'examples': [{**example, 'num_supervised_tokens': 0}, example]}):
            path.write_text(json.dumps({'summary': invalid}))
            result = self.run_cli(arguments)
            self.assertNotEqual(result.returncode, 0)
        # Local input validation accepts one or an odd number of rows, while
        # still checking every selected row for supervision and text-only input.
        arguments[-1] = '--validate-local-only'
        arguments[arguments.index('--max-examples') + 1] = '0'
        for count in (1, 3):
            path.write_text(json.dumps({'summary': {**summary, 'examples': [example] * count}}))
            result = self.run_cli(arguments)
            self.assertEqual(result.returncode, 0, result.stderr)
        for examples in ([], [example, example, {**example, 'num_supervised_tokens': 0}]):
            path.write_text(json.dumps({'summary': {**summary, 'examples': examples}}))
            self.assertNotEqual(self.run_cli(arguments).returncode, 0)
        path.write_text(json.dumps({'summary': {**summary, 'examples_with_images': 1}}))
        self.assertNotEqual(self.run_cli(arguments).returncode, 0)
        self.assertFalse((self.root / 'output').exists())


class SocketTests(Fixture):
    def test_manager_drives_real_tcp_collectives_and_retains_results(self):
        # SSH is a local relay in this test. Both actual ranks use the native
        # TCP bridge; only the remote host filesystem boundary is simulated.
        toolchain = self.root / 'tools'
        for name in ('bin', 'lib', 'share/antfly/training'):
            (toolchain / name).mkdir(parents=True, exist_ok=True)
        library = toolchain / 'lib/libantfly_tcp.dylib'
        linking = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
        subprocess.run(['c++', '-std=c++20', '-O2', *linking,
                        str(SCRIPTS.parent / 'src/finetune/distributed/tcp_bridge.cpp'), '-o', str(library)], check=True)
        shutil.copy2(SCRIPTS / 'jaccl_smoke.py', toolchain / 'share/antfly/training/jaccl_smoke.py')
        executable = toolchain / 'bin/antfly-inference'
        executable.write_text('#!' + sys.executable + '\nprint(\'{"models": [], "metal": false}\')\n')
        executable.chmod(0o755)
        relay = self.root / 'ssh'
        relay.write_text('#!' + sys.executable + '\nimport os,shlex,sys,json\n'
                         'args=shlex.split(sys.argv[-1])\n'
                         'if "-c" in args and "def supervise(" in args[args.index("-c")+1]:\n'
                         ' config=json.loads(args[-1]); config["lock_file"] += ".remote"; args[-1]=json.dumps(config)\n'
                         'os.execvp(args[0],args)\n')
        relay.chmod(0o755)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        spec = {**self.spec, 'coordinator': f'127.0.0.1:{port}', 'timeout_seconds': 30}
        real_peer = self.manager.run_peer
        def peer(host, task, cancel=None, timeout=60):
            if host and task['action'] == 'prepare':
                return {'run_dir': task['run_dir']}
            return real_peer(host, task, cancel, timeout)
        with patch.dict(os.environ, {'PATH': str(self.root) + ':' + os.environ['PATH']}), patch.object(self.manager, 'run_peer', peer), patch.object(self.manager, 'validate_spec', return_value=spec):
            job = self.manager.start(spec, True)
            wait_for(lambda: not self.manager.workers, 40)
        result = self.manager.public_job(self.manager.jobs[job['id']])
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(result['report']['rank_exit_codes'], [0, 0])
        self.assertTrue(all(row['cleanup_complete'] for row in result['report']['rank_cleanup']))
        for rank in ('0', '1'):
            _, logs = self.manager.request('GET', f'jobs/{job["id"]}/logs', {}, f'rank={rank}')
            self.assertIn('"status": "pass"', logs['text'])
        self.manager.close()
        self.manager = Manager(self.config)
        self.assertEqual(self.manager.jobs[job['id']]['status'], 'complete')

    def test_service_lifetime_pipe_and_private_rpc(self):
        self.manager.close()
        process = subprocess.Popen([sys.executable, str(SCRIPTS / 'training_service.py'), 'serve', json.dumps(self.config)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            value = request(self.config['state_dir'], {'method': 'GET', 'path': 'peers'})
            self.assertEqual(value['status'], 200)
            self.assertEqual((self.root / 'state/control.sock').stat().st_mode & 0o777, 0o600)
            process.stdin.close(); process.stdin = None
            process.wait(timeout=5)
            self.assertEqual(process.returncode, 0, process.stderr.read())
            self.assertFalse((self.root / 'state/control.sock').exists())
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()


if __name__ == '__main__':
    unittest.main()
