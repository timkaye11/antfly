#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc. SPDX-License-Identifier: Apache-2.0
"""Node-local training manager, owned by Antfly through a lifetime pipe.

Only a private Unix socket is exposed. Authentication and authorization happen
in Antfly's public API. All training processes remain children of this service;
EOF from Antfly stops jobs, including jobs still hashing their inputs.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import parse_qs

import find_distributed_checkpoint as recovery
from launch_jaccl_finetune import remote_command
from training_peer import contained, execute as peer_execute
from training_discovery import Discovery
from training_datasets import DatasetStore, DatasetError, tokenizer_sha256

MAX_MESSAGE = 256 * 1024
ACTIVE = {'queued', 'preflight_running', 'running', 'pausing', 'cancelling'}
TERMINAL = {'complete', 'paused', 'failed', 'cancelled', 'interrupted', 'timeout',
            'preflight_failed', 'cleanup_unconfirmed', 'artifact_mismatch', 'artifact_check_failed'}
SCRIPTS = Path(__file__).resolve().parent


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def atomic_json(path, value):
    temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as target:
        json.dump(value, target, sort_keys=True, allow_nan=False)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def read_json(path):
    if path.stat().st_size > MAX_MESSAGE:
        raise ValueError('JSON document exceeds size limit')
    return json.loads(path.read_text())


def bounded_int(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise RequestError(f'{name} must be between {low} and {high}')
    return value


def two_mac(spec):
    # Retain the meaning of persisted jobs and clients predating execution_mode.
    return spec.get('execution_mode', 'two_mac' if spec.get('peer_id') else 'local') == 'two_mac'


def gliner_config(spec):
    if spec.get('gliner25_config'):
        job = read_json(Path(spec['gliner25_config']))
    else:
        # The native job remains a backend-owned snapshot. Form requests can
        # only set the explicit training fields below, never output/recovery
        # paths, executables, or arbitrary native configuration.
        job = {'version': 1, 'source_dir': spec.get('base_model'), 'train_file': spec.get('train_file'),
               'execution': 'resident_metal', 'run': {'mode': 'lora', 'epochs': 1},
               'peft': {'kind': 'lora', 'mode': 'train', 'rank': 8, 'alpha': 16}}
    options = spec.get('gliner25_options', {})
    integers = {'rank': (1, 1024, 'peft'), 'epochs': (1, 10000, 'run'),
                'batch_size': (1, 8, 'run'), 'accumulation': (1, 65536, 'run'),
                'seed': (0, 4294967295, 'run'), 'max_text_words': (1, 8192, 'tokenization'),
                'max_sequence_tokens': (1, 16384, 'tokenization'), 'max_queries': (1, 256, 'tokenization')}
    decimals = {'alpha': (0, 65536, 'peft'), 'encoder_lr': (0, 1, 'run'), 'task_lr': (0, 1, 'run'),
                'weight_decay': (0, 1, 'run'), 'warmup_ratio': (0, 1, 'run'),
                'max_grad_norm': (0, 1000, 'run'), 'dropout': (0, 1, 'peft')}
    memory_fields = {'memory_total_gib': 'combined_bytes', 'memory_host_gib': 'host_bytes',
                     'memory_backend_gib': 'backend_bytes'}
    mib_fields = {'dataset_memory_mib': ('dataset_limits', 'max_host_bytes', 4096),
                  'source_auxiliary_mib': ('source_limits', 'max_auxiliary_bytes', 1024)}
    allowed = {'execution', 'mode', 'scheduler', 'shuffle', 'targets', 'checkpoint_every_microbatches'} | integers.keys() | decimals.keys() | memory_fields.keys() | mib_fields.keys()
    if not isinstance(options, dict) or set(options) - allowed:
        raise RequestError('unsupported GLiNER2.5 options')
    if 'execution' in options:
        if options['execution'] not in ('native', 'resident_metal'):
            raise RequestError('GLiNER2.5 execution requires CPU or Metal')
        job['execution'] = options['execution']
    if 'mode' in options:
        if options['mode'] not in ('lora', 'dora'):
            raise RequestError('GLiNER2.5 requires LoRA or DoRA')
        job.setdefault('run', {})['mode'] = options['mode']
        job.setdefault('peft', {})['kind'] = options['mode']
    for key, (low, high, section) in integers.items():
        if key in options:
            job.setdefault(section, {})[key] = bounded_int(options[key], low, high, key)
    for key, (low, high, section) in decimals.items():
        if key not in options:
            continue
        value = options[key]
        zero_allowed = key in ('weight_decay', 'warmup_ratio', 'max_grad_norm', 'dropout')
        if type(value) not in (int, float) or not low <= value <= high or (not zero_allowed and value == 0) or (key == 'dropout' and value == 1):
            raise RequestError('invalid GLiNER2.5 ' + key)
        job.setdefault(section, {})[key] = value
    for key, field in memory_fields.items():
        if key in options:
            value = options[key]
            if type(value) not in (int, float) or not 0.25 <= value <= 1024:
                raise RequestError('invalid GLiNER2.5 ' + key)
            job.setdefault('memory', {})[field] = int(value * 1024**3)
    for key, (section, field, maximum) in mib_fields.items():
        if key in options:
            job.setdefault(section, {})[field] = bounded_int(options[key], 4, maximum, key) * 1024**2
    if 'scheduler' in options:
        if options['scheduler'] not in ('linear', 'cosine', 'constant'):
            raise RequestError('unsupported GLiNER2.5 scheduler')
        job.setdefault('run', {})['scheduler'] = options['scheduler']
    if 'shuffle' in options:
        if type(options['shuffle']) is not bool:
            raise RequestError('GLiNER2.5 shuffle must be a boolean')
        job.setdefault('run', {})['shuffle'] = options['shuffle']
    if 'targets' in options:
        targets = options['targets']
        if not isinstance(targets, list) or not 1 <= len(targets) <= 64 or any(
                not isinstance(target, str) or not 1 <= len(target) <= 256 or any(ord(c) < 32 or ord(c) == 127 for c in target)
                for target in targets) or len(set(targets)) != len(targets):
            raise RequestError('GLiNER2.5 adapter targets must be distinct nonempty module names')
        job.setdefault('peft', {})['targets'] = targets
    if 'checkpoint_every_microbatches' in options:
        job['checkpoint_every_microbatches'] = bounded_int(options['checkpoint_every_microbatches'], 1, 1000000, 'checkpoint_every_microbatches')
    return job


def validate_config(config):
    for name in ('state_dir', 'toolchain_dir', 'output_root', 'models_dir'):
        if not isinstance(config.get(name), str) or not Path(config[name]).is_absolute():
            raise ValueError(name + ' must be an absolute path')
    if not config.get('input_roots') or any(not Path(p).is_absolute() for p in config['input_roots']):
        raise ValueError('input_roots must contain absolute paths')
    if len(str(Path(config['state_dir']) / 'control.sock').encode()) > 100:
        raise ValueError('state_dir is too long for a macOS Unix socket')
    bounded_int(config.get('ssh_port', 22), 1, 65535, 'ssh_port')
    return config


class Manager:
    def __init__(self, config):
        self.config = validate_config(copy.deepcopy(config))
        managed_root = str(Path(self.config['output_root']) / 'datasets')
        self.config['input_roots'] = list(dict.fromkeys([*self.config['input_roots'], managed_root]))
        self.state = Path(config['state_dir'])
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state, 0o700)
        self.owner_lock = (self.state / 'manager.lock').open('a')
        try:
            fcntl.flock(self.owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.owner_lock.close()
            raise
        self.jobs_dir = self.state / 'jobs'
        self.jobs_dir.mkdir(exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.closing = threading.Event()
        self.workers = {}
        self.controls = {}
        self.refreshing = set()
        self.refresh_slots = threading.BoundedSemaphore(4)
        self.operation = threading.local()
        self.peers_file = self.state / 'peers.json'
        self.peers = read_json(self.peers_file) if self.peers_file.exists() else {}
        self.jobs = {p.stem: read_json(p) for p in self.jobs_dir.glob('*.json')}
        for job in self.jobs.values():
            if job['status'] in ACTIVE:
                job.update(status='cleanup_unconfirmed', error='Manager restarted; verify rank cleanup before retrying.')
                self.save(job)
        self.datasets = DatasetStore(self)
        self.discovery = Discovery(self.state, config.get('ssh_port', 22)) if config.get('discovery', False) else None

    def save(self, job):
        with self.lock:
            job['updated_at'] = time.time()
            atomic_json(self.jobs_dir / (job['id'] + '.json'), job)

    def save_peers(self):
        # Persist enrollment, not a potentially large inventory cache. A fresh
        # manager checks every peer again before claiming it is connected.
        atomic_json(self.peers_file, {key: {
            'id': value['id'], 'name': value['name'],
            'ssh_destination': value['ssh_destination'], 'status': 'configured',
        } for key, value in self.peers.items()})

    def public_job(self, job, include_report=True):
        with self.lock:
            result = copy.deepcopy(job)
        # Recovery sources are selected exclusively by the Resume endpoint.
        result.get('spec', {}).pop('resume_job_id', None)
        result.pop('fingerprint', None)
        if not include_report:
            for key in ('report', 'transport_report', 'configuration'):
                result.pop(key, None)
            return result
        report = self.state / job['id'] / 'report.json'
        if report.exists():
            result['report'] = read_json(report)
        return result

    def run_peer(self, peer, task, cancel=None, timeout=60):
        if peer is None:
            # Local operations use a subprocess too, so cancellation can kill
            # inventory/model inspection instead of blocking the manager.
            command = [sys.executable, str(SCRIPTS / 'training_peer.py'), json.dumps(task)]
        else:
            command = remote_command(peer['ssh_destination'], ['python3', '-c',
                (SCRIPTS / 'training_peer.py').read_text(), json.dumps(task)])
        return json.loads(self.capture(command, cancel, timeout))

    def capture(self, command, cancel=None, timeout=60, stdin=None):
        process = subprocess.Popen(command, stdin=stdin if stdin is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        deadline = min(time.monotonic() + timeout, getattr(self.operation, 'deadline', float('inf')))
        output = {process.stdout: bytearray(), process.stderr: bytearray()}
        try:
            def check():
                if self.closing.is_set() or (cancel and cancel.is_set()):
                    raise RequestError('operation cancelled', 409)
                if time.monotonic() >= deadline:
                    raise TimeoutError('peer operation timed out')
            with selectors.DefaultSelector() as selector:
                for stream in output:
                    selector.register(stream, selectors.EVENT_READ)
                while selector.get_map():
                    check()
                    for key, _ in selector.select(0.2):
                        data = os.read(key.fd, 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        output[key.fileobj].extend(data)
                        if sum(map(len, output.values())) > MAX_MESSAGE:
                            raise ValueError('peer response exceeds limit')
                while process.poll() is None:
                    check()
                    time.sleep(0.05)
            stdout, stderr = (bytes(output[stream]) for stream in (process.stdout, process.stderr))
            if process.returncode:
                raise ValueError(stderr.decode(errors='replace')[-4096:] or 'peer operation failed')
            return stdout
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            process.stdout.close()
            process.stderr.close()

    def refresh_peer(self, peer_id):
        with self.lock:
            if peer_id in self.refreshing:
                return
            if not self.refresh_slots.acquire(blocking=False):
                return
            self.refreshing.add(peer_id)
            self.peers[peer_id]['status'] = 'checking'
            peer = copy.deepcopy(self.peers[peer_id])
        def work():
            try:
                inventory = self.run_peer(peer, {'action': 'inspect', 'config': self.config})
                updates = {'status': 'connected', 'inventory': inventory, 'error': None, 'checked_at': time.time()}
            except Exception as error:
                updates = {'status': 'offline', 'error': str(error)}
            with self.lock:
                try:
                    if peer_id in self.peers:
                        self.peers[peer_id].update(updates)
                        self.save_peers()
                finally:
                    self.refreshing.discard(peer_id)
                    self.refresh_slots.release()
        threading.Thread(target=work, daemon=True).start()

    def validate_spec(self, spec, preflight=False, resume=False):
        if not isinstance(spec, dict):
            raise RequestError('job specification must be an object')
        allowed = {'execution_mode', 'peer_id', 'coordinator', 'family', 'gliner25_config', 'base_model', 'train_file', 'adapter',
                   'prepared_inputs', 'max_examples', 'epochs', 'learning_rate', 'timeout_seconds',
                   'request_id', 'kind', 'resume_job_id', 'gliner25_options',
                   'dataset_id', 'calibration_dataset_id', 'test_dataset_id'}
        if set(spec) - allowed:
            raise RequestError('unknown job fields: ' + ', '.join(sorted(set(spec) - allowed)))
        if spec.get('resume_job_id') and not resume:
            raise RequestError('use the Resume endpoint to select a recovery source')
        if spec.get('kind') not in (None, 'training', 'transport') or (not preflight and spec.get('kind') == 'transport'):
            raise RequestError('transport checks must use the Preflight endpoint')
        if 'execution_mode' in spec and spec['execution_mode'] not in ('local', 'two_mac'):
            raise RequestError('execution_mode must be local or two_mac')
        distributed = two_mac(spec)
        if distributed:
            if spec.get('peer_id') not in self.peers:
                raise RequestError('select a registered SSH peer')
            endpoint = spec.get('coordinator', '')
            if not isinstance(endpoint, str) or not re.fullmatch(r'(?:[a-zA-Z0-9._-]+|\[[a-fA-F0-9:%]+\]):[0-9]+', endpoint):
                raise RequestError('coordinator must be this Mac\'s reachable host:port')
            port = int(endpoint.rsplit(':', 1)[1])
            bounded_int(port, 1, 65535, 'coordinator port')
            if endpoint.rsplit(':', 1)[0] in ('localhost', '127.0.0.1', '0.0.0.0', '[::]', '[::1]'):
                raise RequestError('select an address reachable from the other Mac')
        elif spec.get('peer_id') or spec.get('coordinator') or spec.get('kind') == 'transport':
            raise RequestError('peer, coordinator and transport checks require two_mac execution')
        bounded_int(spec.get('timeout_seconds', 7200), 1, 604800, 'timeout_seconds')
        if preflight and spec.get('kind') == 'transport':
            return copy.deepcopy(spec)
        if spec.get('family') not in ('gliner25', 'gemma4'):
            raise RequestError('family must be gliner25 or gemma4')
        roots = self.config['input_roots']
        if spec['family'] == 'gliner25':
            if spec.get('gliner25_config'):
                if spec.get('base_model') or spec.get('train_file'):
                    raise RequestError('choose form settings or a GLiNER2.5 job configuration, not both')
                contained(spec['gliner25_config'], roots)
            else:
                contained(spec.get('base_model'), roots)
            job = self.gliner_job(spec)
            if not job.get('train_file'):
                raise RequestError('select a prepared training dataset or supply a training JSONL path')
            if job.get('version') != 1 or job.get('run', {}).get('mode') not in ('lora', 'dora'):
                raise RequestError('GLiNER2.5 requires a version-1 LoRA/DoRA job')
            if job.get('execution', 'native') not in ('native', 'resident_metal'):
                raise RequestError('GLiNER2.5 execution requires CPU or Metal')
            if not job.get('peft'):
                raise RequestError('GLiNER2.5 requires matching PEFT configuration')
            if job.get('resume_from') or job.get('expected_restore_state_sha256'):
                raise RequestError('use the Resume action to select a checkpoint')
            for key in ('source_dir', 'train_file', 'calibration_file', 'test_file'):
                if job.get(key):
                    contained(job[key], roots)
        else:
            if 'calibration_dataset_id' in spec or 'test_dataset_id' in spec:
                raise RequestError('calibration and test datasets are supported for GLiNER2.5 only')
            for key in ('base_model', 'adapter'):
                contained(spec.get(key), roots)
            contained(self.prepared_inputs(spec), roots)
            count = bounded_int(spec.get('max_examples', 2), 2 if distributed else 1, 1000000, 'max_examples')
            if distributed and count % 2:
                raise RequestError('Gemma4 max_examples must be even')
            bounded_int(spec.get('epochs', 1), 1, 10000, 'epochs')
            rate = spec.get('learning_rate', 0.0001)
            if type(rate) not in (int, float) or not 0 < rate <= 1:
                raise RequestError('learning_rate must be positive and at most 1')
        bounded_int(spec.get('timeout_seconds', 7200), 1, 604800, 'timeout_seconds')
        return copy.deepcopy(spec)

    def gliner_job(self, spec):
        job = gliner_config(spec)
        for key, field in (('dataset_id', 'train_file'), ('calibration_dataset_id', 'calibration_file'), ('test_dataset_id', 'test_file')):
            if key not in spec:
                continue
            if spec[key] is None and key != 'dataset_id':
                job[field] = None
            else:
                job[field] = self.datasets.ready(spec[key], 'gliner25')['path']
        return job

    def prepared_inputs(self, spec):
        if spec.get('dataset_id'):
            dataset = self.datasets.ready(spec['dataset_id'], 'gemma4')
            if Path(dataset['spec']['model_dir']).resolve() != Path(spec['base_model']).resolve():
                raise RequestError('prepare the Gemma dataset with the selected base model tokenizer')
            return dataset['path']
        return spec.get('prepared_inputs')

    def local_checkpoint(self, record):
        output = Path(self.config['output_root']) / record['id'] / 'output'
        checkpoint = contained(str(output / 'latest.safetensors'), [self.config['output_root']])
        if not checkpoint.is_file():
            raise RequestError('no local checkpoint is available for this run')
        value = {'checkpoint': 'latest.safetensors'}
        result_path = output / 'result.json'
        if result_path.exists():
            result = read_json(contained(str(result_path), [self.config['output_root']]))
            digest = result.get('state_sha256')
            # Zig encodes a byte array as a JSON string when it is valid UTF-8,
            # and as a numeric array otherwise. Normalize both representations.
            if isinstance(digest, str):
                digest = list(digest.encode('utf-8'))
            if not isinstance(digest, list) or len(digest) != 32 or any(type(v) is not int or not 0 <= v <= 255 for v in digest):
                raise RequestError('invalid local checkpoint state fingerprint')
            value.update(identity=result['identity'], state_sha256=digest)
        return value

    def start(self, spec, preflight=False, resume=False):
        spec = self.validate_spec(spec, preflight, resume)
        request_id = spec.get('request_id')
        if not isinstance(request_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,80}', request_id):
            raise RequestError('request_id must be a stable 8-80 character idempotency key')
        fingerprint = hashlib.sha256(json.dumps([preflight, spec], sort_keys=True).encode()).hexdigest()
        with self.lock:
            for old in self.jobs.values():
                if old['request_id'] == request_id:
                    if old['fingerprint'] != fingerprint:
                        raise RequestError('request_id was already used for another specification', 409)
                    return self.public_job(old)
            if self.workers:
                raise RequestError('this coordinator already has an active training operation', 409)
            if self.closing.is_set():
                raise RequestError('training manager is shutting down', 503)
            job_id = uuid.uuid4().hex
            record = {'id': job_id, 'request_id': request_id, 'fingerprint': fingerprint, 'spec': spec,
                      'kind': 'preflight' if preflight else 'training', 'status': 'queued', 'created_at': time.time()}
            self.jobs[job_id] = record
            try:
                self.save(record)
            except BaseException:
                del self.jobs[job_id]
                raise
            control = {'cancel': threading.Event(), 'pause': threading.Event()}
            self.controls[job_id] = control
            worker = threading.Thread(target=self.work, args=(record, control), daemon=True)
            self.workers[job_id] = worker
            worker.start()
            return self.public_job(record)

    def prepare(self, record, cancel):
        spec = record['spec']
        distributed = two_mac(spec)
        peer = self.peers[spec['peer_id']] if distributed else None
        toolchain = Path(self.config['toolchain_dir'])
        executable = str(toolchain / 'bin/antfly-inference')
        library = str(toolchain / 'lib/libantfly_tcp.dylib')
        shared_scripts = toolchain / 'share/antfly/training'
        run_dir = Path(self.config['output_root']) / record['id']
        output = run_dir / 'output'
        checks = []
        job = None
        if record['kind'] == 'preflight' and spec.get('kind') == 'transport':
            command = [str(shared_scripts / 'jaccl_smoke.py')]
            adapter = None
        elif spec['family'] == 'gliner25':
            job = self.gliner_job(spec)
            # Both form and template requests produce the same native snapshot.
            # An optional template remains unchanged on disk.
            checks = [job['source_dir'], job['train_file']]
            if spec.get('gliner25_config'):
                checks.insert(0, spec['gliner25_config'])
            checks.extend(job[key] for key in ('calibration_file', 'test_file') if job.get(key))
            job['output_dir'] = str(output)
            command = [executable, 'finetune', 'train', 'gliner25', str(run_dir / 'job.json')]
            adapter = str(output / 'model/adapter_model.safetensors')
        else:
            checks = [spec['base_model'], spec['adapter'], self.prepared_inputs(spec)]
            command = [executable, 'finetune', 'train', 'run', 'gemma4-lora', *checks, str(output),
                       '--trainer', 'autodiff', '--max-examples', str(spec.get('max_examples', 2)),
                       '--epochs', str(spec.get('epochs', 1)), '--grad-accum', '1',
                       '--learning-rate', str(spec.get('learning_rate', 0.0001))]
            adapter = str(output / 'adapter_model.safetensors')
        validation_flag = '--validate-local-only' if not distributed and spec.get('family') == 'gemma4' else '--validate-only'
        task = {'action': 'prepare', 'config': self.config, 'inputs': checks, 'run_dir': str(run_dir),
                'distributed': distributed, 'job': job, 'validation_command': command + [validation_flag] if adapter else None,
                'disk_headroom_bytes': max(256 * 1024 * 1024, (job or {}).get('disk_headroom_bytes', 0))}
        if spec.get('resume_job_id'):
            old = self.jobs.get(spec['resume_job_id'])
            if not job or not old or two_mac(old['spec']) != distributed or old['spec'].get('peer_id') != spec.get('peer_id'):
                raise RequestError('invalid GLiNER2.5 recovery source')
            old_dir = str(Path(self.config['output_root']) / old['id'] / 'output')
            if distributed:
                local = json.loads(self.capture([sys.executable, '-c', recovery.SCAN, old_dir], cancel))
                remote = json.loads(self.capture(remote_command(peer['ssh_destination'], ['python3', '-c', recovery.SCAN, old_dir]), cancel))
                first, _ = recovery.newest_common(local, remote)
                record['checkpoint'] = {'identity': first['identity'], 'shared_state_sha256': first['shared_state_sha256']}
            else:
                first = self.local_checkpoint(old)
                record['checkpoint'] = first
                if first.get('state_sha256'):
                    job['expected_restore_state_sha256'] = first['state_sha256']
            # The same generation path exists on each host, but its contents
            # and expected local state hashes are rank-specific by design.
            job['resume_from'] = str(Path(old_dir) / first['checkpoint'])
        for host in ((None, peer) if distributed else (None,)):
            inventory = self.run_peer(host, {'action': 'inspect', 'config': self.config}, cancel)
            if distributed and not inventory.get('tcp'):
                raise RequestError('TCP bridge is missing on a participating Mac')
            if job and job.get('execution') == 'resident_metal' and not inventory.get('metal'):
                raise RequestError('Metal training is unavailable on a participating Mac')
            combined = (job or {}).get('memory', {}).get('combined_bytes', 12 * 1024**3) if job else 0
            if combined and inventory.get('memory_bytes', 0) and combined > inventory['memory_bytes']:
                raise RequestError('configured training memory exceeds a participating Mac\'s physical memory')
        dataset_ids = set(spec[key] for key in ('dataset_id', 'calibration_dataset_id', 'test_dataset_id') if spec.get(key)) if spec.get('kind') != 'transport' else set()
        for identity in dataset_ids:
            dataset = self.datasets.ready(identity, spec['family'])
            if dataset['family'] == 'gemma4' and tokenizer_sha256(spec['base_model']) != dataset['tokenizer_sha256']:
                raise RequestError('the Gemma tokenizer changed after dataset preparation; import the dataset again')
            if distributed:
                self.datasets.stage(dataset, peer, cancel)
            else:
                self.datasets.verify(dataset)
        self.run_peer(None, task, cancel)
        if distributed:
            self.run_peer(peer, task, cancel)
        if job and (not spec.get('resume_job_id') or not distributed):
            checks.append(str(run_dir / 'job.json'))
        record['output_dir'] = str(output)
        if job:
            record['configuration'] = job
        self.save(record)
        base = [sys.executable, str(SCRIPTS / 'launch_jaccl_finetune.py'), '--transport', 'tcp' if distributed else 'local',
                '--report', str(self.state / record['id'] / 'report.json'), '--managed',
                '--lock-file', str(Path(self.config['output_root']) / '.training.lock'),
                '--timeout-seconds', str(spec.get('timeout_seconds', 7200))]
        if distributed:
            base.extend(['--remote', peer['ssh_destination'], '--coordinator', spec['coordinator'], '--library', library])
        for path in checks:
            base.extend(['--check', path])
        if record['kind'] == 'preflight' and spec.get('kind') != 'transport':
            base.append('--preflight-only')
        elif adapter:
            base.extend(['--compare-adapter', adapter])
        return base + ['--', *command]

    def managed_run(self, record, control, command, report_path):
        remaining = int(self.operation.deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError('training operation timed out')
        command[command.index('--timeout-seconds') + 1] = str(remaining)
        process = None
        try:
            with (self.state / record['id'] / 'launcher.log').open('ab') as log:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log, start_new_session=True)
                paused = False
                stop_at = None
                while process.poll() is None:
                    stop = self.closing.is_set() or control['cancel'].is_set() or time.monotonic() >= self.operation.deadline
                    if stop and stop_at is None:
                        stop_at = time.monotonic()
                        os.killpg(process.pid, signal.SIGTERM)
                    if stop_at is not None and time.monotonic() - stop_at > 70:
                        os.killpg(process.pid, signal.SIGKILL)
                    message = b'H'
                    if control['pause'].is_set() and not paused:
                        message += b'P'
                        paused = True
                    try:
                        process.stdin.write(message)
                        process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
                    time.sleep(0.2)
            return read_json(report_path) if report_path.exists() else {'status': 'cleanup_unconfirmed'}
        finally:
            if process is not None:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=70)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                process.stdin.close()

    def work(self, record, control):
        job_dir = self.state / record['id']
        self.operation.deadline = time.monotonic() + record['spec'].get('timeout_seconds', 7200)
        launched = False
        report = {}
        distributed = two_mac(record['spec'])
        world_size = 2 if distributed else 1
        try:
            job_dir.mkdir(mode=0o700)
            record['status'] = 'preflight_running'
            self.save(record)
            command = self.prepare(record, control['cancel'])
            if distributed and record['spec'].get('kind') != 'transport':
                smoke = command[:command.index('--')]
                for flag in ('--check', '--compare-adapter'):
                    while flag in smoke:
                        index = smoke.index(flag)
                        del smoke[index:index + 2]
                if '--preflight-only' in smoke:
                    smoke.remove('--preflight-only')
                smoke_path = job_dir / 'transport.json'
                smoke[smoke.index('--report') + 1] = str(smoke_path)
                smoke.extend(['--', str(Path(self.config['toolchain_dir']) / 'share/antfly/training/jaccl_smoke.py')])
                launched = True
                report = self.managed_run(record, control, smoke, smoke_path)
                record['transport_report'] = report
                if report.get('status') != 'complete':
                    raise RequestError('TCP collective preflight failed; see the transport report')
            if control['cancel'].is_set() or self.closing.is_set():
                raise RequestError('operation cancelled')
            record['status'] = 'running' if record['kind'] == 'training' else 'preflight_running'
            self.save(record)
            report_file = job_dir / 'report.json'
            # Transport cleanup cannot confirm cleanup of a later training
            # launch if that launcher fails before returning its report.
            report = {}
            launched = True
            report = self.managed_run(record, control, command, report_file)
            status = report.get('status', 'failed')
            receipts = report.get('rank_cleanup', [])
            clean = len(receipts) == world_size and all(r and r.get('cleanup_complete') for r in receipts)
            uncertain_cleanup = bool(report.get('rank_logs')) and not clean
            if status == 'preflight_complete':
                status = 'complete'
            if control['cancel'].is_set() or self.closing.is_set():
                no_ranks = report.get('status') in ('preflight_failed', 'preflight_complete') and not report.get('rank_logs')
                status = 'cancelled' if no_ranks or clean else 'cleanup_unconfirmed'
            elif time.monotonic() >= self.operation.deadline:
                status = 'timeout'
            if uncertain_cleanup:
                status = 'cleanup_unconfirmed'
            if status == 'paused' and distributed:
                old_dir = record['output_dir']
                local = json.loads(self.capture([sys.executable, '-c', recovery.SCAN, old_dir]))
                peer = self.peers[record['spec']['peer_id']]
                remote = json.loads(self.capture(remote_command(peer['ssh_destination'], ['python3', '-c', recovery.SCAN, old_dir])))
                first, _ = recovery.newest_common(local, remote)
                record['checkpoint'] = first
            elif status == 'paused':
                record['checkpoint'] = self.local_checkpoint(record)
            record['status'] = status if status in TERMINAL else 'failed'
            record['report'] = report
        except Exception as error:
            stopped = control['cancel'].is_set() or self.closing.is_set()
            receipts = report.get('rank_cleanup', [])
            clean = not launched or (len(receipts) == world_size and all(r and r.get('cleanup_complete') for r in receipts))
            uncertain_cleanup = launched and not clean and (not report or bool(report.get('rank_logs')))
            if uncertain_cleanup or (stopped and not clean):
                status = 'cleanup_unconfirmed'
            elif stopped:
                status = 'cancelled'
            else:
                status = 'timeout' if isinstance(error, TimeoutError) else 'failed'
            record.update(status=status, error=str(error))
        finally:
            with self.lock:
                try:
                    self.save(record)
                finally:
                    self.workers.pop(record['id'], None)
                    self.controls.pop(record['id'], None)

    def request(self, method, path, body, query=''):
        parts = path.strip('/').split('/')
        if parts == ['datasets', 'huggingface'] and method == 'POST':
            result = self.capture([sys.executable, str(SCRIPTS / 'training_datasets.py'), 'huggingface', json.dumps(body)], timeout=12)
            return 200, json.loads(result)
        if parts[0] == 'datasets':
            return self.datasets.request(method, parts, body)
        if parts[0] == 'peers':
            if method == 'GET' and len(parts) == 1:
                with self.lock:
                    peers = copy.deepcopy(list(self.peers.values()))
                for peer in peers:
                    if peer.get('checked_at', 0) < time.time() - 60 and peer['id'] not in self.refreshing:
                        self.refresh_peer(peer['id'])
                return 200, {'peers': peers, 'nearby': self.discovery.snapshot() if self.discovery else [],
                             'discovery_error': self.discovery.error if self.discovery else None,
                             'config': {key: self.config[key] for key in ('models_dir', 'output_root', 'input_roots')},
                             'capabilities': {'families': ['gliner25', 'gemma4'], 'transport': 'tcp', 'execution_modes': ['local', 'two_mac']}}
            if method == 'POST' and len(parts) == 1:
                destination = body.get('ssh_destination', '')
                if not isinstance(destination, str) or not re.fullmatch(r'[a-zA-Z0-9_][a-zA-Z0-9_.@:-]{0,252}', destination):
                    raise RequestError('use an SSH alias or user@hostname; SSH options are not accepted')
                peer_id = hashlib.sha256(destination.encode()).hexdigest()[:24]
                name = body.get('name', destination)
                if not isinstance(name, str) or len(name) > 128:
                    raise RequestError('peer name is too long')
                with self.lock:
                    if len(self.peers) >= 32 and peer_id not in self.peers:
                        raise RequestError('at most 32 configured peers are supported')
                    self.peers[peer_id] = {'id': peer_id, 'name': name, 'ssh_destination': destination, 'status': 'configured'}
                    self.save_peers()
                self.refresh_peer(peer_id)
                return 202, self.peers[peer_id]
            if len(parts) >= 2 and parts[1] in self.peers:
                if method == 'POST' and parts[2:] == ['refresh']:
                    self.refresh_peer(parts[1])
                    return 202, self.peers[parts[1]]
                if method == 'DELETE' and len(parts) == 2:
                    with self.lock:
                        if any(j['spec'].get('peer_id') == parts[1] and j['status'] in ACTIVE for j in self.jobs.values()):
                            raise RequestError('peer is participating in an active job', 409)
                        del self.peers[parts[1]]
                        self.save_peers()
                    return 200, {'removed': True}
        if parts[0] in ('jobs', 'preflights'):
            if method == 'POST' and len(parts) == 1:
                return 202, self.start(body, parts[0] == 'preflights')
            if method == 'GET' and len(parts) == 1:
                with self.lock:
                    jobs = sorted(self.jobs.values(), key=lambda j: j['created_at'], reverse=True)[:200]
                    return 200, {'jobs': [self.public_job(job, include_report=False) for job in jobs]}
            if len(parts) > 1:
                with self.lock:
                    record = self.jobs.get(parts[1])
                if record is None:
                    raise RequestError('job not found', 404)
                if method == 'GET' and len(parts) == 2:
                    return 200, self.public_job(record)
                if method == 'GET' and parts[2:] == ['logs']:
                    values = parse_qs(query)
                    rank = values.get('rank', ['launcher'])[0]
                    if rank not in ('launcher', '0', '1'):
                        raise RequestError('rank must be 0, 1, or launcher')
                    cursor = int(values.get('cursor', ['0'])[0])
                    if cursor < 0:
                        raise RequestError('cursor must be nonnegative')
                    path = self.state / record['id'] / ('launcher.log' if rank == 'launcher' else f'report-logs/rank{rank}.log')
                    if not path.exists():
                        return 200, {'text': '', 'cursor': 0}
                    with path.open('rb') as source:
                        source.seek(min(cursor, path.stat().st_size))
                        data = source.read(64 * 1024)
                        return 200, {'text': data.decode(errors='replace'), 'cursor': source.tell()}
                if method == 'POST' and parts[2:] == ['resume']:
                    if record['spec'].get('family') != 'gliner25' or record['status'] not in TERMINAL:
                        raise RequestError('resume requires an ended GLiNER2.5 job', 409)
                    spec = {**record['spec'], 'resume_job_id': record['id'], 'request_id': body.get('request_id')}
                    return 202, self.start(spec, resume=True)
                if method == 'POST' and parts[2:] in (['cancel'], ['pause']):
                    action = parts[2]
                    with self.lock:
                        control = self.controls.get(record['id'])
                        if control is None:
                            raise RequestError('job is no longer active', 409)
                        if action == 'pause' and (record['spec'].get('family') != 'gliner25' or record['kind'] != 'training' or record['status'] != 'running'):
                            raise RequestError('only running GLiNER2.5 training can pause', 409)
                        control[action].set()
                        record['status'] = 'pausing' if action == 'pause' else 'cancelling'
                        self.save(record)
                    return 202, self.public_job(record)
        raise RequestError('training resource not found', 404)

    def close(self):
        self.closing.set()
        with self.lock:
            workers = list(self.workers.values())
        for worker in workers:
            worker.join(timeout=75)
        if self.discovery:
            self.discovery.close()
        self.owner_lock.close()


def serve(config):
    manager = Manager(config)
    path = manager.state / 'control.sock'
    path.unlink(missing_ok=True)
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(15)
            try:
                line = self.rfile.readline(MAX_MESSAGE + 1)
                if len(line) > MAX_MESSAGE:
                    raise RequestError('training request too large', 413)
                value = json.loads(line)
                status, body = manager.request(value['method'], value['path'], value.get('body', {}), value.get('query', ''))
            except (RequestError, DatasetError) as error:
                status, body = error.status, {'error': str(error)}
            except (ValueError, KeyError, TypeError, OSError) as error:
                status, body = 400, {'error': str(error)}
            try:
                self.wfile.write(json.dumps({'status': status, 'body': body}, allow_nan=False).encode() + b'\n')
            except (BrokenPipeError, ConnectionError):
                pass
    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True
        request_queue_size = 8
        slots = threading.BoundedSemaphore(8)

        def process_request(self, request, address):
            if not self.slots.acquire(blocking=False):
                request.sendall(b'{"status":503,"body":{"error":"training manager busy"}}\n')
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, address)
            except BaseException:
                self.slots.release()
                raise

        def process_request_thread(self, request, address):
            try:
                super().process_request_thread(request, address)
            finally:
                self.slots.release()
    server = Server(str(path), Handler)
    os.chmod(path, 0o600)
    def stop():
        manager.closing.set()
        server.shutdown()
    def lifetime():
        try:
            while sys.stdin.buffer.read(1):
                pass
        finally:
            stop()
    threading.Thread(target=lifetime, daemon=True).start()
    def interrupted(signum, frame):
        threading.Thread(target=stop, daemon=True).start()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        manager.close()
        path.unlink(missing_ok=True)


def request(state_dir, value):
    deadline = time.monotonic() + 5
    while True:
        connection = socket.socket(socket.AF_UNIX)
        connection.settimeout(15)
        try:
            connection.connect(str(Path(state_dir) / 'control.sock'))
            break
        except (FileNotFoundError, ConnectionRefusedError):
            connection.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    with connection:
        connection.sendall(json.dumps(value).encode() + b'\n')
        with connection.makefile('rb') as source:
            response = source.readline(4 * 1024 * 1024 + 1)
        if len(response) > 4 * 1024 * 1024:
            raise ValueError('training response too large')
        return json.loads(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['serve', 'request'])
    parser.add_argument('config_or_state')
    parser.add_argument('request_json', nargs='?')
    args = parser.parse_args()
    if args.action == 'serve':
        serve(json.loads(args.config_or_state))
    else:
        print(json.dumps(request(args.config_or_state, json.loads(args.request_json))))
