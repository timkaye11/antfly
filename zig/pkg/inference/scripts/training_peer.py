#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc. SPDX-License-Identifier: Apache-2.0
"""Bounded SSH peer operations. The coordinator sends this source over SSH."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import uuid


def contained(value, roots, exists=True):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError('training paths must be absolute')
    path = Path(value).resolve(strict=exists)
    if not any(path.is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError('path is outside configured training roots: ' + value)
    return path


def check_input(value, roots):
    path = contained(value, roots)
    if path.is_dir():
        # Do not let a directory hash or native loader escape the allowlist
        # through an embedded symlink in a pre-staged model package.
        for child in path.rglob('*'):
            if child.is_symlink():
                contained(str(child), roots)
    return path


def validate_dataset(path, distributed=True):
    count = 0
    with Path(path).open('rb') as source:
        while True:
            line = source.readline(1024 * 1024 + 1)
            if not line:
                break
            if len(line) > 1024 * 1024:
                raise ValueError('training example exceeds 1 MiB')
            if line.strip():
                count += 1
    if not count:
        raise ValueError('training requires at least one example')
    if distributed and (count < 2 or count % 2):
        raise ValueError('GLiNER2.5 requires an even number of training examples')


def inspect(config):
    toolchain = Path(config['toolchain_dir'])
    result = subprocess.run([str(toolchain / 'bin/antfly-inference'), 'peer-info',
                             config['models_dir']], capture_output=True, text=True,
                            check=True, timeout=20)
    value = json.loads(result.stdout)
    value.update(hostname=platform.node(), os=platform.platform(), architecture=platform.machine())
    if sys.platform == 'darwin':
        def sysctl(key):
            return subprocess.check_output(['/usr/sbin/sysctl', '-n', key], text=True, timeout=3).strip()
        value['memory_bytes'] = int(sysctl('hw.memsize'))
        value['chip'] = sysctl('machdep.cpu.brand_string')
    value['tcp'] = (toolchain / 'lib/libantfly_tcp.dylib').is_file()
    value['jaccl'] = (toolchain / 'lib/libantfly_jaccl.dylib').is_file()
    value['disk_free_bytes'] = shutil.disk_usage(config['output_root']).free
    return value


def execute(task):
    config = task['config']
    roots = config['input_roots']
    output_root = Path(config['output_root']).resolve(strict=True)
    if task['action'] == 'inspect':
        return inspect(config)
    if task['action'] == 'stage_dataset':
        identity, name, size, expected = (task.get(key) for key in ('dataset_id', 'filename', 'size_bytes', 'sha256'))
        if not isinstance(identity, str) or not re.fullmatch(r'[0-9a-f]{32}', identity) or name not in ('dataset.jsonl', 'prepared.json'):
            raise ValueError('invalid managed dataset destination')
        if type(size) is not int or not 0 < size <= 64 * 1024**2 or not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('invalid managed dataset size or digest')
        directory = contained(str(output_root / 'datasets' / identity), [str(output_root)], exists=False)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = contained(str(directory / name), [str(directory)], exists=False)
        if target.exists():
            digest = hashlib.sha256()
            with target.open('rb') as existing:
                for block in iter(lambda: existing.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected or target.stat().st_size != size:
                raise ValueError('a different dataset already exists at this destination')
            return {'sha256': expected}
        if shutil.disk_usage(output_root).free < size + 256 * 1024**2:
            raise ValueError('insufficient disk headroom for dataset staging')
        temporary = directory / (name + '.' + uuid.uuid4().hex + '.upload')
        try:
            remaining, digest = size, hashlib.sha256()
            with temporary.open('xb') as output:
                while remaining:
                    data = sys.stdin.buffer.read(min(65536, remaining))
                    if not data:
                        raise ValueError('dataset transfer ended before the declared size')
                    output.write(data)
                    digest.update(data)
                    remaining -= len(data)
                if sys.stdin.buffer.read(1) or digest.hexdigest() != expected:
                    raise ValueError('dataset transfer size or hash mismatch')
                output.flush()
                os.fsync(output.fileno())
            # Publish without overwriting a file another coordinator created.
            os.link(temporary, target)
            return {'sha256': expected}
        finally:
            temporary.unlink(missing_ok=True)
    if task['action'] == 'prepare':
        for path in task['inputs']:
            check_input(path, roots)
        if task.get('job'):
            validate_dataset(task['job']['train_file'], task.get('distributed', True))
        with (output_root / '.training.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(output_root).free < task['disk_headroom_bytes']:
            raise ValueError('insufficient disk headroom for training')
        run_dir = contained(task['run_dir'], [str(output_root)], exists=False)
        run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        try:
            job = task.get('job')
            if job is not None:
                if Path(job['output_dir']).resolve() != run_dir / 'output':
                    raise ValueError('unexpected output directory')
                for key in ('source_dir', 'train_file', 'calibration_file', 'test_file'):
                    if job.get(key):
                        contained(job[key], roots)
                if job.get('resume_from'):
                    contained(job['resume_from'], [str(output_root)])
                with (run_dir / 'job.json').open('x') as target:
                    json.dump(job, target, sort_keys=True)
        except BaseException:
            shutil.rmtree(run_dir)
            raise
        if task.get('validation_command'):
            result = subprocess.run(task['validation_command'], capture_output=True, timeout=45)
            if result.returncode:
                raise ValueError('native training validation failed: ' + result.stderr.decode(errors='replace')[-4096:])
        return {'run_dir': str(run_dir)}
    raise ValueError('unsupported peer operation')


if __name__ == '__main__':
    try:
        task = json.loads(sys.argv[1])
        signal.alarm(600 if task.get('action') == 'stage_dataset' else 60)
        print(json.dumps(execute(task)))
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
