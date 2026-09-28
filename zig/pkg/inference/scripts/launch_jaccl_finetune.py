#!/usr/bin/env python3
"""Launch the same pre-staged Antfly LoRA training command on two TB5 Macs.

The caller supplies identical absolute paths on both machines. The launcher
checks bytes at those paths before starting either rank and never copies data.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


PREFLIGHT = r'''
import hashlib, json, pathlib, platform, subprocess, sys
paths = json.loads(sys.argv[1])
rank = int(sys.argv[2])
topology_path = pathlib.Path(sys.argv[3])
version = tuple(int(p) for p in platform.mac_ver()[0].split('.')[:2])
if version < (26, 2): raise SystemExit('macOS 26.2 or newer is required for TB5 RDMA')
devices = subprocess.run(['ibv_devices'], capture_output=True, text=True, check=True).stdout
rows = [line.strip() for line in devices.splitlines()
        if len(line.split()) == 2 and line.split()[1] and
        all(char in '0123456789abcdefABCDEF' for char in line.split()[1])]
if not rows: raise SystemExit('no RDMA device found by ibv_devices')
found_devices = {line.split()[0] for line in rows}
with topology_path.open() as source: topology = json.load(source)
if (not isinstance(topology, list) or len(topology) != 2 or
        any(not isinstance(row, list) or len(row) != 2 for row in topology) or
        topology[0][0] is not None or topology[1][1] is not None):
    raise SystemExit('JACCL topology must be a 2x2 matrix with null diagonal')
link = topology[rank][1 - rank]
link_names = [link] if isinstance(link, str) else link
if (not isinstance(link_names, list) or not link_names or
        any(not isinstance(name, str) or not name for name in link_names)):
    raise SystemExit('JACCL topology needs an RDMA device for the peer')
if not set(link_names).issubset(found_devices):
    raise SystemExit('JACCL topology names not present in local ibv_devices: ' +
                     repr(sorted(set(link_names) - found_devices)))
def digest(path):
    path = pathlib.Path(path)
    if not path.exists(): raise SystemExit('missing pre-staged path: ' + str(path))
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    for f in files:
        relative = f.name if path.is_file() else str(f.relative_to(path))
        h.update(relative.encode()); h.update(b'\0')
        with f.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
print(json.dumps({'macos': platform.mac_ver()[0], 'devices': rows,
                  'topology_devices': link_names,
                  'digests': {p: digest(p) for p in paths}}, sort_keys=True))
'''

HASH_FILE = r'''
import hashlib, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file(): raise SystemExit('missing adapter artifact: ' + str(path))
hash = hashlib.sha256()
with path.open('rb') as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b''): hash.update(chunk)
print(hash.hexdigest())
'''


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', required=True, help='SSH destination of rank 1')
    parser.add_argument('--coordinator', required=True, help='rank 0 JACCL host:port reachable over Thunderbolt')
    parser.add_argument('--devices-file', required=True, help='JACCL topology JSON at the same path on both Macs')
    parser.add_argument('--library', required=True, help='libantfly_jaccl.dylib at the same path on both Macs')
    parser.add_argument('--check', action='append', default=[], help='model or dataset path to hash on both Macs; repeat as needed')
    parser.add_argument('--report', required=True, help='local JSON run report path')
    parser.add_argument('--compare-adapter', help='same absolute adapter safetensors path on both Macs, hashed after successful training')
    parser.add_argument('--timeout-seconds', type=int, default=0, help='stop a stalled two-rank command after this many seconds; 0 disables the limit')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='verify both hosts and write a report without starting ranks')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='-- <training executable> <arguments>')
    args = parser.parse_args()
    if args.command and args.command[0] == '--': args.command.pop(0)
    if not args.preflight_only and not args.command:
        parser.error('supply a command after -- or use --preflight-only')
    if not args.preflight_only and not args.check and Path(args.command[0]).name != 'jaccl_smoke.py':
        parser.error('supply at least one --check model or dataset path')
    if args.command and (Path(args.command[0]).name == 'train-gliner25' or args.command[1:5] == ['finetune', 'train', 'run', 'gliner25']):
        parser.error('GLiNER2.5 distributed optimizer and replay are not implemented')
    if args.timeout_seconds < 0:
        parser.error('--timeout-seconds must be nonnegative')
    if args.compare_adapter and args.preflight_only:
        parser.error('--compare-adapter requires a training command')
    for path in (args.devices_file, args.library, *args.check, *args.command[:1], *([args.compare_adapter] if args.compare_adapter else [])):
        if not Path(path).is_absolute(): parser.error('all executable, library, topology and checked paths must be absolute')
    return args


def remote_command(host, command, env=None):
    words = ['env'] + [f'{key}={value}' for key, value in (env or {}).items()] + command
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(words)]


def main():
    args = parse_args()
    paths = sorted(set([args.devices_file, args.library, *args.command[:1], *args.check]))
    rank_env = {'ANTFLY_JACCL_COORDINATOR': args.coordinator,
                'ANTFLY_JACCL_DEVICES_FILE': args.devices_file,
                'ANTFLY_JACCL_LIBRARY': args.library}
    remote = remote_command(args.remote, args.command, {**rank_env, 'ANTFLY_JACCL_RANK': '1'}) if args.command else None
    local_env = {**os.environ, **rank_env, 'ANTFLY_JACCL_RANK': '0'}
    if args.dry_run:
        print(json.dumps({'local': args.command, 'remote': remote, 'checked_paths': paths,
                          'preflight_only': args.preflight_only,
                          'compare_adapter': args.compare_adapter}, indent=2))
        return 0
    report = {'started_unix': time.time(), 'command': args.command, 'remote': args.remote,
              'coordinator': args.coordinator, 'checked_paths': paths, 'status': 'preflight_running'}
    report_path = Path(args.report)
    if report_path.exists():
        raise SystemExit(f'report already exists: {report_path}')
    log_dir = report_path.parent / (report_path.stem + '-logs')
    if not args.preflight_only and log_dir.exists():
        raise SystemExit(f'rank log directory already exists: {log_dir}')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    def write_report():
        temporary = report_path.with_suffix(report_path.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(report_path)
    write_report()
    preflight_stage = 'rank0'
    try:
        local_check = subprocess.run([sys.executable, '-c', PREFLIGHT, json.dumps(paths), '0', args.devices_file], capture_output=True, text=True, check=True)
        preflight_stage = 'rank1'
        remote_check = subprocess.run(remote_command(args.remote, ['python3', '-c', PREFLIGHT, json.dumps(paths), '1', args.devices_file]), capture_output=True, text=True, check=True)
        local_info, remote_info = json.loads(local_check.stdout), json.loads(remote_check.stdout)
        report.update(local_preflight=local_info, remote_preflight=remote_info)
        preflight_stage = 'hash_compare'
        if local_info['digests'] != remote_info['digests']:
            raise ValueError('pre-staged file hash mismatch between ranks')
    except (subprocess.CalledProcessError, ValueError, OSError) as error:
        report.update(finished_unix=time.time(), status='preflight_failed', preflight_stage=preflight_stage,
                      error=(error.stderr or str(error)) if isinstance(error, subprocess.CalledProcessError) else str(error))
        write_report()
        raise
    if args.preflight_only:
        report.update(finished_unix=time.time(), status='preflight_complete')
        write_report()
        print(json.dumps({'status': 'preflight_complete', 'report': str(report_path)}))
        return 0
    log_dir.mkdir(parents=True, exist_ok=False)
    log_paths = [log_dir / 'rank0.log', log_dir / 'rank1.log']
    report.update(status='running', rank_logs=[str(path) for path in log_paths])
    write_report()
    processes = []
    logs = []
    timed_out = False
    try:
        logs = [path.open('wb') for path in log_paths]
        processes.append(subprocess.Popen(args.command, env=local_env, stdout=logs[0], stderr=subprocess.STDOUT))
        processes.append(subprocess.Popen(remote, stdout=logs[1], stderr=subprocess.STDOUT))
        deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds else None
        while True:
            codes = [process.poll() for process in processes]
            if all(code is not None for code in codes): break
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                for process in processes:
                    if process.poll() is None: process.terminate()
                break
            if any(code is not None and code != 0 for code in codes):
                for process in processes:
                    if process.poll() is None: process.terminate()
                break
            time.sleep(0.2)
        codes = []
        for process in processes:
            try:
                codes.append(process.wait(timeout=5))
            except subprocess.TimeoutExpired:
                process.kill()
                codes.append(process.wait())
    except BaseException:
        for process in processes:
            if process.poll() is None: process.terminate()
        for process in processes:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        report.update(finished_unix=time.time(), status='failed')
        write_report()
        raise
    finally:
        for log in logs: log.close()
    report.update(finished_unix=time.time(), rank_exit_codes=codes,
                  status='timeout' if timed_out else 'complete' if codes == [0, 0] else 'failed')
    if report['status'] == 'complete' and args.compare_adapter:
        try:
            local_hash = subprocess.run([sys.executable, '-c', HASH_FILE, args.compare_adapter], capture_output=True, text=True, check=True).stdout.strip()
            remote_hash = subprocess.run(remote_command(args.remote, ['python3', '-c', HASH_FILE, args.compare_adapter]), capture_output=True, text=True, check=True).stdout.strip()
            report['adapter_sha256'] = {'rank0': local_hash, 'rank1': remote_hash}
            if local_hash != remote_hash:
                report['status'] = 'artifact_mismatch'
        except (subprocess.CalledProcessError, OSError) as error:
            report.update(status='artifact_check_failed', error=(error.stderr or str(error)) if isinstance(error, subprocess.CalledProcessError) else str(error))
    write_report()
    print(json.dumps({'status': report['status'], 'rank_exit_codes': codes,
                      'rank_logs': report['rank_logs'], 'report': str(report_path)}))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        raise SystemExit(error.returncode)
