#!/usr/bin/env python3
"""Launch the same pre-staged Antfly LoRA training command on two Macs.

The caller supplies identical absolute paths on both machines. The launcher
checks bytes at those paths before starting either rank and never copies data.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


PREFLIGHT = r'''
import hashlib, json, pathlib, platform, subprocess, sys
paths = json.loads(sys.argv[1])
rank = int(sys.argv[2])
transport = sys.argv[3]
rows, link_names = [], []
if transport == 'jaccl':
    topology_path = pathlib.Path(sys.argv[4])
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
print(json.dumps({'macos': platform.mac_ver()[0], 'transport': transport, 'devices': rows,
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
    parser.add_argument('--transport', choices=('jaccl', 'tcp'), default='jaccl')
    parser.add_argument('--coordinator', required=True, help='rank 0 host:port reachable by rank 1')
    parser.add_argument('--devices-file', help='JACCL topology JSON at the same path on both Macs')
    parser.add_argument('--library', required=True, help='selected transport bridge at the same path on both Macs')
    parser.add_argument('--check', action='append', default=[], help='model or dataset path to hash on both Macs; repeat as needed')
    parser.add_argument('--report', required=True, help='local JSON run report path')
    parser.add_argument('--compare-adapter', help='same absolute adapter safetensors path on both Macs, hashed after successful training')
    parser.add_argument('--timeout-seconds', type=int, default=0, help='stop a stalled two-rank command after this many seconds; 0 disables the limit')
    parser.add_argument('--tcp-startup-timeout-seconds', type=int, default=1800,
                        help='TCP rendezvous deadline, including rank preparation skew (default: 1800)')
    parser.add_argument('--tcp-io-timeout-seconds', type=int, default=1800,
                        help='TCP socket wait deadline, including peer computation (default: 1800)')
    parser.add_argument('--heartbeat-timeout-seconds', type=int, default=30,
                        help='supervisor lease; stops training if launcher heartbeats disappear')
    parser.add_argument('--shutdown-grace-seconds', type=int, default=30,
                        help='grace before supervisors kill their training process groups')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='verify both hosts and write a report without starting ranks')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='-- <training executable> <arguments>')
    args = parser.parse_args()
    if args.command and args.command[0] == '--': args.command.pop(0)
    if not args.preflight_only and not args.command:
        parser.error('supply a command after -- or use --preflight-only')
    if not args.preflight_only and not args.check and Path(args.command[0]).name != 'jaccl_smoke.py':
        parser.error('supply at least one --check model or dataset path')
    if args.transport == 'jaccl' and not args.devices_file:
        parser.error('--devices-file is required for JACCL')
    if args.transport == 'tcp' and args.devices_file:
        parser.error('--devices-file is only used with JACCL')
    if args.timeout_seconds < 0:
        parser.error('--timeout-seconds must be nonnegative')
    for name in ('tcp_startup_timeout_seconds', 'tcp_io_timeout_seconds',
                 'heartbeat_timeout_seconds', 'shutdown_grace_seconds'):
        if not 1 <= getattr(args, name) <= 604800:
            parser.error(name.replace('_', '-') + ' must be between 1 and 604800')
    if args.compare_adapter and args.preflight_only:
        parser.error('--compare-adapter requires a training command')
    for path in (*([args.devices_file] if args.devices_file else []), args.library, *args.check, *args.command[:1], *([args.compare_adapter] if args.compare_adapter else [])):
        if not Path(path).is_absolute(): parser.error('all executable, library, topology and checked paths must be absolute')
    return args


def remote_command(host, command, env=None):
    words = ['env'] + [f'{key}={value}' for key, value in (env or {}).items()] + command
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3', host, shlex.join(words)]


def supervisor_command(args, rank, source):
    env = {'ANTFLY_DISTRIBUTED_TRANSPORT': args.transport,
           'ANTFLY_DISTRIBUTED_COORDINATOR': args.coordinator,
           'ANTFLY_DISTRIBUTED_LIBRARY': args.library,
           'ANTFLY_DISTRIBUTED_RANK': str(rank),
           'ANTFLY_TCP_STARTUP_TIMEOUT_SECONDS': str(args.tcp_startup_timeout_seconds),
           'ANTFLY_TCP_IO_TIMEOUT_SECONDS': str(args.tcp_io_timeout_seconds)}
    if args.devices_file:
        env['ANTFLY_DISTRIBUTED_DEVICES_FILE'] = args.devices_file
    config = {'command': args.command, 'env': env, 'timeout': args.timeout_seconds,
              'heartbeat_timeout': args.heartbeat_timeout_seconds,
              'shutdown_grace': args.shutdown_grace_seconds}
    command = [sys.executable if rank == 0 else 'python3', '-c', source, json.dumps(config)]
    return command if rank == 0 else remote_command(args.remote, command)


def send_control(process, message):
    if process.poll() is not None:
        return
    try:
        os.write(process.stdin.fileno(), message)
    except (BrokenPipeError, BlockingIOError):
        # A blocked SSH connection cannot block the launcher. The remote
        # supervisor will expire its independent heartbeat lease.
        pass


def stop_ranks(processes, timeout):
    for process in processes:
        send_control(process, b'S')
    deadline = time.monotonic() + timeout
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def cleanup_receipt(path):
    with path.open('rb') as source:
        source.seek(0, os.SEEK_END)
        source.seek(max(0, source.tell() - 65536))
        lines = source.read().splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(value, dict) and value.get('event') == 'distributed_rank_exit':
            return value
    return None


def main():
    args = parse_args()
    paths = sorted(set([*([args.devices_file] if args.devices_file else []), args.library, *args.command[:1], *args.check]))
    supervisor_source = Path(__file__).with_name('distributed_rank_supervisor.py').read_text()
    local = supervisor_command(args, 0, supervisor_source) if args.command else None
    remote = supervisor_command(args, 1, supervisor_source) if args.command else None
    if args.dry_run:
        print(json.dumps({'local': local, 'remote': remote, 'checked_paths': paths,
                          'preflight_only': args.preflight_only,
                          'compare_adapter': args.compare_adapter}, indent=2))
        return 0
    report = {'started_unix': time.time(), 'command': args.command, 'remote': args.remote,
              'coordinator': args.coordinator, 'transport': args.transport, 'checked_paths': paths, 'status': 'preflight_running',
              'tcp_startup_timeout_seconds': args.tcp_startup_timeout_seconds,
              'tcp_io_timeout_seconds': args.tcp_io_timeout_seconds,
              'heartbeat_timeout_seconds': args.heartbeat_timeout_seconds,
              'shutdown_grace_seconds': args.shutdown_grace_seconds}
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
        local_check = subprocess.run([sys.executable, '-c', PREFLIGHT, json.dumps(paths), '0', args.transport, args.devices_file or ''], capture_output=True, text=True, check=True)
        preflight_stage = 'rank1'
        remote_check = subprocess.run(remote_command(args.remote, ['python3', '-c', PREFLIGHT, json.dumps(paths), '1', args.transport, args.devices_file or '']), capture_output=True, text=True, check=True)
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
    cleanup_timeout = args.heartbeat_timeout_seconds + args.shutdown_grace_seconds + 5
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    previous_sigint = signal.getsignal(signal.SIGINT)
    try:
        logs = [path.open('wb') for path in log_paths]
        for command, log in zip((local, remote), logs):
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            os.set_blocking(process.stdin.fileno(), False)
            send_control(process, b'H')
        deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds else None
        heartbeat = time.monotonic()
        while True:
            if time.monotonic() >= heartbeat:
                for process in processes:
                    send_control(process, b'H')
                heartbeat = time.monotonic() + min(1, args.heartbeat_timeout_seconds / 3)
            codes = [process.poll() for process in processes]
            if all(code is not None for code in codes): break
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                stop_ranks(processes, cleanup_timeout)
                break
            if any(code is not None and code != 0 for code in codes):
                stop_ranks(processes, cleanup_timeout)
                break
            time.sleep(0.2)
        codes = [process.wait() for process in processes]
    except BaseException:
        # Ignore additional interrupts while the supervisors finish cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_ranks(processes, cleanup_timeout)
        report.update(finished_unix=time.time(), status='failed',
                      rank_cleanup=[cleanup_receipt(path) for path in log_paths if path.exists()])
        write_report()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        for process in processes:
            if process.stdin: process.stdin.close()
        for log in logs: log.close()
    report['rank_cleanup'] = [cleanup_receipt(path) for path in log_paths]
    timed_out = timed_out or any(receipt and receipt.get('reason') == 'timeout'
                                for receipt in report['rank_cleanup'])
    report.update(finished_unix=time.time(), rank_exit_codes=codes,
                  status='timeout' if timed_out else 'complete' if codes == [0, 0] else 'failed')
    if report['status'] == 'complete' and not all(
            receipt and receipt.get('cleanup_complete') and receipt.get('exit_code') == 0
            for receipt in report['rank_cleanup']):
        report['status'] = 'cleanup_unconfirmed'
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
