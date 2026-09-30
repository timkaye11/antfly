#!/usr/bin/env python3
"""Own one training process group; the launcher's stdin heartbeats are a lease.

This source is sent to Python on each host, so no remote installation is needed.
The training command receives /dev/null on stdin and never consumes the lease.
"""
import json
import fcntl
import os
import select
import signal
import subprocess
import sys
import time


def supervise(config):
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    child = None
    reason = "command_exit"
    code = 1
    lock = None
    paused = False
    try:
        if config.get('lock_file'):
            lock = open(config['lock_file'], 'a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Only explicit launcher configuration can enable distributed execution.
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(('ANTFLY_DISTRIBUTED_', 'ANTFLY_TCP_', 'ANTFLY_JACCL_'))}
        child = subprocess.Popen(config['command'], env={**environment, **config['env']},
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        lease = time.monotonic() + config['heartbeat_timeout']
        deadline = (time.monotonic() + config['timeout']) if config['timeout'] else None
        while child.poll() is None:
            now = time.monotonic()
            if stopping or now >= lease or (deadline is not None and now >= deadline):
                reason = 'stop_requested' if stopping else 'lease_expired' if now >= lease else 'timeout'
                code = 124 if reason == 'timeout' else 125
                break
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable:
                data = os.read(sys.stdin.fileno(), 4096)
                if not data or b'S' in data:
                    reason = 'launcher_disconnected' if not data else 'stop_requested'
                    code = 125
                    break
                if b'P' in data and config.get('pause_supported') and not paused:
                    # Signal only the CLI parent. It owns the disposable worker
                    # and forwards the cooperative request at a safe boundary.
                    child.send_signal(signal.SIGINT)
                    paused = True
                if any(byte not in b'HP' for byte in data) or (b'P' in data and not config.get('pause_supported')):
                    reason = 'invalid_lease'
                    code = 125
                    break
                lease = now + config['heartbeat_timeout']
        else:
            code = child.returncode
    finally:
        cleaned = True
        if child is not None:
            # Signal the group even if its leader exited: it may have left
            # descendants holding devices or the SSH output descriptors.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if child.poll() is None:
                try:
                    child.wait(timeout=config['shutdown_grace'])
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            # Do not acknowledge cleanup while descendants still occupy the
            # group. Bound this check even if the OS cannot reap a process.
            deadline = time.monotonic() + 2
            while True:
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    break
                if time.monotonic() >= deadline:
                    cleaned = False
                    if code == 0: code = 125
                    break
                time.sleep(0.02)
        print('\n' + json.dumps({'event': 'distributed_rank_exit', 'rank': config.get('rank', config['env'].get('ANTFLY_DISTRIBUTED_RANK', '0')),
                                 'reason': reason, 'exit_code': code, 'cleanup_complete': cleaned}), flush=True)
        if lock is not None:
            lock.close()
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(supervise(json.loads(sys.argv[1])))
