#!/usr/bin/env python3
"""Own one training process group; the launcher's stdin heartbeats are a lease.

This source is sent to Python on each host, so no remote installation is needed.
The training command receives /dev/null on stdin and never consumes the lease.
"""
import json
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
    try:
        child = subprocess.Popen(config['command'], env={**os.environ, **config['env']},
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
                if any(byte != ord('H') for byte in data):
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
        print('\n' + json.dumps({'event': 'distributed_rank_exit', 'rank': config['env']['ANTFLY_DISTRIBUTED_RANK'],
                                 'reason': reason, 'exit_code': code, 'cleanup_complete': cleaned}), flush=True)
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(supervise(json.loads(sys.argv[1])))
