#!/usr/bin/env python3
"""Local lifecycle/failure tests; no models, SSH server, or RDMA device required."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import find_distributed_checkpoint as recovery

SCRIPTS = Path(__file__).resolve().parent
SUPERVISOR = SCRIPTS / 'distributed_rank_supervisor.py'


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError('condition did not become true')


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def receipt(rank, step):
    return {'format': 'antfly.distributed-checkpoint/v1', 'rank': rank,
            'checkpoint': f'checkpoint-{step}-{step}.safetensors',
            'identity': {'microbatch_step': step, 'optimizer_step': step},
            'shared_run_fingerprint': '01' * 32, 'shared_state_sha256': f'{step:02x}' * 32}


class RecoveryTests(unittest.TestCase):
    def test_failed_generation_preserves_previous_common_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directories = [Path(temporary) / str(rank) for rank in range(2)]
            for rank, directory in enumerate(directories):
                directory.mkdir()
                for step in (0, 1):
                    item = receipt(rank, step)
                    (directory / item['checkpoint']).write_bytes(b'durable checkpoint')
                    (directory / (item['checkpoint'] + '.json')).write_text(json.dumps(item))
            # Rank 0 published a new snapshot and receipt; rank 1's write failed.
            item = receipt(0, 2)
            (directories[0] / item['checkpoint']).write_bytes(b'new checkpoint')
            (directories[0] / (item['checkpoint'] + '.json')).write_text(json.dumps(item))
            # Stray/malformed receipts must not be mistaken for recovery points.
            (directories[1] / 'checkpoint-3-3.safetensors.json').write_text('{}')
            def scan(directory):
                return json.loads(subprocess.check_output([sys.executable, '-c', recovery.SCAN, str(directory)]))
            first, second = recovery.newest_common(*(scan(d) for d in directories))
            self.assertEqual(first['identity']['microbatch_step'], 1)
            self.assertEqual(first['checkpoint'], second['checkpoint'])
            # Both files but only one receipt still selects the older pair.
            (directories[1] / item['checkpoint']).write_bytes(b'new checkpoint')
            self.assertEqual(recovery.newest_common(*(scan(d) for d in directories))[0]['identity']['microbatch_step'], 1)
            item = receipt(1, 2)
            (directories[1] / (item['checkpoint'] + '.json')).write_text(json.dumps(item))
            self.assertEqual(recovery.newest_common(*(scan(d) for d in directories))[0]['identity']['microbatch_step'], 2)

    def test_divergent_state_or_run_is_not_a_common_checkpoint(self):
        for field in ('shared_state_sha256', 'shared_run_fingerprint'):
            peer = receipt(1, 2)
            peer[field] = '09' * 32
            common = recovery.newest_common([receipt(0, 1), receipt(0, 2)], [receipt(1, 1), peer])
            self.assertEqual(common[0]['identity']['microbatch_step'], 1)
        with self.assertRaises(ValueError):
            recovery.newest_common([receipt(0, 1)], [receipt(1, 2)])


class SupervisorTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'darwin', 'Mac process-group reaping test; container PID 1 may retain orphan zombies')
    def test_supervisor_removes_descendants_when_the_command_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / 'descendant.pid'
            descendant = ('import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                          f'pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)')
            # The command deliberately leaves a child behind and exits without
            # a trailing newline, exercising cleanup and receipt framing.
            command = ('import subprocess,sys,time; '
                       f'subprocess.Popen([sys.executable,"-c",{descendant!r}]); '
                       'time.sleep(0.3); sys.stdout.write("last training output")')
            config = {'command': [sys.executable, '-c', command], 'env': {'ANTFLY_DISTRIBUTED_RANK': '1'},
                      'timeout': 5, 'heartbeat_timeout': 5, 'shutdown_grace': 0.2}
            process = subprocess.Popen([sys.executable, str(SUPERVISOR), json.dumps(config)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                process.stdin.write(b'H'); process.stdin.flush()
                wait_for(pid_file.exists)
                pid = int(pid_file.read_text())
                process.wait(timeout=6)
                wait_for(lambda: not alive(pid))
                self.assertEqual(process.returncode, 0, process.stderr.read())
                event = json.loads(process.stdout.read().splitlines()[-1])
                self.assertTrue(event['cleanup_complete'])
            finally:
                if process.poll() is None: process.kill()
                process.communicate()
                if pid_file.exists() and alive(int(pid_file.read_text())):
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)

    def test_eof_and_expired_lease_kill_a_term_resistant_process_group(self):
        for mode in ('eof', 'lease'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                pid_file = Path(temporary) / 'pid'
                code = ('import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                        f'pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)')
                config = {'command': [sys.executable, '-c', code], 'env': {'ANTFLY_DISTRIBUTED_RANK': '1'},
                          'timeout': 0, 'heartbeat_timeout': 1, 'shutdown_grace': 0.2}
                process = subprocess.Popen([sys.executable, str(SUPERVISOR), json.dumps(config)],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    process.stdin.write(b'H'); process.stdin.flush()
                    wait_for(pid_file.exists)
                    pid = int(pid_file.read_text())
                    if mode == 'eof':
                        process.stdin.close(); process.stdin = None
                    process.wait(timeout=5)
                    wait_for(lambda: not alive(pid))
                    event = json.loads(process.stdout.read())
                    self.assertTrue(event['cleanup_complete'])
                    self.assertEqual(event['reason'], 'launcher_disconnected' if mode == 'eof' else 'lease_expired')
                finally:
                    if process.poll() is None: process.kill()
                    process.communicate()
                    if pid_file.exists() and alive(int(pid_file.read_text())):
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)

    def test_launcher_timeout_and_peer_failure_confirm_remote_cleanup(self):
        for mode in ('timeout', 'failure'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                relay = directory / 'ssh'
                relay.write_text('#!' + sys.executable + '\nimport subprocess,sys,shlex\np=subprocess.Popen(shlex.split(sys.argv[-1])); sys.exit(p.wait())\n')
                relay.chmod(0o755)
                task = directory / 'task'
                task.write_text('#!' + sys.executable + '\nimport os,time,pathlib,sys\n'
                                'rank=os.environ["ANTFLY_DISTRIBUTED_RANK"]\n'
                                f'pathlib.Path({str(directory)!r}, "rank"+rank+".pid").write_text(str(os.getpid()))\n'
                                + ('if rank=="0":\n time.sleep(0.5)\n sys.exit(7)\n' if mode == 'failure' else '')
                                + 'time.sleep(60)\n')
                task.chmod(0o755)
                report = directory / 'report.json'
                command = [sys.executable, str(SCRIPTS / 'launch_jaccl_finetune.py'), '--remote', 'fixture',
                           '--transport', 'tcp', '--coordinator', '127.0.0.1:1', '--library', str(task),
                           '--check', str(task), '--report', str(report), '--timeout-seconds', '2',
                           '--shutdown-grace-seconds', '1', '--heartbeat-timeout-seconds', '2', '--', str(task)]
                try:
                    result = subprocess.run(command, env={**os.environ, 'PATH': str(directory) + ':' + os.environ['PATH']},
                                            capture_output=True, text=True, timeout=15)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    data = json.loads(report.read_text())
                    self.assertEqual(data['status'], 'timeout' if mode == 'timeout' else 'failed')
                    self.assertTrue(all(row and row['cleanup_complete'] for row in data['rank_cleanup']), data)
                    self.assertEqual(len(list(directory.glob('rank*.pid'))), 2, data)
                    for path in directory.glob('rank*.pid'):
                        pid = int(path.read_text())
                        wait_for(lambda: not alive(pid))
                finally:
                    for path in directory.glob('rank*.pid'):
                        pid = int(path.read_text())
                        if alive(pid): os.kill(pid, signal.SIGKILL)


class TcpTimeoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.library = str(Path(cls.temporary.name) / 'tcp.dylib')
        linking = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
        subprocess.run(['c++', '-std=c++20', '-O2', *linking,
                        str(SCRIPTS.parent / 'src/finetune/distributed/tcp_bridge.cpp'), '-o', cls.library], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_launcher_collectives_complete_under_supervision(self):
        with tempfile.TemporaryDirectory() as temporary, socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            endpoint = f'127.0.0.1:{sock.getsockname()[1]}'
            sock.close()
            directory = Path(temporary)
            relay = directory / 'ssh'
            relay.write_text('#!' + sys.executable + '\nimport subprocess,sys,shlex\np=subprocess.Popen(shlex.split(sys.argv[-1])); sys.exit(p.wait())\n')
            relay.chmod(0o755)
            report = directory / 'report.json'
            result = subprocess.run([
                sys.executable, str(SCRIPTS / 'launch_jaccl_finetune.py'),
                '--transport', 'tcp', '--remote', 'fixture', '--coordinator', endpoint,
                '--library', self.library, '--report', str(report), '--timeout-seconds', '15',
                '--', str(SCRIPTS / 'jaccl_smoke.py')],
                env={**os.environ, 'PATH': str(directory) + ':' + os.environ['PATH']},
                capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            data = json.loads(report.read_text())
            self.assertEqual(data['status'], 'complete')
            self.assertEqual(data['rank_exit_codes'], [0, 0])
            self.assertTrue(all(row['cleanup_complete'] for row in data['rank_cleanup']))
            for path in data['rank_logs']:
                events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.startswith('{')]
                smoke = next(row for row in events if row.get('event') == 'collective_smoke')
                self.assertEqual(smoke['status'], 'pass')

    def test_configured_rendezvous_accepts_delayed_peer_and_times_out_when_missing(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); endpoint = f'127.0.0.1:{sock.getsockname()[1]}'
        code = '''import ctypes,os,sys
b=ctypes.CDLL(sys.argv[1]); h=ctypes.c_void_p()
b.antfly_jaccl_open.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)]
b.antfly_jaccl_close.argtypes=[ctypes.c_void_p]
b.antfly_jaccl_last_error.restype=ctypes.c_char_p
r=b.antfly_jaccl_open(int(sys.argv[2]),sys.argv[3].encode(),b'',ctypes.byref(h))
if r: print(b.antfly_jaccl_last_error().decode())
else: b.antfly_jaccl_close(h)
sys.exit(0 if r==0 else 1)
'''
        env = {**os.environ, 'ANTFLY_TCP_STARTUP_TIMEOUT_SECONDS': '4', 'ANTFLY_TCP_IO_TIMEOUT_SECONDS': '4'}
        rank0 = subprocess.Popen([sys.executable, '-c', code, self.library, '0', endpoint], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            time.sleep(1.2)
            rank1 = subprocess.run([sys.executable, '-c', code, self.library, '1', endpoint], env=env,
                                   capture_output=True, timeout=8)
            self.assertEqual(rank1.returncode, 0, rank1.stdout + rank1.stderr)
            stdout, stderr = rank0.communicate(timeout=8)
            self.assertEqual(rank0.returncode, 0, stdout + stderr)
        finally:
            if rank0.poll() is None: rank0.kill()
            rank0.communicate()
        env['ANTFLY_TCP_STARTUP_TIMEOUT_SECONDS'] = '1'
        result = subprocess.run([sys.executable, '-c', code, self.library, '0', endpoint], env=env,
                                capture_output=True, timeout=4)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'timed out waiting', result.stdout)
        env['ANTFLY_TCP_STARTUP_TIMEOUT_SECONDS'] = '0'
        result = subprocess.run([sys.executable, '-c', code, self.library, '0', endpoint], env=env,
                                capture_output=True, timeout=4)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'ANTFLY_TCP_STARTUP_TIMEOUT_SECONDS', result.stdout)


if __name__ == '__main__':
    unittest.main()
