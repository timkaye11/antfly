#!/usr/bin/env python3
"""Find the newest acknowledged GLiNER2.5 checkpoint shared by both ranks.

Prints rank-local resume_from paths. Never modifies checkpoint directories.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from launch_jaccl_finetune import remote_command


# Send the same scanner to the peer; no helper installation is required there.
SCAN = r'''
import json, pathlib, re, sys
directory = pathlib.Path(sys.argv[1])
if not directory.is_dir(): raise SystemExit('checkpoint directory is missing')
receipts = []
for path in directory.glob('checkpoint-*-*.safetensors.json'):
    if path.stat().st_size > 8192: continue
    try:
        value = json.loads(path.read_text())
        name = value['checkpoint']
        if not isinstance(name, str) or not re.fullmatch(r'checkpoint-[0-9]+-[0-9]+\.safetensors', name): continue
        if path.name != name + '.json' or not (directory / name).is_file(): continue
        receipts.append(value)
    except (ValueError, KeyError, TypeError):
        continue
print(json.dumps(receipts))
'''


def validated(receipts, rank):
    result = {}
    for receipt in receipts:
        try:
            if receipt['format'] != 'antfly.distributed-checkpoint/v1' or receipt['rank'] != rank:
                continue
            identity = receipt['identity']
            key = (identity['microbatch_step'], identity['optimizer_step'])
            if any(type(number) is not int or number < 0 for number in key):
                continue
            if receipt['checkpoint'] != f'checkpoint-{key[0]}-{key[1]}.safetensors':
                continue
            for field in ('shared_run_fingerprint', 'shared_state_sha256'):
                digest = receipt[field]
                if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                    raise ValueError('invalid digest')
            result[key] = receipt
        except (ValueError, KeyError, TypeError):
            continue
    return result


def newest_common(local, peer):
    local, peer = validated(local, 0), validated(peer, 1)
    for key in sorted(local.keys() & peer.keys(), reverse=True):
        first, second = local[key], peer[key]
        if all(first[field] == second[field] for field in ('shared_run_fingerprint', 'shared_state_sha256')):
            return first, second
    raise ValueError('no common acknowledged checkpoint; retain both directories for diagnosis')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', required=True, help='rank 1 SSH destination')
    parser.add_argument('--directory', required=True, help='rank 0 checkpoint directory')
    parser.add_argument('--remote-directory', help='defaults to the same absolute directory on rank 1')
    args = parser.parse_args()
    remote_directory = args.remote_directory or args.directory
    if not all(Path(path).is_absolute() for path in (args.directory, remote_directory)):
        parser.error('checkpoint directories must be absolute')
    local = subprocess.run([sys.executable, '-c', SCAN, args.directory], check=True, capture_output=True, text=True)
    peer = subprocess.run(remote_command(args.remote, ['python3', '-c', SCAN, remote_directory]),
                          check=True, capture_output=True, text=True, timeout=60)
    first, second = newest_common(json.loads(local.stdout), json.loads(peer.stdout))
    print(json.dumps({'identity': first['identity'],
                      'rank0_resume_from': str(Path(args.directory) / first['checkpoint']),
                      'rank1_resume_from': str(Path(remote_directory) / second['checkpoint'])}, indent=2))


if __name__ == '__main__':
    main()
