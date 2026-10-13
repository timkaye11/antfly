#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate CPU FP32 attention goldens from BF16 inputs for native cuDNN tests."""

import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    args.output_dir.mkdir(exist_ok=False)
    generator = torch.Generator().manual_seed(20261013)
    cases = [(2, 17, 0), (8, 512, 0), (32, 512, 0), (2, 1024, 512)]
    reports = []
    for index, (batch, sequence, radius) in enumerate(cases):
        directory = args.output_dir / f'case-{index}'
        directory.mkdir()
        q = (torch.randn((batch, sequence, 4, 256), generator=generator) * .5).to(torch.bfloat16)
        k = (torch.randn((batch, sequence, 2, 256), generator=generator) * .5).to(torch.bfloat16)
        v = torch.randn((batch, sequence, 2, 256), generator=generator).to(torch.bfloat16)
        (directory / 'shape.u64').write_bytes(struct.pack('<6Q', batch, sequence, 4, 2, 256, radius))
        for name, value in [('q', q), ('k', k), ('v', v)]:
            (directory / (name + '.bf16')).write_bytes(value.contiguous().view(torch.uint16).numpy().tobytes())
        scores = q.permute(0, 2, 1, 3).float() @ k.repeat_interleave(2, dim=2).permute(0, 2, 3, 1).float()
        values = v.repeat_interleave(2, dim=2).permute(0, 2, 1, 3).float()
        positions = torch.arange(sequence)
        for variant in range(2):
            patterns = (sequence, max(1, sequence // 4), max(1, sequence // 16), min(9, sequence))
            lengths = torch.tensor([patterns[(b + variant) % 4] for b in range(batch)], dtype=torch.int32)
            valid = positions[None, :] < lengths[:, None]
            allowed = valid[:, None, :, None] & valid[:, None, None, :]
            if radius:
                allowed = allowed & ((positions[:, None] - positions[None, :]).abs() <= radius)[None, None]
            probabilities = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1).nan_to_num(0)
            expected = (probabilities @ values).permute(0, 2, 1, 3).contiguous().to(torch.bfloat16).float()
            (directory / f'lengths-{variant}.i32').write_bytes(lengths.numpy().tobytes())
            (directory / f'expected-{variant}.f32').write_bytes(expected.numpy().tobytes())
        files = {path.name: {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(directory.iterdir())}
        reports.append({'batch': batch, 'sequence': sequence, 'radius': radius, 'files': files})
    manifest = {'scope': 'CPU attention component goldens; not model quality or performance evidence',
                'torch': torch.__version__, 'scale': 1, 'cases': reports}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'cases': len(cases), 'length_variants': 2, 'directory': str(args.output_dir)}))


if __name__ == '__main__':
    main()
