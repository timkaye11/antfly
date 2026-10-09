#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only bounded Darwin uncached range-read probe; never purges global caches."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import random
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mib", type=int, default=96)
    args = parser.parse_args()
    if not 12 <= args.mib <= 1024:
        parser.error("mib must be 12..1024")
    fd = os.open(args.shard, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, 48, 1)  # Darwin F_NOCACHE, matching RangeReader.
        size = os.fstat(fd).st_size
        if size < 128 * 1024 * 1024:
            parser.error("select a data shard at least 128 MiB long")
        runs = []
        for workers in (1, 3, 6):
            for block in (512 * 1024, 1024 * 1024, 3 * 1024 * 1024):
                rng = random.Random(17 + workers * 101 + block)
                count = args.mib * 1024 * 1024 // block
                offsets = [
                    rng.randrange(4096, (size - block) // 16384) * 16384
                    for _ in range(count)
                ]

                def read(offset):
                    data = os.pread(fd, block, offset)
                    if len(data) != block:
                        raise OSError("short range read")
                    return len(data)

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    start = time.monotonic()
                    transferred = sum(executor.map(read, offsets))
                    elapsed = time.monotonic() - start
                result = {
                    "workers": workers,
                    "block_bytes": block,
                    "read_bytes": transferred,
                    "seconds": elapsed,
                    "GB_per_second": transferred / elapsed / 1e9,
                }
                runs.append(result)
                print(json.dumps(result), flush=True)
        args.output.write_text(
            json.dumps(
                {
                    "path": str(args.shard),
                    "size": size,
                    "read_only": True,
                    "F_NOCACHE": True,
                    "note": "Distinct pseudorandom ranges per case; F_NOCACHE requests no caching but existing OS/device cache is not invalidated. Excludes compute; not physical device traffic.",
                    "runs": runs,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
