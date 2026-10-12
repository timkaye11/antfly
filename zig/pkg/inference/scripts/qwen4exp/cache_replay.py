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

"""Replay little-endian u16 [token,48,10] routes under fixed-byte cache policies.

This is an offline I/O model, not measured inference speed. Current-route experts
are pinned together before eviction. Scores concern decode after prompt warmup.
"""

import argparse
import collections
import json
import struct
from pathlib import Path


def acquisitions(routes, prompt_tokens, chunk, read_group=1):
    if chunk:
        for start in range(0, prompt_tokens, chunk):
            for layer in range(48):
                experts = sorted(
                    {
                        expert
                        for token in range(start, min(start + chunk, prompt_tokens))
                        for expert in routes[token * 48 + layer]
                    }
                )
                for offset in range(0, len(experts), read_group):
                    yield start, layer, tuple(experts[offset : offset + read_group])
    else:
        for step, ids in enumerate(routes[: prompt_tokens * 48]):
            yield step // 48, step % 48, ids
    for step in range(prompt_tokens * 48, len(routes)):
        yield step // 48, step % 48, routes[step]


def replay(
    routes, strides, capacities, prompt_tokens, global_pool=False, chunk=0, read_group=1
):
    pools = [collections.OrderedDict() for _ in range(1 if global_pool else 48)]
    budget = sum(a * b for a, b in zip(strides, capacities))
    used = [0] * len(pools)
    stats = {
        "prefill": {"hits": 0, "misses": 0, "bytes": 0},
        "decode": {"hits": 0, "misses": 0, "bytes": 0},
    }
    for token, layer, ids in acquisitions(routes, prompt_tokens, chunk, read_group):
        pool_id = 0 if global_pool else layer
        pool = pools[pool_id]
        capacity = budget if global_pool else capacities[layer] * strides[layer]
        pinned = {(layer, expert) for expert in ids}
        metric = stats["prefill" if token < prompt_tokens else "decode"]
        for expert in ids:
            key = (layer, expert)
            if key in pool:
                pool.move_to_end(key)
        # Preserve router order for deterministic admission and tie-breaking.
        for expert in ids:
            key = (layer, expert)
            if key in pool:
                metric["hits"] += 1
                continue
            metric["misses"] += 1
            metric["bytes"] += strides[layer]
            while used[pool_id] + strides[layer] > capacity:
                victim = next((k for k in pool if k not in pinned), None)
                if victim is None:
                    raise ValueError("cache cannot hold a complete route")
                used[pool_id] -= pool.pop(victim)
            pool[key] = strides[layer]
            used[pool_id] += strides[layer]
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("receipt", type=Path)
    a = p.parse_args()
    d = json.loads(a.receipt.read_text())
    if isinstance(d["expert_slot_counts"], str):
        d["expert_slot_counts"] = list(map(ord, d["expert_slot_counts"]))
    raw = a.trace.read_bytes()
    if len(raw) % (48 * 10 * 2):
        p.error("trace must contain complete token layers")
    routes = list(struct.iter_unpack("<10H", raw))
    if len(routes) != (d["prompt_tokens"] + d["decode_steps"]) * 48:
        p.error("receipt and trace lengths differ")
    if any(len(set(ids)) != 10 or max(ids) >= 512 for ids in routes):
        p.error("invalid expert route")
    result = {
        "measured": {"prefill": d["prefill_expert_io"], "decode": d["decode_expert_io"]}
    }
    for name, global_pool in (("per_layer_lru", False), ("global_byte_lru", True)):
        result[name] = replay(
            routes,
            d["expert_strides"],
            d["expert_slot_counts"],
            d["prompt_tokens"],
            global_pool,
            d.get("prefill_chunk", 0),
            d.get("prefill_read_group", 1),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
