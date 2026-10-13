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

This is an offline I/O model, not measured inference speed. Every policy sees
the same acquisitions: the experts of one acquisition are pinned together
before anything is evicted, and reads are whole experts. Predicted bytes are
application read ranges; throughput additionally depends on storage and on
how well reads overlap compute.

Two forms:

    cache_replay.py TRACE RECEIPT
        Compare the measured run with per-layer and global LRU at the
        receipt's own cache size.

    cache_replay.py --manifest DIR/manifest.json --output sweep.json
        Sweep cache budgets over every recorded workload (see
        record_routes.py) and compare replacement and allocation policies.
"""

import argparse
import collections
import heapq
import json
import math
import struct
from pathlib import Path

LAYERS = 48
EXPERTS = 512
TOP_K = 10
DECAY_INTERVAL = 1024
DEFAULT_BUDGETS_GB = (1.6, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 36)
POLICIES = (
    "lfu_decay_slots",
    "lru_per_layer",
    "lru_global",
    "lfu_ghost",
    "lfu_ghost_weighted",
    "belady_per_layer",
    "belady_global",
)


def budget_key(budget_gb):
    """One spelling per budget, whether it was typed as 30 or 30.0."""
    return f"{float(budget_gb):g}"


def load_routes(path):
    raw = Path(path).read_bytes()
    if len(raw) % (LAYERS * TOP_K * 2):
        raise ValueError("trace must contain complete token layers")
    routes = list(struct.iter_unpack(f"<{TOP_K}H", raw))
    if any(len(set(ids)) != TOP_K or max(ids) >= EXPERTS for ids in routes):
        raise ValueError("invalid expert route")
    return routes


def receipt_segments(receipt):
    """One-shot run: the prompt is prefilled, every later position is decode."""
    prompt = receipt["prompt_tokens"]
    return [
        ("prefill", 0, prompt),
        ("decode", prompt, prompt + receipt["decode_steps"]),
    ]


def session_segments(turns):
    """Session log: per turn, tokens [reused, reused+prefilled) then decode."""
    result = []
    for turn in turns:
        start = turn["reused_tokens"]
        middle = start + turn["prefilled_tokens"]
        if result and start != result[-1][2]:
            raise ValueError("session turns must extend one sequence")
        result += [("prefill", start, middle), ("decode", middle, turn["position"])]
    return result


def acquisitions(routes, segments, chunk=0, read_group=TOP_K, session=True):
    """Yield (phase, token, layer, experts) in runtime order.

    Chunked prefill is layer-major: each layer acquires the sorted union of
    the chunk's experts in groups, as the runtime does. Every decode step
    acquires one route per layer in router order. A session sends a one-token
    chunk down that token path too; the one-shot CLI only does so for a
    one-token prompt and otherwise runs even a trailing single token as a
    chunk.
    """
    for phase, start, end in segments:
        if phase == "prefill" and chunk:
            at = start
            while at < end:
                stop = min(end, at + chunk)
                if stop - at == 1 and (session or end - start == 1):
                    for layer in range(LAYERS):
                        yield phase, at, layer, routes[at * LAYERS + layer]
                else:
                    for layer in range(LAYERS):
                        experts = sorted(
                            {
                                expert
                                for token in range(at, stop)
                                for expert in routes[token * LAYERS + layer]
                            }
                        )
                        for offset in range(0, len(experts), read_group):
                            yield (
                                phase,
                                at,
                                layer,
                                tuple(experts[offset : offset + read_group]),
                            )
                at = stop
        else:
            for token in range(start, end):
                for layer in range(LAYERS):
                    yield phase, token, layer, routes[token * LAYERS + layer]


def equal_slots(strides, budget):
    """The runtime's allocation: one route everywhere, then round-robin."""
    slots = [TOP_K] * LAYERS
    used = sum(stride * TOP_K for stride in strides)
    if used > budget:
        return None
    progress = True
    while progress:
        progress = False
        for layer, stride in enumerate(strides):
            if slots[layer] < EXPERTS and stride <= budget - used:
                slots[layer] += 1
                used += stride
                progress = True
    return slots


class Stats:
    def __init__(self, warm_from):
        self.warm_from = warm_from
        self.rows = {
            name: {"hits": 0, "misses": 0, "bytes": 0}
            for name in ("prefill", "decode", "decode_warm")
        }

    def note(self, phase, token, hit, stride):
        names = [phase]
        if phase == "decode" and token >= self.warm_from:
            names.append("decode_warm")
        for name in names:
            row = self.rows[name]
            if hit:
                row["hits"] += 1
            else:
                row["misses"] += 1
                row["bytes"] += stride


def replay_slots(sequence, strides, slots, stats, expert_keyed):
    """Per-layer frequency policy with halving decay.

    `expert_keyed=False` is the runtime's directory: the counter lives on the
    slot and restarts when a different expert is installed. `expert_keyed=True`
    keeps one counter per expert, so an evicted expert's history (a ghost
    entry) still competes when it is routed again.
    """
    resident = [{} for _ in range(LAYERS)]  # expert -> slot
    owner = [[None] * count for count in slots]
    use = [([0] * EXPERTS if expert_keyed else [0] * count) for count in slots]
    seen = [[0] * count for count in slots]
    epoch = [0] * LAYERS
    for phase, token, layer, experts in sequence:
        epoch[layer] += 1
        now = epoch[layer]
        mine, owners, counts, last = (
            resident[layer],
            owner[layer],
            use[layer],
            seen[layer],
        )
        if now % DECAY_INTERVAL == 0:
            for index in range(len(counts)):
                counts[index] >>= 1
        pinned = set()
        missing = []
        for expert in experts:
            slot = mine.get(expert)
            if slot is None:
                missing.append(expert)
                stats.note(phase, token, False, strides[layer])
                continue
            stats.note(phase, token, True, strides[layer])
            pinned.add(slot)
            counts[expert if expert_keyed else slot] += 1
            last[slot] = now
        for expert in missing:
            victim = None
            best = None
            for slot, current in enumerate(owners):
                if slot in pinned:
                    continue
                if current is None:
                    victim = slot
                    break
                key = (counts[current if expert_keyed else slot], last[slot], slot)
                if best is None or key < best:
                    best, victim = key, slot
            if victim is None:
                raise ValueError("cache cannot hold a complete route")
            if owners[victim] is not None:
                del mine[owners[victim]]
            owners[victim] = expert
            mine[expert] = victim
            if expert_keyed:
                counts[expert] += 1
            else:
                counts[victim] = 1
            last[victim] = now
            pinned.add(victim)


def replay_lru(sequence, strides, capacities, stats, global_pool):
    pools = [collections.OrderedDict() for _ in range(1 if global_pool else LAYERS)]
    used = [0] * len(pools)
    for phase, token, layer, experts in sequence:
        pool_id = 0 if global_pool else layer
        pool = pools[pool_id]
        capacity = capacities if global_pool else capacities[layer] * strides[layer]
        pinned = {(layer, expert) for expert in experts}
        for expert in experts:
            key = (layer, expert)
            if key in pool:
                pool.move_to_end(key)
        # Preserve router order for deterministic admission and tie-breaking.
        for expert in experts:
            key = (layer, expert)
            if key in pool:
                stats.note(phase, token, True, strides[layer])
                continue
            stats.note(phase, token, False, strides[layer])
            while used[pool_id] + strides[layer] > capacity:
                victim = next((k for k in pool if k not in pinned), None)
                if victim is None:
                    raise ValueError("cache cannot hold a complete route")
                used[pool_id] -= pool.pop(victim)
            pool[key] = strides[layer]
            used[pool_id] += strides[layer]


def replay_belady(sequence, strides, capacities, stats, global_pool):
    """Evict the resident expert whose next use is furthest away.

    Exact for one layer (uniform sizes); for the global pool, where strides
    differ slightly by layer, it is the standard near-optimal bound.
    """
    future = collections.defaultdict(list)
    for index, (_, _, layer, experts) in enumerate(sequence):
        for expert in experts:
            future[(layer, expert)].append(index)
    cursor = collections.defaultdict(int)
    never = len(sequence) + 1

    def next_use(key, index):
        uses = future[key]
        at = cursor[key]
        while at < len(uses) and uses[at] <= index:
            at += 1
        cursor[key] = at
        return uses[at] if at < len(uses) else never

    pools = [{} for _ in range(1 if global_pool else LAYERS)]  # key -> next use
    heaps = [[] for _ in pools]
    used = [0] * len(pools)
    for index, (phase, token, layer, experts) in enumerate(sequence):
        pool_id = 0 if global_pool else layer
        pool, heap = pools[pool_id], heaps[pool_id]
        capacity = capacities if global_pool else capacities[layer] * strides[layer]
        pinned = {(layer, expert) for expert in experts}
        for expert in experts:
            key = (layer, expert)
            hit = key in pool
            stats.note(phase, token, hit, strides[layer])
            if not hit:
                while used[pool_id] + strides[layer] > capacity:
                    skipped = []
                    victim = None
                    while heap:
                        distance, candidate = heapq.heappop(heap)
                        if pool.get(candidate) != -distance:
                            continue  # stale entry
                        if candidate in pinned:
                            skipped.append((distance, candidate))
                            continue
                        victim = candidate
                        break
                    for entry in skipped:
                        heapq.heappush(heap, entry)
                    if victim is None:
                        raise ValueError("cache cannot hold a complete route")
                    del pool[victim]
                    used[pool_id] -= strides[victim[0]]
                used[pool_id] += strides[layer]
            upcoming = next_use(key, index)
            pool[key] = upcoming
            heapq.heappush(heap, (-upcoming, key))


def miss_curves(sequence):
    """Per-layer LRU decode misses for every slot count (Mattson stack)."""
    stacks = [[] for _ in range(LAYERS)]
    depth = [[0] * (EXPERTS + 2) for _ in range(LAYERS)]
    cold = [0] * LAYERS
    for phase, _, layer, experts in sequence:
        stack = stacks[layer]
        for expert in experts:
            try:
                position = stack.index(expert)
            except ValueError:
                position = None
            if position is None:
                if phase == "decode":
                    cold[layer] += 1
            else:
                if phase == "decode":
                    depth[layer][position + 1] += 1
                del stack[position]
            stack.insert(0, expert)
    curves = []
    for layer in range(LAYERS):
        misses = [0] * (EXPERTS + 1)
        beyond = sum(depth[layer])
        for count in range(EXPERTS + 1):
            beyond -= depth[layer][count]
            misses[count] = cold[layer] + beyond
        curves.append(misses)
    return curves


def weighted_slots(strides, budget, curves, grant=2):
    """Give slots to the layers whose LRU miss curves gain the most per byte.

    Curves come from calibration traces. A layer's next grant is valued by its
    best average misses saved per slot over any further extent, i.e. the next
    segment of the curve's convex hull, so flat stretches do not stall the
    allocation. Gains within ten percent count as equal and go to the layer
    holding fewer slots: calibration noise then spreads the budget instead of
    handing one layer a whole segment.
    """
    slots = [TOP_K] * LAYERS
    used = sum(stride * TOP_K for stride in strides)
    if used > budget:
        return None

    def push(heap, layer):
        have = slots[layer]
        affordable = min(EXPERTS - have, (budget - used) // strides[layer])
        base = curves[layer][have]
        best = 0.0
        for extra in range(1, affordable + 1):
            best = max(best, (base - curves[layer][have + extra]) / extra)
        if best > 0:
            bucket = math.floor(math.log(best) / math.log(1.1))
            heapq.heappush(heap, (-bucket, have, layer))

    heap = []
    for layer in range(LAYERS):
        push(heap, layer)
    while heap:
        _, have, layer = heapq.heappop(heap)
        if have == slots[layer]:
            take = min(grant, EXPERTS - have, (budget - used) // strides[layer])
            slots[layer] += take
            used += take * strides[layer]
        # Re-value from the current state; an unaffordable layer drops out.
        push(heap, layer)
    # Spend what is left evenly: it cannot hurt and keeps the comparison at
    # equal bytes.
    progress = True
    while progress:
        progress = False
        for layer, stride in enumerate(strides):
            if slots[layer] < EXPERTS and stride <= budget - used:
                slots[layer] += 1
                used += stride
                progress = True
    return slots


def replay_policy(name, sequence, strides, budget, warm_from, curves=None):
    stats = Stats(warm_from)
    slots = equal_slots(strides, budget)
    if slots is None:
        return None
    if name == "lfu_decay_slots":
        replay_slots(sequence, strides, slots, stats, expert_keyed=False)
    elif name == "lfu_ghost":
        replay_slots(sequence, strides, slots, stats, expert_keyed=True)
    elif name == "lfu_ghost_weighted":
        if curves is None:
            return None
        slots = weighted_slots(strides, budget, curves)
        replay_slots(sequence, strides, slots, stats, expert_keyed=True)
    elif name == "lru_per_layer":
        replay_lru(sequence, strides, slots, stats, global_pool=False)
    elif name == "lru_global":
        replay_lru(sequence, strides, budget, stats, global_pool=True)
    elif name == "belady_per_layer":
        replay_belady(sequence, strides, slots, stats, global_pool=False)
    elif name == "belady_global":
        replay_belady(sequence, strides, budget, stats, global_pool=True)
    else:
        raise ValueError(f"unknown policy {name}")
    result = stats.rows
    if name not in ("lru_global", "belady_global"):
        result["slots_min_max"] = [min(slots), max(slots)]
    return result


def summarize(row, tokens):
    accesses = row["hits"] + row["misses"]
    return {
        "hit_rate": row["hits"] / accesses if accesses else None,
        "bytes_per_token": row["bytes"] / tokens if tokens else None,
        "misses_per_token": row["misses"] / tokens if tokens else None,
    }


def load_workload(entry, base):
    routes = load_routes(base / entry["trace"])
    if "receipt" in entry:
        receipt = json.loads((base / entry["receipt"]).read_text())
        segments = receipt_segments(receipt)
    else:
        log = json.loads((base / entry["session"]).read_text())
        receipt = log
        segments = session_segments(log["turns"])
    if len(routes) != segments[-1][2] * LAYERS:
        raise ValueError(f"{entry['name']}: trace and segment lengths differ")
    return {
        "name": entry["name"],
        "routes": routes,
        "segments": segments,
        "strides": receipt["expert_strides"],
        "chunk": receipt.get("prefill_chunk", 0),
        "read_group": receipt.get("prefill_read_group", TOP_K),
        "session": "session" in entry,
        "measured": {
            "prefill": receipt.get("prefill_expert_io"),
            "decode": receipt.get("decode_expert_io"),
            "slots": receipt.get("expert_slot_counts"),
        },
    }


def sweep(workloads, budgets_gb, policies, warm_fraction):
    report = {
        "workloads": {},
        "budgets_gb": list(budgets_gb),
        "policies": list(policies),
    }
    prepared = []
    for work in workloads:
        sequence = list(
            acquisitions(
                work["routes"],
                work["segments"],
                work["chunk"],
                work["read_group"],
                work["session"],
            )
        )
        decode = [
            token
            for phase, token, layer, _ in sequence
            if phase == "decode" and layer == 0
        ]
        warm_from = decode[int(len(decode) * warm_fraction)] if decode else 0
        prepared.append(
            (
                work,
                sequence,
                len(decode),
                sum(1 for t in decode if t >= warm_from),
                warm_from,
            )
        )
    curves = [miss_curves(sequence) for _, sequence, *_ in prepared]
    for index, (work, sequence, decode_tokens, warm_tokens, warm_from) in enumerate(
        prepared
    ):
        # Calibrate the weighted allocation on the other workloads only.
        others = [c for i, c in enumerate(curves) if i != index]
        held_out = (
            [
                [sum(c[layer][n] for c in others) for n in range(EXPERTS + 1)]
                for layer in range(LAYERS)
            ]
            if others
            else None
        )
        distinct = [
            len({e for phase, _, l, ids in sequence if l == layer for e in ids})
            for layer in range(LAYERS)
        ]
        entry = {
            "decode_tokens": decode_tokens,
            "warm_decode_tokens": warm_tokens,
            "prefill_tokens": sum(
                e - s for phase, s, e in work["segments"] if phase == "prefill"
            ),
            "distinct_experts_per_layer": {
                "min": min(distinct),
                "mean": sum(distinct) / LAYERS,
                "max": max(distinct),
            },
            "all_miss_bytes_per_token": sum(work["strides"]) * TOP_K,
            "results": {},
        }
        measured = work["measured"]
        if measured["decode"] and measured["slots"]:
            # The runtime's policy at the recorded slot counts must reproduce
            # the receipt's hits and misses; otherwise the model of
            # acquisitions has drifted from the executor and nothing below
            # should be trusted.
            stats = Stats(warm_from)
            replay_slots(sequence, work["strides"], measured["slots"], stats, False)
            entry["model_check"] = {
                phase: {
                    "measured": [measured[phase]["hits"], measured[phase]["misses"]],
                    "replayed": [
                        stats.rows[phase]["hits"],
                        stats.rows[phase]["misses"],
                    ],
                }
                for phase in ("prefill", "decode")
            }
            entry["model_check"]["exact"] = all(
                row["measured"] == row["replayed"]
                for row in entry["model_check"].values()
            )
        for budget_gb in budgets_gb:
            budget = int(budget_gb * 1e9)
            row = {}
            for name in policies:
                result = replay_policy(
                    name, sequence, work["strides"], budget, warm_from, held_out
                )
                if result is None:
                    row[name] = None
                    continue
                row[name] = {
                    "decode": summarize(result["decode"], decode_tokens),
                    "decode_warm": summarize(result["decode_warm"], warm_tokens),
                    "prefill_bytes": result["prefill"]["bytes"],
                    "slots_min_max": result.get("slots_min_max"),
                }
            entry["results"][budget_key(budget_gb)] = row
        report["workloads"][work["name"]] = entry
    return report


def print_table(report, budget_gb):
    key = budget_key(budget_gb)
    print(
        f"\nPredicted decode I/O at a {budget_gb} GB expert cache (MB read per token; warm = last part of decode)"
    )
    header = f"{'workload':<12}{'policy':<22}{'hit rate':>9}{'MB/token':>10}{'warm hit':>10}{'warm MB/tok':>12}"
    print(header)
    for name, entry in report["workloads"].items():
        row = entry["results"].get(key)
        if row is None:
            continue
        for policy, result in row.items():
            if result is None:
                print(f"{name:<12}{policy:<22}{'n/a':>9}")
                continue
            d, w = result["decode"], result["decode_warm"]
            print(
                f"{name:<12}{policy:<22}{d['hit_rate']:>9.3f}{d['bytes_per_token'] / 1e6:>10.1f}"
                f"{w['hit_rate']:>10.3f}{w['bytes_per_token'] / 1e6:>12.1f}"
            )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("trace", type=Path, nargs="?")
    p.add_argument("receipt", type=Path, nargs="?")
    p.add_argument("--manifest", type=Path, help="record_routes.py manifest.json")
    p.add_argument("--budgets-gb", type=float, nargs="+", default=DEFAULT_BUDGETS_GB)
    p.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    p.add_argument("--report-gb", type=float, default=30)
    p.add_argument(
        "--warm-fraction",
        type=float,
        default=0.5,
        help="decode tokens before this fraction are warm-up for the warm figures",
    )
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.manifest:
        if a.trace or a.receipt:
            p.error("pass either TRACE RECEIPT or --manifest")
        if not 0 <= a.warm_fraction < 1:
            p.error("warm-fraction must be in [0, 1)")
        manifest = json.loads(a.manifest.read_text())
        workloads = [
            load_workload(entry, a.manifest.parent) for entry in manifest["workloads"]
        ]
        report = sweep(workloads, a.budgets_gb, a.policies, a.warm_fraction)
        report["source"] = {k: v for k, v in manifest.items() if k != "workloads"}
        report["note"] = (
            "Offline I/O model on recorded routes. Belady policies are offline bounds; "
            "lfu_decay_slots is the runtime's policy. Not measured throughput."
        )
        text = json.dumps(report, indent=2) + "\n"
        if a.output:
            a.output.write_text(text)
        for budget in sorted({a.report_gb} & set(a.budgets_gb)) or [max(a.budgets_gb)]:
            print_table(report, budget)
        if not a.output:
            print(text)
        return
    if not a.trace or not a.receipt:
        p.error("TRACE and RECEIPT are required without --manifest")
    d = json.loads(a.receipt.read_text())
    if isinstance(d["expert_slot_counts"], str):
        d["expert_slot_counts"] = list(map(ord, d["expert_slot_counts"]))
    routes = load_routes(a.trace)
    if len(routes) != (d["prompt_tokens"] + d["decode_steps"]) * LAYERS:
        p.error("receipt and trace lengths differ")
    sequence = list(
        acquisitions(
            routes,
            receipt_segments(d),
            d.get("prefill_chunk", 0),
            d.get("prefill_read_group", 1),
            session=False,
        )
    )
    result = {
        "measured": {"prefill": d["prefill_expert_io"], "decode": d["decode_expert_io"]}
    }
    budget = sum(a * b for a, b in zip(d["expert_strides"], d["expert_slot_counts"]))
    for name, global_pool in (("per_layer_lru", False), ("global_byte_lru", True)):
        stats = Stats(0)
        replay_lru(
            sequence,
            d["expert_strides"],
            budget if global_pool else d["expert_slot_counts"],
            stats,
            global_pool,
        )
        result[name] = {k: stats.rows[k] for k in ("prefill", "decode")}
    # The runtime's own policy at the receipt's slot counts should reproduce
    # the measured hits and misses exactly; a difference means the model of
    # acquisitions has drifted from the executor.
    stats = Stats(0)
    replay_slots(
        sequence,
        d["expert_strides"],
        d["expert_slot_counts"],
        stats,
        expert_keyed=False,
    )
    result["runtime_policy_model"] = {k: stats.rows[k] for k in ("prefill", "decode")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
