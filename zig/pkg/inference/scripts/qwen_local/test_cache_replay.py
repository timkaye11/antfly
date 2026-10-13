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

import json
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

# Appended: this directory's modules keep precedence over same-named ones there.
sys.path.append(str(Path(__file__).resolve().parent.parent / "qwen4exp"))

import cache_replay as replay

STRIDES = [3_000_000 + 1000 * layer for layer in range(replay.LAYERS)]


def synthetic_routes(tokens, seed=7, hot=40):
    """Skewed routes: most picks come from a small per-layer hot set."""
    rng = random.Random(seed)
    routes = []
    for _ in range(tokens):
        for layer in range(replay.LAYERS):
            chosen = set()
            while len(chosen) < replay.TOP_K:
                if rng.random() < 0.8:
                    chosen.add((layer * 7 + rng.randrange(hot)) % replay.EXPERTS)
                else:
                    chosen.add(rng.randrange(replay.EXPERTS))
            routes.append(tuple(chosen))
    return routes


def decode_misses(name, sequence, budget, curves=None):
    return replay.replay_policy(name, sequence, STRIDES, budget, 0, curves)["decode"]


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.routes = synthetic_routes(60)
        self.segments = [("prefill", 0, 20), ("decode", 20, 60)]
        self.sequence = list(replay.acquisitions(self.routes, self.segments, 8, 10))
        self.accesses = 40 * replay.LAYERS * replay.TOP_K

    def test_acquisitions_follow_runtime_order(self):
        # Chunked prefill is layer-major with sorted unions in groups of ten.
        prefill = [row for row in self.sequence if row[0] == "prefill"]
        first = [row for row in prefill if row[1] == 0 and row[2] == 0]
        union = sorted({e for t in range(8) for e in self.routes[t * replay.LAYERS]})
        self.assertEqual([e for row in first for e in row[3]], union)
        self.assertTrue(all(len(row[3]) <= 10 for row in first))
        self.assertEqual(next(row[2] for row in prefill if row[1] == 0), 0)
        # Decode acquires each token's routes in router order, layer by layer.
        decode = [row for row in self.sequence if row[0] == "decode"]
        self.assertEqual(len(decode), 40 * replay.LAYERS)
        self.assertEqual(decode[0][1:], (20, 0, self.routes[20 * replay.LAYERS]))
        self.assertEqual(decode[49][1:3], (21, 1))
        # A session sends a one-token chunk down the token path (router order);
        # the one-shot CLI runs a trailing single token as a sorted chunk.
        single = list(replay.acquisitions(self.routes, [("prefill", 0, 9)], 8, 10))
        tail = [row for row in single if row[1] == 8]
        self.assertEqual(len(tail), replay.LAYERS)
        self.assertEqual(tail[0][3], self.routes[8 * replay.LAYERS])
        oneshot = list(
            replay.acquisitions(self.routes, [("prefill", 0, 9)], 8, 10, session=False)
        )
        tail = [row for row in oneshot if row[1] == 8]
        self.assertEqual(tail[0][3], tuple(sorted(self.routes[8 * replay.LAYERS])))
        lone = list(
            replay.acquisitions(self.routes, [("prefill", 0, 1)], 8, 10, session=False)
        )
        self.assertEqual(lone[0][3], self.routes[0])

    def test_equal_slots_match_the_runtime_allocation(self):
        self.assertIsNone(replay.equal_slots(STRIDES, sum(STRIDES) * 10 - 1))
        self.assertEqual(replay.equal_slots(STRIDES, sum(STRIDES) * 10), [10] * 48)
        slots = replay.equal_slots(STRIDES, sum(STRIDES) * 10 + STRIDES[0] + STRIDES[1])
        self.assertEqual(slots[:3], [11, 11, 10])
        full = replay.equal_slots(STRIDES, sum(STRIDES) * 1000)
        self.assertEqual(full, [replay.EXPERTS] * 48)
        # Beyond the old 128-slot ceiling on a large budget.
        large = replay.equal_slots(STRIDES, int(30e9))
        self.assertTrue(all(128 < count <= 512 for count in large))
        self.assertLessEqual(sum(a * b for a, b in zip(large, STRIDES)), int(30e9))

    def test_policies_account_every_access_and_respect_offline_bounds(self):
        budget = sum(STRIDES) * 16
        curves = replay.miss_curves(self.sequence)
        results = {
            name: decode_misses(name, self.sequence, budget, curves)
            for name in replay.POLICIES
        }
        for name, row in results.items():
            self.assertEqual(row["hits"] + row["misses"], self.accesses, name)
            self.assertGreater(row["misses"], 0, name)
        # Furthest-next-use is optimal within a layer for every online policy.
        for name in ("lru_per_layer", "lfu_decay_slots", "lfu_ghost"):
            self.assertLessEqual(
                results["belady_per_layer"]["misses"], results[name]["misses"], name
            )
        # Pooling capacity can only help the offline bound.
        self.assertLessEqual(
            results["belady_global"]["misses"], results["belady_per_layer"]["misses"]
        )

    def test_full_residency_leaves_only_first_touches(self):
        budget = sum(STRIDES) * replay.EXPERTS
        seen = [set() for _ in range(replay.LAYERS)]
        cold = 0
        for phase, _, layer, experts in self.sequence:
            for expert in experts:
                if expert not in seen[layer]:
                    seen[layer].add(expert)
                    cold += phase == "decode"
        for name in replay.POLICIES:
            row = decode_misses(
                name, self.sequence, budget, replay.miss_curves(self.sequence)
            )
            self.assertEqual(row["misses"], cold, name)

    def test_minimum_cache_misses_every_new_route_expert(self):
        # One route per layer: a decode step hits only experts repeated from
        # the acquisition just before it in the same layer.
        budget = sum(STRIDES) * replay.TOP_K
        sequence = list(replay.acquisitions(self.routes, [("decode", 0, 60)]))
        row = decode_misses("lru_per_layer", sequence, budget)
        expected = 0
        previous = {}
        for _, _, layer, experts in sequence:
            expected += len(set(experts) - previous.get(layer, set()))
            previous[layer] = set(experts)
        self.assertEqual(row["misses"], expected)
        self.assertIsNone(
            replay.replay_policy("lru_per_layer", sequence, STRIDES, budget - 1, 0)
        )

    def test_runtime_policy_keeps_frequent_experts_and_ghosts_remember(self):
        # Layer 0 only: A is used often, then B.. displace others, then A returns.
        def route(*ids):
            filler = [e for e in range(400, 400 + replay.TOP_K) if e not in ids]
            return tuple(list(ids) + filler)[: replay.TOP_K]

        hot = route(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        sequence = [("decode", t, 0, hot) for t in range(5)]
        sequence += [("decode", 5, 0, route(*range(20, 30)))]
        sequence += [("decode", 6, 0, route(*range(30, 40)))]
        sequence += [("decode", 7, 0, hot)]
        for expert_keyed in (False, True):
            stats = replay.Stats(0)
            slots = [20] + [10] * 47
            replay.replay_slots(sequence, STRIDES, slots, stats, expert_keyed)
            # Twenty slots: the hot ten (five uses each) survive both visitors.
            self.assertEqual(stats.rows["decode"]["misses"], 30)
        # With twelve slots the hot route is partly evicted. Slot-keyed counts
        # restart at one on reinstall; ghost counts keep the earlier uses.
        tail = sequence + [
            ("decode", 8, 0, route(*range(20, 30))),
            ("decode", 9, 0, hot),
        ]
        misses = {}
        for expert_keyed in (False, True):
            stats = replay.Stats(0)
            replay.replay_slots(tail, STRIDES, [12] + [10] * 47, stats, expert_keyed)
            misses[expert_keyed] = stats.rows["decode"]["misses"]
        self.assertLessEqual(misses[True], misses[False])

    def test_weighted_allocation_stays_in_budget_and_follows_demand(self):
        # Layers 0-23 route within ten experts; 24-47 across four hundred.
        rng = random.Random(3)
        routes = []
        for _ in range(80):
            for layer in range(replay.LAYERS):
                pool = range(10) if layer < 24 else range(400)
                routes.append(tuple(rng.sample(pool, replay.TOP_K)))
        sequence = list(replay.acquisitions(routes, [("decode", 0, 80)]))
        curves = replay.miss_curves(sequence)
        budget = sum(STRIDES) * 40
        slots = replay.weighted_slots(STRIDES, budget, curves)
        self.assertLessEqual(sum(a * b for a, b in zip(slots, STRIDES)), budget)
        self.assertTrue(all(replay.TOP_K <= count <= replay.EXPERTS for count in slots))
        # Layers that never leave one route get nothing beyond it; the budget
        # goes to the layers whose curves still fall.
        self.assertEqual(slots[:24], [replay.TOP_K] * 24)
        self.assertGreater(sum(slots[24:]), 6 * sum(slots[:24]))
        equal = decode_misses("lfu_ghost", sequence, budget)
        weighted = decode_misses("lfu_ghost_weighted", sequence, budget, curves)
        self.assertLess(weighted["misses"], equal["misses"])
        # The curves model one access at a time; the replay pins a whole
        # route first, which can only spare a few evictions.
        stats = replay.Stats(0)
        replay.replay_lru(sequence, STRIDES, [40] * 48, stats, global_pool=False)
        modelled = sum(c[40] for c in curves)
        self.assertLessEqual(stats.rows["decode"]["misses"], modelled)
        self.assertGreater(stats.rows["decode"]["misses"], 0.97 * modelled)

    def test_model_check_flags_a_replay_that_disagrees_with_the_receipt(self):
        slots = [12] * replay.LAYERS
        sequence = list(
            replay.acquisitions(self.routes, self.segments, 8, 10, session=False)
        )
        stats = replay.Stats(0)
        replay.replay_slots(sequence, STRIDES, slots, stats, expert_keyed=False)
        io = {
            phase: {
                "hits": stats.rows[phase]["hits"],
                "misses": stats.rows[phase]["misses"],
            }
            for phase in ("prefill", "decode")
        }
        work = {
            "name": "chat",
            "routes": self.routes,
            "segments": self.segments,
            "strides": STRIDES,
            "chunk": 8,
            "read_group": 10,
            "session": False,
            "measured": {
                "prefill": io["prefill"],
                "decode": io["decode"],
                "slots": slots,
            },
        }
        report = replay.sweep([work], (2.0,), ("lru_per_layer",), 0.5)
        self.assertTrue(report["workloads"]["chat"]["model_check"]["exact"])
        work["measured"]["decode"] = {"hits": io["decode"]["hits"] + 1, "misses": 0}
        report = replay.sweep([work], (2.0,), ("lru_per_layer",), 0.5)
        self.assertFalse(report["workloads"]["chat"]["model_check"]["exact"])

    def test_session_segments_and_manifest_sweep(self):
        turns = [
            {"reused_tokens": 0, "prefilled_tokens": 12, "position": 30},
            {"reused_tokens": 30, "prefilled_tokens": 6, "position": 60},
        ]
        self.assertEqual(
            replay.session_segments(turns),
            [
                ("prefill", 0, 12),
                ("decode", 12, 30),
                ("prefill", 30, 36),
                ("decode", 36, 60),
            ],
        )
        with self.assertRaises(ValueError):
            replay.session_segments([turns[0], {**turns[1], "reused_tokens": 0}])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = b"".join(struct.pack("<10H", *ids) for ids in self.routes)
            (root / "chat.u16").write_bytes(raw)
            (root / "session.u16").write_bytes(raw)
            receipt = {
                "prompt_tokens": 20,
                "decode_steps": 40,
                "expert_strides": STRIDES,
                "prefill_chunk": 8,
                "prefill_read_group": 10,
            }
            (root / "chat.json").write_text(json.dumps(receipt))
            log = {"turns": turns, "expert_strides": STRIDES, "prefill_chunk": 8}
            (root / "session.json").write_text(json.dumps(log))
            entries = [
                {"name": "chat", "trace": "chat.u16", "receipt": "chat.json"},
                {"name": "session", "trace": "session.u16", "session": "session.json"},
            ]
            workloads = [replay.load_workload(entry, root) for entry in entries]
            budgets = (0.1, 1.6, 4.0)
            report = replay.sweep(workloads, budgets, replay.POLICIES, 0.5)
            self.assertEqual(report["workloads"]["chat"]["decode_tokens"], 40)
            # No measured counters in this receipt, so no model check is claimed.
            self.assertNotIn("model_check", report["workloads"]["chat"])
            self.assertEqual(report["workloads"]["session"]["decode_tokens"], 42)
            self.assertEqual(report["workloads"]["chat"]["warm_decode_tokens"], 20)
            # Below one route per layer nothing can run.
            self.assertTrue(
                all(
                    v is None
                    for v in report["workloads"]["chat"]["results"]["0.1"].values()
                )
            )
            # A budget typed as 4 or 4.0 names the same row.
            self.assertEqual(replay.budget_key(4), replay.budget_key(4.0))
            self.assertIn("4", report["workloads"]["chat"]["results"])
            for name in replay.POLICIES:
                series = [
                    report["workloads"]["chat"]["results"][replay.budget_key(b)][name][
                        "decode"
                    ]["bytes_per_token"]
                    for b in budgets[1:]
                ]
                self.assertLessEqual(series[1], series[0], name)
            (root / "bad.u16").write_bytes(raw[:-2])
            with self.assertRaises(ValueError):
                replay.load_routes(root / "bad.u16")
            (root / "short.json").write_text(
                json.dumps({**receipt, "decode_steps": 39})
            )
            with self.assertRaises(ValueError):
                replay.load_workload(
                    {"name": "x", "trace": "chat.u16", "receipt": "short.json"}, root
                )


if __name__ == "__main__":
    unittest.main()
