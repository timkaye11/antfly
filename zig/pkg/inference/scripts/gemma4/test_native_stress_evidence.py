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

import copy
import unittest
from unittest.mock import patch

import qualify_native_stress as stress

from qualify_native_stress import assert_equivalent, summarize_blocks, TEN_REQUESTS


def completed_blocks(metal=False):
    return [
        {
            "passed": True,
            "operations": [
                {
                    "kind": kind,
                    "passed": True,
                    **({"outcome": "compacted"} if kind == "prepare" else {}),
                }
                for kind in TEN_REQUESTS * 5
            ],
            "quiescent_memory": ({
                "group_rss_bytes": 5 * 1024**3,
                "free_memory_percent": 42.0,
                "swapout_bytes": 0,
            } if metal else {
                "host_rss_bytes": 5 * 1024**3,
                "gpu_memory_bytes": 11 * 1024**3,
            }),
        }
        for _ in range(20)
    ]


class StressMemoryObservationTests(unittest.TestCase):
    def observe(self, record, group, rss=2048):
        with (
            patch("os.getpgid", return_value=group) as getpgid,
            patch.object(stress, "group_rss_mib", return_value=rss) as probe,
            patch.object(stress, "memory_free_percent", return_value=50),
            patch.object(stress, "swapout_bytes", return_value=0),
        ):
            row = stress.memory_observation(record, "metal")
            getpgid.assert_called_once_with(record["worker_pid"])
            probe.assert_called_once_with(group)
            return row

    def test_shared_guard_group_is_measured_instead_of_worker_pid(self):
        record = {"worker_pid": 201, "process_topology": {"shared_pgid": True, "pgid": 101}}
        self.assertEqual(self.observe(record, 101)["group_rss_bytes"], 2 * 1024**3)

    def test_independent_worker_group_still_uses_its_own_pid(self):
        record = {"worker_pid": 201, "process_topology": {"shared_pgid": False}}
        self.assertEqual(self.observe(record, 201)["group_rss_bytes"], 2 * 1024**3)

    def test_worker_group_migration_is_rejected(self):
        record = {"worker_pid": 201, "process_topology": {"shared_pgid": True, "pgid": 101}}
        with patch("os.getpgid", return_value=202), self.assertRaisesRegex(AssertionError, "group changed"):
            stress.memory_observation(record, "metal")

    def test_missing_group_rss_cannot_pass_the_memory_cap(self):
        with self.assertRaisesRegex(AssertionError, "RSS cap exceeded"):
            self.observe({"worker_pid": 201}, 201, rss=0)


class StressEvidenceTests(unittest.TestCase):
    def test_metal_bounded_profile_proves_100_mixed_and_10_independent_compactions(self):
        result = summarize_blocks(completed_blocks(metal=True)[:2], "bounded")
        self.assertEqual(result["mixed_requests"], 100)
        self.assertEqual(result["actual_compactions"], 10)
        self.assertEqual(result["independent_repreparations"], 10)
        self.assertFalse(result["successive_compaction_cycles_qualified"])
        self.assertIn("public checkpoint protocol", result["successive_compaction_reason"])
        self.assertEqual(set(result["quiescent_growth"]), {"group_rss_bytes"})

    def test_metal_rejects_inconsistent_or_growing_quiescent_evidence(self):
        blocks = completed_blocks(metal=True)[:2]
        blocks[1]["quiescent_memory"].pop("swapout_bytes")
        with self.assertRaisesRegex(AssertionError, "inconsistent"):
            summarize_blocks(blocks, "bounded")
        blocks = completed_blocks(metal=True)[:2]
        blocks[1]["quiescent_memory"]["group_rss_bytes"] += 129 * 1024**2
        with self.assertRaisesRegex(AssertionError, "quiescent memory growth"):
            summarize_blocks(blocks, "bounded")

    def test_bounded_profile_reports_its_actual_scope(self):
        blocks = completed_blocks()[:2]
        result = summarize_blocks(blocks, "bounded")
        self.assertEqual(result["mixed_requests"], 100)
        self.assertEqual(result["actual_compactions"], 10)
        self.assertFalse(result["extended_run_qualified"])
        self.assertEqual(
            result["quiescent_growth"]["host_rss_bytes"]["window_blocks"], 1
        )
        with self.assertRaises(AssertionError):
            summarize_blocks(blocks)
        with self.assertRaises(AssertionError):
            summarize_blocks(blocks[:1], "bounded")
        blocks[-1]["quiescent_memory"]["host_rss_bytes"] += 129 * 1024**2
        with self.assertRaisesRegex(AssertionError, "quiescent memory growth"):
            summarize_blocks(blocks, "bounded")

    def test_bounded_profile_rejects_missing_work_and_fallbacks(self):
        blocks = completed_blocks()[:2]
        blocks[0]["operations"].pop()
        with self.assertRaises(AssertionError):
            summarize_blocks(blocks, "bounded")
        blocks = completed_blocks()[:2]
        blocks[0]["operations"][0]["outcome"] = "unchanged_insufficient_savings"
        with self.assertRaisesRegex(AssertionError, "actual compactions"):
            summarize_blocks(blocks, "bounded")

    def test_only_complete_campaign_counts(self):
        blocks = completed_blocks()
        result = summarize_blocks(blocks)
        self.assertEqual(result["mixed_requests"], 1000)
        self.assertEqual(result["actual_compactions"], 100)
        with self.assertRaises(AssertionError):
            summarize_blocks(blocks[:-1])
        blocks[-1]["operations"].pop()
        with self.assertRaises(AssertionError):
            summarize_blocks(blocks)

    def test_unchanged_fallback_cannot_satisfy_compaction_count(self):
        blocks = completed_blocks()
        blocks[7]["operations"][0]["outcome"] = "unchanged_insufficient_savings"
        with self.assertRaisesRegex(AssertionError, "actual compactions"):
            summarize_blocks(blocks)

    def test_failed_request_cannot_be_hidden_by_later_recovery(self):
        blocks = completed_blocks()
        blocks[7]["operations"][3]["passed"] = False
        with self.assertRaisesRegex(AssertionError, "successful expected outcomes"):
            summarize_blocks(blocks)

    def test_growth_below_hard_cap_still_fails_diagnostic(self):
        blocks = completed_blocks()
        for block in blocks[-5:]:
            block["quiescent_memory"]["host_rss_bytes"] += 129 * 1024**2
        with self.assertRaisesRegex(
            AssertionError, "sustained quiescent memory growth"
        ):
            summarize_blocks(blocks)

    def test_changed_usage_fails_even_if_answer_is_identical(self):
        expected = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "violet-729"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 8189,
                "cached_prompt_tokens": 8189,
                "completion_tokens": 5,
            },
        }
        actual = copy.deepcopy(expected)
        assert_equivalent(actual, expected)
        actual["usage"]["cached_prompt_tokens"] = 0
        with self.assertRaisesRegex(AssertionError, "token accounting"):
            assert_equivalent(actual, expected)


if __name__ == "__main__":
    unittest.main()
