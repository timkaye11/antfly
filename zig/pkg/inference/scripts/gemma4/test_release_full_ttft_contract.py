import unittest

import release_attention_matching as release


class FullTtftContract(unittest.TestCase):
    def attempt(self, variant, phases=None):
        names = phases or list(sorted(release.FULL_TTFT_PHASES[variant]))
        start = 1_000_000_000
        intervals = []
        cursor = start
        for name in names:
            intervals.append({"name": name, "start_ns": cursor, "end_ns": cursor + 1_000_000,
                              "authority": "client_monotonic_boundary"})
            cursor += 1_000_000
        return {
            "ttft_ms": float(len(names)),
            "full_ttft_authority": {
                "schema": "antfly.attention-matching-full-ttft-attempt.v1",
                "clock": "posix_CLOCK_MONOTONIC",
                "timing_scope": "full_request_pipeline_to_first_public_content",
                "outer_start_ns": start,
                "first_public_content_ns": cursor,
                "first_public_content_kind": "nonempty_public_delta",
                "phases": intervals,
                "fresh_owner_id": "owner-" + variant,
                "fresh_artifact_id": "artifact-" + variant,
                "restore_ttft_ms": 1.0 if variant == "attention_matching" else None,
            },
        }

    def test_all_variants_require_closed_full_pipeline_authority(self):
        for variant in release.PERFORMANCE_VARIANTS:
            with self.subTest(variant=variant):
                self.assertEqual(("owner-" + variant, "artifact-" + variant),
                                 release.full_ttft_attempt_contract(self.attempt(variant), variant))

    def test_old_numeric_restore_only_ttft_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing full-TTFT timing authority"):
            release.full_ttft_attempt_contract({"ttft_ms": 1.0}, "attention_matching")

    def test_gap_and_missing_fit_phase_are_rejected(self):
        attempt = self.attempt("attention_matching")
        attempt["full_ttft_authority"]["phases"][1]["start_ns"] += 1
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            release.full_ttft_attempt_contract(attempt, "attention_matching")
        names = release.FULL_TTFT_PHASES["attention_matching"] - {"fit_validation"}
        with self.assertRaisesRegex(ValueError, "incomplete full-TTFT phase closure"):
            release.full_ttft_attempt_contract(self.attempt("attention_matching", sorted(names)),
                                                "attention_matching")

    def test_reported_ttft_must_equal_outer_monotonic_interval(self):
        attempt = self.attempt("packed_exact")
        attempt["ttft_ms"] += .001
        with self.assertRaisesRegex(ValueError, "differs from authoritative outer clock"):
            release.full_ttft_attempt_contract(attempt, "packed_exact")


if __name__ == "__main__":
    unittest.main()
