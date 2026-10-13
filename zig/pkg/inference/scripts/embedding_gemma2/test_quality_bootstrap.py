#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The faster quality gate must preserve the reviewed statistical test."""

import unittest

from qualify_finetune_quality import bootstrap_delta, bootstrap_deltas


class BootstrapTests(unittest.TestCase):
    def assert_reference_bounds(self, pairs, seed=20261008):
        expected = [bootstrap_delta(candidate, base, seed) for candidate, base in pairs]
        self.assertEqual(bootstrap_deltas(pairs, seed), expected)

    def test_all_draws_match_for_sparse_dense_and_cancelling_deltas(self):
        self.assert_reference_bounds([
            ([.6] * 20, [.5] * 20),
            ([.1, .2, 1., 0., .7] * 11, [.1, .3, 0., 1., .5] * 11),
            ([0., 1e-100, .5, .9] * 23, [0., 0., .5, .91] * 23),
            ([.1, .2, .3, .7, .8, .9] * 7, [.9, .8, .7, .3, .2, .1] * 7),
        ])

    def test_mixed_query_counts_and_custom_seed_keep_input_order(self):
        first = ([.1, .5, .9] * 13, [.2, .5, .8] * 13)
        self.assert_reference_bounds([first, ([1., 0.], [0., 1.]), first], seed=43)

    def test_zero_differences_and_single_queries_preserve_the_denominator(self):
        self.assert_reference_bounds([
            ([.2, .7] * 9, [.2, .7] * 9),
            ([.7], [.2]), ([.2], [.7]),
            ([1.] + [0.] * 99, [0.] * 100),
        ])

    def test_no_metrics_and_invalid_pairs(self):
        self.assertEqual(bootstrap_deltas([]), [])
        for pair in (([], []), ([.5], []), ([float('nan')], [0.]),
                     ([.5], [float('inf')]), (['bad'], [.5])):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                bootstrap_deltas([pair])


if __name__ == '__main__':
    unittest.main()
