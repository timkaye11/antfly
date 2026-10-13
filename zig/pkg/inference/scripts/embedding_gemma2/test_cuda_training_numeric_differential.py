# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

import unittest

from cuda_training_numeric_differential import compare


class NumericDifferentialTests(unittest.TestCase):
    def test_finite_rounding_noise_passes_but_a_wrong_gradient_fails(self):
        self.assertTrue(compare([1.0000001, -1.0000001, 0.], [1., -1., 0.])['pass'])
        self.assertFalse(compare([1., -1., .0001], [1., -1., 0.])['pass'])

    def test_nonfinite_actual_or_reference_cannot_pass(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            for actual, expected in (([value], [0.]), ([0.], [value])):
                with self.subTest(actual=actual, expected=expected), self.assertRaises(ValueError):
                    compare(actual, expected)

    def test_broadcasting_cannot_hide_wrong_output_shapes(self):
        with self.assertRaises(ValueError):
            compare([[1., 1.]], [1., 1.])


if __name__ == '__main__':
    unittest.main()
