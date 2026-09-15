import math
import unittest

from gemma4_drift_metrics import compare_vectors


class DriftMetricsTests(unittest.TestCase):
    def test_equal_norms_do_not_imply_equal_vectors(self):
        result = compare_vectors([1, 0], [0, 1])
        self.assertEqual(result["relative_norm_difference"], 0)
        self.assertAlmostEqual(result["relative_l2_error"], math.sqrt(2))
        self.assertEqual(result["cosine"], 0)

    def test_reversed_update_has_twice_the_reference_error(self):
        result = compare_vectors([1, 2], [-1, -2])
        self.assertEqual(result["relative_l2_error"], 2)
        self.assertAlmostEqual(result["cosine"], -1)

    def test_zero_reference_is_explicit(self):
        self.assertIsNone(compare_vectors([0], [1])["relative_l2_error"])
        self.assertEqual(compare_vectors([0], [0])["relative_l2_error"], 0)

    def test_shape_and_nonfinite_fail_closed(self):
        for left, right in (([1], [[1]]), ([], []), ([float("nan")], [1]), ([1], [float("inf")])):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                compare_vectors(left, right)


if __name__ == "__main__":
    unittest.main()
