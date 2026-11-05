import unittest

import numpy as np

from norm_k.geometry import (cosine, difference, euclidean, four_point,
                             hoyer_sparsity, k_norm, k_relative, pair_features)


class GeometryTests(unittest.TestCase):
    def test_cosine_direction(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [-1, 0]), -1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 0]), 0.0)

    def test_difference_and_distance(self):
        np.testing.assert_allclose(difference([1, 2], [3, 5]), [2, 3])
        np.testing.assert_allclose(difference([3, 5], [1, 2]), [-2, -3])
        self.assertAlmostEqual(euclidean([1, 2], [3, 5]), np.sqrt(13))

    def test_hoyer(self):
        self.assertAlmostEqual(hoyer_sparsity([1, 1, 1, 1]), 0.0)
        self.assertAlmostEqual(hoyer_sparsity([1, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(hoyer_sparsity([0, 0, 0, 0]), 0.0)

    def test_additive_four_point_is_zero(self):
        b, n, p, both = [0, 0], [1, 0], [0, 2], [1, 2]
        np.testing.assert_allclose(four_point(b, n, p, both), [0, 0])
        self.assertAlmostEqual(k_norm(b, n, p, both), 0.0)
        self.assertAlmostEqual(k_relative(b, n, p, both), 0.0)

    def test_interaction_and_relative_scale(self):
        b, n, p, both = [0, 0], [1, 0], [0, 2], [2, 2]
        self.assertAlmostEqual(k_norm(b, n, p, both), 1.0)
        self.assertAlmostEqual(k_relative(b, n, p, both), 1.0 / (3.0 + 1e-9))
        self.assertAlmostEqual(k_relative(np.array(b) * 10, np.array(n) * 10,
                                          np.array(p) * 10, np.array(both) * 10),
                               1.0 / 3.0, places=8)

    def test_symmetry_of_pair_features(self):
        left, right = [0.5, 0.1, 0], [0.3, 0.7, 0]
        first, second = pair_features(left, right), pair_features(right, left)
        for key in first:
            self.assertAlmostEqual(first[key], second[key])

    def test_invalid_vectors_fail_fast(self):
        for left, right in [([1, 2], [1]), ([float("nan")], [1]), ([[1, 2]], [1, 2])]:
            with self.assertRaises(ValueError):
                cosine(left, right)
        with self.assertRaises(ValueError):
            k_relative([0, 0], [1, 0], [0, 1], [1, 1], epsilon=0)


if __name__ == "__main__":
    unittest.main()
