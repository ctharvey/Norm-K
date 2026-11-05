import unittest

from norm_k.statistics import bootstrap_mean_interval


class StatisticsTests(unittest.TestCase):
    def test_constant_mean_interval(self):
        self.assertEqual(bootstrap_mean_interval([3, 3, 3, 3]), (3.0, 3.0))

    def test_interval_contains_sample_mean(self):
        lower, upper = bootstrap_mean_interval([1, 2, 3, 4, 5], seed=123)
        self.assertLess(lower, 3)
        self.assertGreater(upper, 3)
        self.assertEqual((lower, upper), bootstrap_mean_interval([1, 2, 3, 4, 5], seed=123))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            bootstrap_mean_interval([1])
        with self.assertRaises(ValueError):
            bootstrap_mean_interval([1, 2], confidence=1.0)


if __name__ == "__main__":
    unittest.main()
