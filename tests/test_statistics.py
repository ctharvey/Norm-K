import unittest

from norm_k.statistics import bootstrap_mean_interval, grouped_ap_delta_interval


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

    def test_grouped_paired_ap_bootstrap(self):
        labels = [value for _ in range(12) for value in (1, 0)]
        groups = [str(group) for group in range(12) for _ in (1, 0)]
        baseline = [value for _ in range(12) for value in (0.1, 0.9)]
        augmented = [value for _ in range(12) for value in (0.9, 0.1)]
        result = grouped_ap_delta_interval(labels, baseline, augmented, groups, repetitions=100)
        self.assertIsNotNone(result)
        self.assertGreater(result["ci_lower"], 0)
        self.assertEqual(result["subject_clusters"], 12)

    def test_too_few_subject_clusters_refuses_interval(self):
        self.assertIsNone(grouped_ap_delta_interval([1, 0, 1, 0],
                          [0.2, 0.8, 0.2, 0.8], [0.8, 0.2, 0.8, 0.2],
                          ["a", "a", "b", "b"]))


if __name__ == "__main__":
    unittest.main()
