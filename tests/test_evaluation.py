import unittest

from norm_k.dataset import generate_pairs
from norm_k.evaluation import review_budget_metrics, token_features


class EvaluationTests(unittest.TestCase):
    def test_review_budget(self):
        data = review_budget_metrics([True, False, True, False], [0.9, 0.8, 0.4, 0.1], [0.5, 1.0])
        self.assertEqual(data[0]["reviewed"], 2)
        self.assertEqual(data[0]["hits"], 1)
        self.assertAlmostEqual(data[0]["recall"], 0.5)
        self.assertEqual(data[1]["hits"], 2)

    def test_token_features(self):
        result = token_features("Alice does not own 2 dogs", "Alice owns 3 dogs")
        self.assertEqual(result["negation_disagreement"], 1)
        self.assertEqual(result["number_disagreement"], 1)

    def test_heldout_demo_has_all_classes(self):
        pairs = generate_pairs(100)
        self.assertEqual(len({p.label for p in pairs}), 5)


if __name__ == "__main__":
    unittest.main()
