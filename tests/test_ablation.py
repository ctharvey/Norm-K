import unittest

from norm_k.ablation import run_k_ablation
from norm_k.counterfactuals import counterfactual_square
from norm_k.dataset import generate_pairs, split_template
from norm_k.embeddings import HashingEncoder


class AblationTests(unittest.TestCase):
    def test_matched_cohort_and_k_additive_comparison(self):
        try:
            from sklearn.linear_model import LogisticRegression  # noqa: F401
        except ImportError:
            self.skipTest("Research extras not installed")
        pairs = generate_pairs(32)
        examples = [(pair, square) for pair in pairs
                    if (square := counterfactual_square(pair, "negation")) is not None]
        selected = [pair for pair, _ in examples]
        squares = [square for _, square in examples]
        texts = list(dict.fromkeys(t for sq in squares for t in
                                   (sq.base, sq.other, sq.transformed_base, sq.transformed_other)))
        vectors = dict(zip(texts, HashingEncoder(64).encode(texts)))
        result = run_k_ablation(selected, squares, vectors,
                                [split_template(pair.template_family) for pair in selected])
        self.assertEqual(result["eligible_count"], len(selected))
        self.assertEqual(result["baseline"]["test"], result["with_k"]["test"])
        self.assertIn("delta_average_precision", result)


if __name__ == "__main__":
    unittest.main()
