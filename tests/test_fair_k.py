import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from norm_k.ablation import (load_matched_benchmark, preregistered_decision, run_fair_k_test)
from norm_k.counterfactuals import CounterfactualSquare, counterfactual_square, is_mirror
from norm_k.embeddings import HashingEncoder
from norm_k.geometry import k_norm, k_relative
from norm_k.schema import Pair

BENCHMARK = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "matched-2x2-v1.tsv"


def _pair(text_a, text_b, label="contradiction"):
    return Pair(id="p", text_a=text_a, text_b=text_b, label=label, subject="Alice", attribute="a",
                relation_family="r", template_family="t", transformation="x", cardinality="single",
                same_fact=True, incompatible=True, same_valid_time=True, rationale="r")


class MirrorTests(unittest.TestCase):
    def test_negation_pair_is_a_mirror_and_k_is_degenerate(self):
        pair = _pair("Alice owns a bike.", "Alice does not own a bike.")
        square = counterfactual_square(pair, "negation")
        self.assertTrue(is_mirror(square))
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=8), rng.normal(size=8)
        self.assertAlmostEqual(k_norm(a, b, b, a), 2 * float(np.linalg.norm(a - b)))
        self.assertAlmostEqual(k_relative(a, b, b, a), 1.0, places=6)

    def test_value_change_is_not_a_mirror(self):
        pair = _pair("Alice is in Rome.", "Alice is in Oslo.")
        self.assertFalse(is_mirror(counterfactual_square(pair, "negation")))


class DecisionRuleTests(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(preregistered_decision(0.05, {"ci_lower": 0.01, "ci_upper": 0.09}, 0.02),
                         "PASS")
        self.assertEqual(preregistered_decision(0.05, {"ci_lower": -0.01, "ci_upper": 0.11}, 0.02),
                         "INCONCLUSIVE")
        self.assertEqual(preregistered_decision(0.01, {"ci_lower": 0.002, "ci_upper": 0.02}, 0.02),
                         "STOP")
        self.assertEqual(preregistered_decision(0.0, {"ci_lower": -0.02, "ci_upper": 0.02}, 0.02),
                         "STOP")
        self.assertEqual(preregistered_decision(0.5, None, 0.02), "INCONCLUSIVE")

    def test_gain_inside_the_shuffled_null_is_not_a_pass(self):
        self.assertEqual(preregistered_decision(0.05, {"ci_lower": 0.01, "ci_upper": 0.09}, 0.06),
                         "INCONCLUSIVE")


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = load_matched_benchmark(BENCHMARK)

    def test_design_is_balanced(self):
        cells = Counter((p.relation_family, p.template_family) for p in self.pairs)
        self.assertEqual(len({family for family, _ in cells}), 12)
        self.assertEqual(set(cells.values()), {3})
        self.assertEqual(len(self.pairs), 144)
        for pair in self.pairs:
            positive_cell = pair.template_family.endswith("contradiction")
            self.assertEqual(pair.label == "contradiction", positive_cell, pair.text_a)

    def test_negation_appears_in_both_classes(self):
        negated = Counter(p.label == "contradiction" for p in self.pairs
                          if " not " in f" {p.text_a} {p.text_b} ")
        self.assertGreater(negated[True], 0)
        self.assertGreater(negated[False], 0)

    def test_every_pair_supports_both_operations_with_no_mirrors(self):
        for pair in self.pairs:
            for operation in ("negation", "subject_swap"):
                square = counterfactual_square(pair, operation)
                self.assertIsNotNone(square, f"{operation}: {pair.text_a} / {pair.text_b}")
                self.assertFalse(is_mirror(square), pair.text_a)


class FairTestTests(unittest.TestCase):
    def test_mirrors_are_excluded_and_report_is_complete(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("Research extras not installed")
        pairs = load_matched_benchmark(BENCHMARK)
        mirror = _pair("Alice owns a bike.", "Alice does not own a bike.")
        rows = [(p, counterfactual_square(p, "negation")) for p in pairs]
        rows.append((mirror, counterfactual_square(mirror, "negation")))
        selected, squares = [p for p, _ in rows], [s for _, s in rows]
        texts = sorted({t for s in squares for t in
                        (s.base, s.other, s.transformed_base, s.transformed_other)})
        vectors = dict(zip(texts, HashingEncoder(64).encode(texts)))
        result = run_fair_k_test(selected, squares, vectors, repetitions=100, permutations=2)
        self.assertEqual(result["mirror_pairs_excluded"], 1)
        self.assertEqual(result["pairs_scored"], 144)
        self.assertEqual(result["families"], 12)
        self.assertIn(result["decision"], {"PASS", "STOP", "INCONCLUSIVE"})
        self.assertEqual(result["primary_interval"]["cluster_unit"], "relation_family")

    def test_square_order_must_match_pairs(self):
        pair = _pair("Alice is in Rome.", "Alice is in Oslo.")
        square = CounterfactualSquare("other", "negation", "contradiction", pair.text_a, pair.text_b,
                                      pair.text_a, pair.text_b)
        with self.assertRaises(ValueError):
            run_fair_k_test([pair], [square], {})


if __name__ == "__main__":
    unittest.main()
