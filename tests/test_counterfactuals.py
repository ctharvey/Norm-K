import unittest

from norm_k.counterfactuals import (counterfactual_square, replace_subject,
                                    toggle_negation)
from norm_k.dataset import generate_pairs


class CounterfactualTests(unittest.TestCase):
    def test_negation_is_involution_on_supported_grammar(self):
        samples = (
            "In 2026, Alice owns a bicycle.",
            "In 2026, Alice does not own a bicycle.",
            "Alice is happy.", "Alice is not happy.",
            "Alice was 32 years old.", "Alice was not 32 years old.",
            "Alice lives in Denver.", "Alice does not live in Denver.",
        )
        for statement in samples:
            with self.subTest(statement=statement):
                self.assertEqual(toggle_negation(toggle_negation(statement)), statement)

    def test_negation_declines_unsupported_grammar(self):
        self.assertIsNone(toggle_negation("Alice bought a bicycle."))
        self.assertIsNone(toggle_negation("No bicycles were bought."))

    def test_subject_swap_only_changes_whole_names(self):
        self.assertEqual(replace_subject("Alice likes Alice's cat.", "Alice", "Bob"),
                         "Bob likes Bob's cat.")
        self.assertIsNone(replace_subject("Alicia owns a bike.", "Alice", "Bob"))

    def test_pair_level_square_has_both_counterfactual_corners(self):
        pairs = generate_pairs(8)
        matched = 0
        for pair in pairs:
            square = counterfactual_square(pair, "subject_swap")
            if square:
                self.assertEqual(square.pair_id, pair.id)
                self.assertNotEqual(square.base, square.transformed_base)
                self.assertNotEqual(square.other, square.transformed_other)
                matched += 1
        self.assertGreater(matched, 0)

    def test_unrelated_pair_is_not_one_sided_subject_swap(self):
        pairs = [p for p in generate_pairs(12) if p.label == "unrelated"]
        self.assertTrue(all(counterfactual_square(p, "subject_swap") is None for p in pairs))


if __name__ == "__main__":
    unittest.main()
