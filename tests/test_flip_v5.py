import unittest

import numpy as np

from norm_k.counterfactuals import toggle_negation
from norm_k.flip_v5 import POOL, gate_passes, matched_edit, pool_texts, score_set, tense_edit
from norm_k.public_v2 import _pair


def unit(vector):
    return vector / np.linalg.norm(vector)


def at_distance(base, target, rng):
    """A unit vector whose cosine distance from base is target."""
    other = rng.normal(size=base.shape)
    other = unit(other - other.dot(base) * base)
    similarity = 1 - target
    return similarity * base + np.sqrt(1 - similarity ** 2) * other


class Edits(unittest.TestCase):
    def test_tense_swaps_the_verb_negation_edits(self):
        self.assertEqual(tense_edit("The cat is here."), "The cat was here.")
        self.assertEqual(tense_edit("It was not late."), "It is not late.")
        self.assertIsNone(tense_edit("Cats sleep."))
        self.assertEqual(tense_edit("A consumed-is word."), "A consumed-was word.")

    def test_every_edit_changes_the_text(self):
        for edit in POOL.values():
            self.assertNotEqual(edit("The cat is here."), "The cat is here.")


class Matching(unittest.TestCase):
    def test_picks_the_edit_closest_to_negation(self):
        rng = np.random.default_rng(0)
        text = "The cat is here."
        base = unit(rng.normal(size=32))
        vectors = {text: base, toggle_negation(text): at_distance(base, 0.20, rng)}
        for index, edit in enumerate(POOL.values()):
            vectors[edit(text)] = at_distance(base, 0.05 + 0.04 * index, rng)
        name, size, target = matched_edit(text, vectors)
        self.assertAlmostEqual(target, 0.20)
        self.assertLess(abs(size - target), 0.021)
        self.assertEqual(name, list(POOL)[4])

    def test_gate(self):
        self.assertTrue(gate_passes({"mean_gap": 0.009, "median_abs_gap": 0.019}))
        self.assertFalse(gate_passes({"mean_gap": 0.011, "median_abs_gap": 0.019}))
        self.assertFalse(gate_passes({"mean_gap": 0.0, "median_abs_gap": 0.021}))


class Scoring(unittest.TestCase):
    def build(self, planted_in_pool):
        rng = np.random.default_rng(3)
        pairs = [_pair(str(i), f"Item {i} is warm.", f"Item {i} is cold.",
                       "contradiction" if i % 3 == 0 else "neutral", "g", "s") for i in range(300)]
        vectors = {}
        for text in pool_texts([t for p in pairs for t in (p.text_a, p.text_b)]):
            vectors[text] = unit(rng.normal(size=64))
        for pair_index, pair in enumerate(pairs):
            for text, partner in ((pair.text_a, pair.text_b), (pair.text_b, pair.text_a)):
                base = vectors[text]
                edited = [toggle_negation(text)] + [edit(text) for edit in POOL.values()]
                for changed in edited:
                    vectors[changed] = at_distance(base, 0.2, rng)
                if pair_index % 3 == 0:
                    moved = 0.8 * vectors[partner] + 0.2 * base
                    vectors[toggle_negation(text)] = unit(moved)
                    if planted_in_pool:
                        for edit in POOL.values():
                            vectors[edit(text)] = unit(moved)
        return pairs, vectors

    def test_negation_only_signal_survives(self):
        pairs, vectors = self.build(False)
        result = score_set(pairs, vectors, repetitions=200)
        self.assertGreater(result["negation_vs_matched"]["delta"], 0.03)

    def test_failed_gate_forces_unclear(self):
        pairs, vectors = self.build(False)
        result = score_set(pairs, vectors, repetitions=200)
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["negation_vs_matched"]["verdict"], "UNCLEAR")

    def test_signal_in_every_edit_is_size_explained(self):
        pairs, vectors = self.build(True)
        result = score_set(pairs, vectors, repetitions=200)
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["negation_vs_matched"]["verdict"], "SIZE-EXPLAINED")


if __name__ == "__main__":
    unittest.main()
