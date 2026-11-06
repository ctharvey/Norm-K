import unittest

import numpy as np

from norm_k.counterfactuals import toggle_negation
from norm_k.flip_v5 import (POOL, diluted_pool, gate_margin, gate_passes, gate_verdict, matched_edit, pool_texts,
                            recall_at, score_set, tense_edit, wrong_partner)
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

    def test_reported_checks_are_present(self):
        pairs, vectors = self.build(False)
        result = score_set(pairs, vectors, repetitions=200)
        for key in ("wrong_partner_ap", "statement_only_ap", "matched_at_verb", "recall_at_budget", "gate"):
            self.assertIn(key, result)

    def test_wrong_partner_removes_a_signal_that_needs_the_partner(self):
        pairs, vectors = self.build(False)
        y = np.asarray([p.label == "contradiction" for p in pairs], dtype=int)
        assigned = {t: POOL["note"](t) for p in pairs for t in (p.text_a, p.text_b)}
        found = wrong_partner(pairs, y, assigned, vectors, seed=1)
        self.assertLess(found["negation"], 0.5)


class Gate(unittest.TestCase):
    def test_recall_counts_only_what_passes(self):
        y = np.asarray([1, 0, 1, 0])
        self.assertEqual(recall_at(y, np.asarray([4.0, 3.0, 2.0, 1.0]), 0.25), 0.5)
        self.assertEqual(recall_at(y, np.asarray([4.0, 3.0, 2.0, 1.0]), 0.75), 1.0)

    def pool_inputs(self):
        rng = np.random.default_rng(5)
        pairs = [_pair(str(i), f"Item {i} is warm.", f"Item {i} is cold.",
                       "contradiction" if i % 2 == 0 else "neutral", "g", "s") for i in range(40)]
        statements = [t for p in pairs for t in (p.text_a, p.text_b)]
        assigned = {t: POOL["note"](t) for t in statements}
        vectors = {t: unit(rng.normal(size=16)) for t in pool_texts(statements)}
        y = np.asarray([p.label == "contradiction" for p in pairs], dtype=int)
        return pairs, y, assigned, vectors

    def test_pool_keeps_real_pairs_and_hits_prevalence(self):
        pairs, y, assigned, vectors = self.pool_inputs()
        pool = diluted_pool(pairs, y, assigned, vectors, 0.05, seed=1)
        self.assertEqual(len(pool["y"]), 400)
        self.assertEqual(pool["y"].sum(), y.sum())
        first = pairs[0]
        self.assertAlmostEqual(pool["cosine"][0], float(vectors[first.text_a] @ vectors[first.text_b]))
        self.assertTrue(np.allclose(pool["flip_negation"] + pool["cosine"], pool["negated_cosine"]))

    def test_pool_stops_at_the_available_pairs(self):
        pairs, y, assigned, vectors = self.pool_inputs()
        pool = diluted_pool(pairs, y, assigned, vectors, 0.0001, seed=1)
        self.assertEqual(len(pool["y"]), 40 * 40)

    def test_margin_is_recall_over_the_stronger_cosine_direction(self):
        y = np.asarray([1] * 5 + [0] * 95)
        flip = np.asarray([1.0] * 5 + [0.0] * 95)
        low = np.asarray([0.0] * 5 + [1.0] * 95)
        found = gate_margin({"y": y, "flip_negation": flip, "cosine": low}, "flip_negation", 1, 50)
        self.assertEqual(found["cosine_recall"], 1.0)
        self.assertEqual(found["margin"], 0.0)

    def test_verdicts(self):
        clear = {"margin": 0.10, "ci_lower": 0.04, "ci_upper": 0.16}
        self.assertEqual(gate_verdict(clear, 0.02, 0.01), "GATE-PASS")
        self.assertEqual(gate_verdict(clear, 0.02, None), "GATE-PASS")
        self.assertEqual(gate_verdict(clear, -0.01, 0.01), "UNCLEAR")
        self.assertEqual(gate_verdict(clear, 0.02, 0.11), "UNCLEAR")
        self.assertEqual(gate_verdict({"margin": 0.01, "ci_lower": -0.02, "ci_upper": 0.04}, 0.02, 0.0), "GATE-FAIL")


if __name__ == "__main__":
    unittest.main()
