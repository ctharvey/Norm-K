import json
import unittest

import numpy as np

from norm_k.counterfactuals import toggle_negation
from norm_k.flip_v4 import (edited_texts, edits, flip_score, placebo_edit, score_set, select_fresh,
                            swap_null, verdict)
from norm_k.public_v2 import _pair


def line(pair_id, premise, hypothesis, gold="neutral"):
    return json.dumps({"pairID": pair_id, "sentence1": premise, "sentence2": hypothesis, "genre": "fiction",
                       "gold_label": gold, "annotator_labels": [gold]})


class PlaceboEdit(unittest.TestCase):
    def test_lands_on_the_verb_negation_edits(self):
        for text in ("The cat is here and the dog was there.", "The dog was there and the cat is here.",
                     "The dog was there but the cat is not here.", "It was not late."):
            verb_end = len(__import__("os").path.commonprefix([text, toggle_negation(text)]))
            changed = placebo_edit(text, "really")
            self.assertEqual(changed.count(" really"), 1)
            self.assertLessEqual(abs(changed.index(" really") - verb_end), 4, text)

    def test_keeps_an_existing_negation(self):
        self.assertEqual(placebo_edit("It is not late.", "really"), "It is really not late.")

    def test_ignores_words_containing_the_verb(self):
        self.assertEqual(placebo_edit("This wasp is big.", "indeed"), "This wasp is indeed big.")
        self.assertIsNone(placebo_edit("Cats sleep.", "indeed"))


class Selection(unittest.TestCase):
    def test_uses_one_pair_per_premise_and_skips_seen_text(self):
        lines = [line("1a", "The cat is here.", "The cat is big."),
                 line("1b", "The cat is here.", "The cat is small."),
                 line("2a", "The dog is here.", "The dog is big."),
                 line("3a", "The cow is here.", "The cow is big.")]
        picked = select_fresh(lines, {"The cow is big."})
        self.assertEqual(sorted(p.text_a for p in picked), ["The cat is here.", "The dog is here."])

    def test_does_not_depend_on_labels(self):
        rows = [("1", "The cat is here.", "The cat is big."), ("2", "The dog is here.", "The dog is big."),
                ("3", "The hen is here.", "The hen is big.")]
        neutral = select_fresh([line(*row) for row in rows], set(), 2)
        contradicting = select_fresh([line(*row, gold="contradiction") for row in rows], set(), 2)
        self.assertEqual([p.text_a for p in neutral], [p.text_a for p in contradicting])


class Scoring(unittest.TestCase):
    def vectors(self, pairs, seed=0):
        rng = np.random.default_rng(seed)
        found = {}
        for text in edited_texts(pairs):
            v = rng.normal(size=16)
            found[text] = v / np.linalg.norm(v)
        return found

    def test_flip_rises_when_the_edit_moves_b_onto_a(self):
        pair = _pair("x", "It is warm.", "It is cold.", "contradiction", "g", "s")
        vectors = self.vectors([pair])
        negate = edits()["negation"]
        vectors[negate(pair.text_b)] = vectors[pair.text_a]
        vectors[negate(pair.text_a)] = vectors[pair.text_b]
        self.assertGreater(flip_score(pair, negate, vectors), 0)

    def test_random_vectors_show_no_specific_negation_effect(self):
        pairs = [_pair(str(i), f"Item {i} is warm.", f"Item {i} is cold.",
                       "contradiction" if i % 3 == 0 else "neutral", "g", "s") for i in range(300)]
        result = score_set(pairs, self.vectors(pairs), repetitions=200)
        self.assertNotEqual(result["negation_vs_placebo"]["verdict"], "NEGATION-SPECIFIC")
        self.assertNotEqual(result["replication"]["verdict"], "REPLICATES")

    def test_planted_negation_signal_is_found(self):
        pairs = [_pair(str(i), f"Item {i} is warm.", f"Item {i} is cold.",
                       "contradiction" if i % 3 == 0 else "neutral", "g", "s") for i in range(300)]
        vectors, negate = self.vectors(pairs), edits()["negation"]
        for i, pair in enumerate(pairs):
            if i % 3 == 0:
                vectors[negate(pair.text_b)] = vectors[pair.text_a]
                vectors[negate(pair.text_a)] = vectors[pair.text_b]
        result = score_set(pairs, vectors, repetitions=200)
        self.assertEqual(result["negation_vs_placebo"]["verdict"], "NEGATION-SPECIFIC")
        self.assertEqual(result["replication"]["verdict"], "REPLICATES")

    def test_planted_signal_in_every_edit_is_edit_general(self):
        pairs = [_pair(str(i), f"Item {i} is warm.", f"Item {i} is cold.",
                       "contradiction" if i % 3 == 0 else "neutral", "g", "s") for i in range(300)]
        vectors = self.vectors(pairs)
        for i, pair in enumerate(pairs):
            if i % 3 == 0:
                for edit in edits().values():
                    vectors[edit(pair.text_b)] = vectors[pair.text_a]
                    vectors[edit(pair.text_a)] = vectors[pair.text_b]
        result = score_set(pairs, vectors, repetitions=200)
        self.assertEqual(result["negation_vs_placebo"]["verdict"], "EDIT-GENERAL")

    def test_swap_null_is_small_for_equal_scores(self):
        rng = np.random.default_rng(1)
        y = (rng.random(300) < 0.3).astype(int)
        score = rng.random(300)
        self.assertLess(swap_null(y, score, score, 1, shuffles=20), 1e-9)


class Verdict(unittest.TestCase):
    def check(self, delta, low, high, null=0.01, hash_delta=0.0):
        return verdict(delta, {"ci_lower": low, "ci_upper": high}, null, hash_delta, "YES", "NO")

    def test_rule(self):
        self.assertEqual(self.check(0.05, 0.01, 0.09), "YES")
        self.assertEqual(self.check(0.05, -0.01, 0.09), "UNCLEAR")
        self.assertEqual(self.check(0.05, 0.01, 0.09, null=0.06), "UNCLEAR")
        self.assertEqual(self.check(0.05, 0.01, 0.09, hash_delta=0.06), "UNCLEAR")
        self.assertEqual(self.check(0.01, -0.02, 0.029), "NO")
        self.assertEqual(self.check(0.01, -0.02, 0.05), "UNCLEAR")
        self.assertEqual(verdict(0.05, None, 0.0, None, "YES", "NO"), "UNCLEAR")


if __name__ == "__main__":
    unittest.main()
