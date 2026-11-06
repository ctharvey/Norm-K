import tempfile
import unittest
from pathlib import Path

import numpy as np

from norm_k.flip_v6 import (bridge_passes, control, negated, nli_scores, read_nan, score_nan, shuffle_null,
                            size_quality, wrapper_texts)
from norm_k.public_v2 import _pair


def unit(vector):
    return vector / np.linalg.norm(vector)


def pairs_and_vectors(strength, seed=3, premises=12, per_premise=6):
    """Contradictions get a negated A that lands near B when strength is above zero."""
    rng = np.random.default_rng(seed)
    pairs, vectors = [], {}

    def add(text, vector):
        vectors[text] = unit(vector)

    for i in range(premises):
        a = f"Premise {i}."
        base = rng.normal(size=32)
        add(a, base)
        add(negated(a), base + 0.3 * rng.normal(size=32))
        add(control(a), base + 0.3 * rng.normal(size=32))
        for j in range(per_premise):
            gold = "contradiction" if j % 2 else "entailment"
            b = f"Hypothesis {i} {j}."
            other = base + rng.normal(size=32)
            add(b, other)
            add(control(b), other + 0.3 * rng.normal(size=32))
            pull = strength * base if gold == "contradiction" else np.zeros(32)
            add(negated(b), other + 0.3 * rng.normal(size=32) + pull)
            pairs.append(_pair(f"nan-{i}-{j}", a, b, gold, f"family {i % 3}", "nan_nli"))
    return pairs, vectors


class Edits(unittest.TestCase):
    def test_the_two_edits_differ_by_one_word(self):
        first, second = negated("Rarely do they go.").split(), control("Rarely do they go.").split()
        self.assertEqual(len(first), len(second) + 1)
        self.assertEqual([w for w in first if w != "not"], second)

    def test_the_statement_stays_verbatim(self):
        self.assertTrue(negated("Not one came.").endswith("that Not one came."))

    def test_wrapper_texts_has_all_three_forms(self):
        self.assertEqual(len(wrapper_texts(["A.", "B."])), 6)


class Data(unittest.TestCase):
    def test_a_changed_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nan.csv"
            path.write_text("premise,hypothesis,label,Construction\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_nan(path)


class Bridge(unittest.TestCase):
    def row(self, delta, lower, gate=True):
        return {"size_gate": gate, "measure": {"delta": delta, "interval": {"ci_lower": lower, "ci_upper": 1}}}

    def test_needs_delta_lower_bound_and_size_gate(self):
        self.assertTrue(bridge_passes(self.row(0.03, 0.001)))
        self.assertFalse(bridge_passes(self.row(0.029, 0.001)))
        self.assertFalse(bridge_passes(self.row(0.05, 0.0)))
        self.assertFalse(bridge_passes(self.row(0.05, 0.01, gate=False)))


class Null(unittest.TestCase):
    def test_shuffling_keeps_each_groups_label_count(self):
        y = np.asarray([1, 0, 0, 1, 1, 0, 1, 0])
        first = np.arange(8.0)
        groups = ["a", "a", "a", "a", "b", "b", "b", "b"]
        self.assertTrue(np.isfinite(shuffle_null(y, first, first[::-1].copy(), groups, 1, shuffles=20)))

    def test_identical_scores_give_a_zero_null(self):
        y = np.asarray([1, 0, 1, 0])
        scores = np.arange(4.0)
        self.assertEqual(shuffle_null(y, scores, scores, ["a", "a", "b", "b"], 1, shuffles=20), 0.0)


class Score(unittest.TestCase):
    def test_signal_gives_a_positive_delta_and_no_signal_does_not_transfer(self):
        pairs, vectors = pairs_and_vectors(strength=2.0)
        strong = score_nan(pairs, vectors, repetitions=200)["primary"]
        self.assertGreater(strong["delta"], 0.1)
        pairs, vectors = pairs_and_vectors(strength=0.0)
        self.assertNotEqual(score_nan(pairs, vectors, repetitions=200)["primary"]["verdict"], "TRANSFERS")

    def test_failing_the_size_gate_forces_unclear(self):
        pairs, vectors = pairs_and_vectors(strength=2.0)
        for text in list(vectors):
            if text.startswith("It is the case that "):
                vectors[text] = vectors[text.removeprefix("It is the case that ")]
        self.assertEqual(score_nan(pairs, vectors, repetitions=200)["primary"]["verdict"], "UNCLEAR")

    def test_size_quality_gap_is_control_minus_negation(self):
        a = unit(np.asarray([1.0, 0.0]))
        vectors = {"S": a, negated("S"): unit(np.asarray([1.0, 1.0])), control("S"): a}
        self.assertLess(size_quality(["S"], vectors)["mean_gap"], 0)


class Nli(unittest.TestCase):
    def test_orders_and_margin(self):
        y = np.asarray([1, 0])
        table = {"p": dict(c_ab=3, e_ab=0, n_ab=0, c_ba=0, e_ba=3, n_ba=0),
                 "q": dict(c_ab=0, e_ab=3, n_ab=0, c_ba=3, e_ba=0, n_ba=0)}
        found = nli_scores(y, table, ["p", "q"])
        self.assertEqual(found["pc_ab"], 1.0)
        self.assertEqual(found["pc_ba"], 0.5)
        self.assertEqual(found["c_minus_e"], 1.0)


if __name__ == "__main__":
    unittest.main()
