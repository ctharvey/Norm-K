import unittest

import numpy as np

from norm_k.counterfactuals import toggle_negation
from norm_k.gate_v7 import (Candidates, evaluate, gate_verdict, hits, margin, needed_pairs, passed, retrieve,
                            select_pairs)
from norm_k.public_v2 import _pair


def unit(vector):
    return vector / np.linalg.norm(vector)


def world(size=60, seed=5):
    """Contradictions sit further from A in cosine but close once A is negated."""
    rng = np.random.default_rng(seed)
    pairs, vectors = [], {}
    for i in range(size):
        a, b = f"The item {i} is red.", f"The thing {i} is blue."
        base = rng.normal(size=32)
        contradiction = i % 3 == 0
        partner = base + (1.2 if contradiction else 0.7) * rng.normal(size=32)
        vectors[a], vectors[b] = unit(base), unit(partner)
        vectors[toggle_negation(a)] = unit((partner if contradiction else base + rng.normal(size=32)))
        vectors[toggle_negation(b)] = unit((base if contradiction else partner + rng.normal(size=32)))
        pairs.append(_pair(f"w-{i}", a, b, "contradiction" if contradiction else "neutral", "wanli", "wanli"))
    return pairs, vectors


class Selection(unittest.TestCase):
    def pair(self, name, a="The cat is here.", b="The dog is there.", gold="neutral"):
        return _pair(name, a, b, gold, "wanli", "wanli")

    def test_seen_statements_are_excluded(self):
        pairs = [self.pair("1"), self.pair("2", a="The bird is near.")]
        chosen = select_pairs(pairs, {"The cat is here."})
        self.assertEqual([p.id for p in chosen], ["2"])

    def test_labels_do_not_change_the_selection(self):
        first = [self.pair(str(i), a=f"The cat {i} is here.") for i in range(30)]
        second = [self.pair(str(i), a=f"The cat {i} is here.", gold="contradiction") for i in range(30)]
        self.assertEqual([p.id for p in select_pairs(first, set(), limit=10)],
                         [p.id for p in select_pairs(second, set(), limit=10)])

    def test_one_pair_is_kept_per_statement_pair(self):
        pairs = [self.pair("1", gold="contradiction"), self.pair("2", gold="neutral")]
        self.assertEqual(len(select_pairs(pairs, set())), 1)


class Passing(unittest.TestCase):
    def test_passes_the_rounded_share(self):
        self.assertEqual(passed(np.arange(100.0), 0.10).sum(), 10)
        self.assertTrue(passed(np.arange(100.0), 0.10)[-1])

    def test_a_pair_needs_passing_and_confirming(self):
        c = Candidates(["a1", "a2"], ["b1", "b2"], np.zeros(2), np.zeros(2))
        mask = np.asarray([True, False])
        nli = {("a1", "b1"): 0.9, ("a2", "b2"): 0.9}
        self.assertEqual(hits(c, mask, [("a1", "b1"), ("a2", "b2")], nli).tolist(), [1.0, 0.0])
        nli[("a1", "b1")] = 0.1
        self.assertEqual(hits(c, mask, [("a1", "b1")], nli).tolist(), [0.0])

    def test_a_contradiction_never_retrieved_is_a_miss(self):
        c = Candidates(["a1"], ["b1"], np.zeros(1), np.zeros(1))
        self.assertEqual(hits(c, np.ones(1, dtype=bool), [("a9", "b9")], None).tolist(), [0.0])


class Rule(unittest.TestCase):
    def found(self, value, lower, upper):
        return {"margin": value, "ci_lower": lower, "ci_upper": upper}

    def test_no_room_is_unclear_not_a_fail(self):
        self.assertEqual(gate_verdict(self.found(0.0, -0.01, 0.01), 0.05, None), "UNCLEAR")

    def test_pass_needs_margin_lower_bound_and_hash(self):
        self.assertEqual(gate_verdict(self.found(0.08, 0.02, 0.12), 0.3, 0.01), "GATE-PASS")
        self.assertEqual(gate_verdict(self.found(0.08, 0.02, 0.12), 0.3, 0.09), "UNCLEAR")
        self.assertEqual(gate_verdict(self.found(0.08, 0.0, 0.12), 0.3, 0.01), "UNCLEAR")

    def test_fail_needs_the_upper_bound_below_the_bar(self):
        self.assertEqual(gate_verdict(self.found(0.01, -0.02, 0.049), 0.3, 0.0), "GATE-FAIL")
        self.assertEqual(gate_verdict(self.found(0.03, 0.01, 0.06), 0.3, 0.0), "UNCLEAR")

    def test_margin_of_identical_hits_is_zero(self):
        same = np.asarray([1.0, 0.0, 1.0, 1.0])
        self.assertEqual(margin(same, same, repetitions=100)["margin"], 0.0)


class Pipeline(unittest.TestCase):
    def test_candidates_are_k_per_distinct_a(self):
        pairs, vectors = world()
        self.assertEqual(len(retrieve(pairs, vectors, 5).a), 5 * len(pairs))

    def test_hash_style_reads_only_what_passes(self):
        pairs, vectors = world()
        candidates = retrieve(pairs, vectors)
        self.assertLess(len(needed_pairs(candidates, False)), len(needed_pairs(candidates, True)))

    def test_end_to_end_found_cannot_exceed_ceiling(self):
        pairs, vectors = world()
        nli = {pair: 0.9 for pair in needed_pairs(retrieve(pairs, vectors), True)}
        result = evaluate(pairs, vectors, nli, repetitions=100)
        self.assertLessEqual(max(result["found"].values()), result["ceiling"])
        self.assertLessEqual(result["reachable"]["5"], result["reachable"]["20"])
        self.assertEqual(result["passed_to_nli"], round(0.1 * result["candidates"]))


if __name__ == "__main__":
    unittest.main()
