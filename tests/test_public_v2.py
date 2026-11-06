import json
import unittest

import numpy as np

from norm_k.embeddings import HashingEncoder
from norm_k.public_v2 import (COMPARISONS, PublicData, clears_bar, eligible, main_clause, mnli_pairs,
                              public_decision, run_public_test, select_public, short_single,
                              wanli_pairs, _held_out_scores, _pair)
from norm_k.counterfactuals import counterfactual_square


def make_pair(pair_id, a, b, gold="neutral", genre="g"):
    return _pair(pair_id, a, b, gold, genre, "mnli_matched")


def item(clears, upper=0.1):
    return {"clears_bar": clears, "interval": {"ci_lower": 0.0, "ci_upper": upper}}


class EligibilityTests(unittest.TestCase):
    def test_broken_edits_seen_in_real_data_are_rejected(self):
        for text in ("People sacrifice their lives for farmers and slaves.",
                     "There is rumor that the money will double.",
                     "While he was gone, I spent some time observing the room.",
                     "We need to tell them what is going on.",
                     "yes it is kind and it's fun"):
            self.assertFalse(main_clause(text), text)

    def test_main_clause_edits_are_kept(self):
        for text in ("Marriage is crucial to society.", "The plane is making strange moves.",
                     "She was a good mistress.", "Marriage is not crucial to society."):
            self.assertTrue(main_clause(text), text)

    def test_length_and_sentence_limits(self):
        self.assertFalse(short_single(" ".join(["word"] * 26) + "."))
        self.assertFalse(short_single("It is warm. It is dry."))
        self.assertTrue(short_single("It is warm."))

    def test_mirror_pair_is_not_eligible(self):
        self.assertFalse(eligible(make_pair("m", "The door is open.", "The door is not open.")))

    def test_pair_needs_both_statements_to_qualify(self):
        self.assertTrue(eligible(make_pair("a", "The door is open.", "The room was cold.")))
        self.assertFalse(eligible(make_pair("b", "The door is open.", "He left the room.")))


class LoaderTests(unittest.TestCase):
    def test_mnli_drops_rows_without_gold_and_counts_agreement(self):
        rows = [{"gold_label": "-", "pairID": "1", "sentence1": "a", "sentence2": "b", "genre": "g",
                 "annotator_labels": ["neutral"]},
                {"gold_label": "contradiction", "pairID": "2", "sentence1": "a", "sentence2": "b",
                 "genre": "g", "annotator_labels": ["contradiction"] * 4 + ["neutral"]}]
        pairs, agreement = mnli_pairs([json.dumps(row) for row in rows], "mnli_matched")
        self.assertEqual([pair.id for pair in pairs], ["mnli-2"])
        self.assertEqual(pairs[0].label, "contradiction")
        self.assertEqual(agreement["mnli-2"], 4)

    def test_only_contradiction_is_the_positive_class(self):
        rows = [{"id": n, "premise": "a", "hypothesis": "b", "gold": gold}
                for n, gold in enumerate(("entailment", "neutral", "contradiction"))]
        labels = [pair.label for pair in wanli_pairs(json.dumps(row) for row in rows)]
        self.assertEqual(labels, ["compatible", "compatible", "contradiction"])

    def test_replication_cannot_share_a_statement_with_primary(self):
        primary = make_pair("p", "The door is open.", "The room was cold.")
        shared = make_pair("w1", "The door is open.", "The lake was frozen.")
        fresh = make_pair("w2", "The lamp is bright.", "The lake was frozen.")
        data = select_public([primary], {"p": 5}, [shared, fresh])
        self.assertEqual([pair.id for pair in data.replication], ["w2"])


class DecisionTests(unittest.TestCase):
    interval = {"ci_lower": 0.01, "ci_upper": 0.2}

    def test_each_condition_can_fail_the_bar(self):
        self.assertTrue(clears_bar(0.05, self.interval, 0.02, 0.01))
        self.assertFalse(clears_bar(0.029, self.interval, 0.02, 0.01))
        self.assertFalse(clears_bar(0.05, {"ci_lower": 0.0, "ci_upper": 0.2}, 0.02, 0.01))
        self.assertFalse(clears_bar(0.05, self.interval, 0.06, 0.01))
        self.assertFalse(clears_bar(0.05, self.interval, 0.02, 0.06))
        self.assertFalse(clears_bar(0.05, None, 0.02, 0.01))

    def test_pass_needs_both_models(self):
        self.assertEqual(public_decision({"k_over_full": item(True), "k_over_rf": item(True)}), "PASS")
        self.assertEqual(public_decision({"k_over_full": item(True), "k_over_rf": item(False)}),
                         "INCONCLUSIVE")

    def test_stop_if_either_model_rules_out_useful_gain(self):
        self.assertEqual(public_decision({"k_over_full": item(True), "k_over_rf": item(False, 0.02)}),
                         "STOP")

    def test_held_out_scores_never_train_on_the_held_out_genre(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(60, 2))
        y = rng.integers(0, 2, size=60)
        groups = np.repeat(["a", "b", "c"], 20)
        flipped = y.copy()
        flipped[groups == "a"] = 1 - flipped[groups == "a"]
        for kind in ("lr", "rf"):
            before = _held_out_scores(kind, x, y, groups, 1)
            after = _held_out_scores(kind, x, flipped, groups, 1)
            # Genre a's own labels can't reach its scores, but do reach other genres'.
            # Tolerance only covers the forest's parallel summation order.
            np.testing.assert_allclose(before[groups == "a"], after[groups == "a"], rtol=1e-12)
            self.assertFalse(np.allclose(before[groups == "b"], after[groups == "b"]))


class RunTests(unittest.TestCase):
    def test_small_run_reports_every_comparison(self):
        subjects = ("door", "lamp", "river", "road", "tower", "wall", "field", "bridge")
        adjectives = ("red", "old", "wide", "tall", "quiet", "dark")
        primary, agreement, replication = [], {}, []
        for number in range(72):
            subject = subjects[number % 8]
            a = f"The {subject} {number} is {adjectives[number % 6]}."
            b = f"The {subject} {number} was {adjectives[(number + 1) % 6]}."
            gold = "contradiction" if number % 2 else "neutral"
            pair = make_pair(f"p{number}", a, b, gold, f"genre{number % 3}")
            primary.append(pair)
            agreement[pair.id] = 3 + number % 3
        for number in range(12):
            replication.append(make_pair(f"w{number}", f"The kite {number} is blue.", f"The boat {number} was green.",
                                         "contradiction" if number % 2 else "neutral", "wanli"))
        data = PublicData(primary, replication, agreement)
        texts = sorted({text for pair in primary + replication
                        for square in [counterfactual_square(pair, "negation")]
                        for text in (square.base, square.other, square.transformed_base,
                                     square.transformed_other)})
        vectors = dict(zip(texts, HashingEncoder(64).encode(texts)))
        nli = {pair.id: float(pair.label == "contradiction") * 0.5 + 0.1
               for pair in primary + replication}
        result = run_public_test(data, vectors, nli, None, repetitions=100, permutations=2)
        self.assertEqual(set(result["comparisons"]), set(COMPARISONS))
        self.assertEqual(set(result["replication"]), set(COMPARISONS))
        self.assertIsNone(result["decision"])
        self.assertIn(result["hash_dry_run_decision"], {"PASS", "STOP", "INCONCLUSIVE"})
        deltas = {name: 1.0 for name in COMPARISONS}
        real = run_public_test(data, vectors, nli, deltas, repetitions=100, permutations=2)
        self.assertFalse(any(item["clears_bar"] for item in real["comparisons"].values()))
        self.assertIn(real["decision"], {"STOP", "INCONCLUSIVE"})


if __name__ == "__main__":
    unittest.main()
