import tempfile
import unittest
from pathlib import Path

from norm_k.dataset import generate_pairs
from norm_k.review import (AnnotationLog, Review, consensus_report, latest_reviews,
                           load_items, pairwise_agreement, queue_from_pairs, save_items)


class ReviewTests(unittest.TestCase):
    def test_queue_is_balanced_and_blind_to_oracle(self):
        queue = queue_from_pairs(generate_pairs(30), per_label=3, seed=42)
        self.assertEqual(len(queue), 15)
        self.assertEqual(len({item.id for item in queue}), 15)
        self.assertTrue(all(item.reference_label for item in queue))
        self.assertTrue(all("reference_label" not in item.for_reviewer() for item in queue))
        self.assertTrue(all(item.id.startswith("pair-") for item in queue))
        self.assertTrue(all(item.reference_label not in item.id for item in queue))

    def test_queue_roundtrip_and_append_only_reviews(self):
        queue = queue_from_pairs(generate_pairs(2), per_label=1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "queue.jsonl"
            reviews_path = Path(temp) / "reviews.jsonl"
            save_items(path, queue)
            self.assertEqual(load_items(path), queue)
            log = AnnotationLog(reviews_path, [item.id for item in queue])
            first = log.add(queue[0].id, "charlie", "equivalent", "medium", "initial")
            second = log.add(queue[0].id, "charlie", "contradiction", "high", "correction")
            self.assertNotEqual(first.review_id, second.review_id)
            self.assertEqual(len(log.read_all()), 2)
            self.assertEqual(latest_reviews(log.read_all())[(queue[0].id, "charlie")].label,
                             "contradiction")
            with self.assertRaisesRegex(ValueError, "not found"):
                log.add("missing", "charlie", "equivalent")

    def test_consensus_needs_two_distinct_reviewers(self):
        queue = queue_from_pairs(generate_pairs(2), per_label=1)
        with tempfile.TemporaryDirectory() as temp:
            log = AnnotationLog(Path(temp) / "reviews.jsonl", [item.id for item in queue])
            pair_id = queue[0].id
            log.add(pair_id, "alpha", "contradiction")
            log.add(pair_id, "alpha", "contradiction")
            report = consensus_report(queue, log.read_all())
            self.assertFalse(report["resolved"])
            log.add(pair_id, "beta", "contradiction")
            report = consensus_report(queue, log.read_all())
            self.assertEqual(len(report["resolved"]), 1)
            log.add(pair_id, "beta", "compatible")
            report = consensus_report(queue, log.read_all())
            self.assertEqual(len(report["resolved"]), 0)
            self.assertEqual(len(report["disagreement"]), 1)

    def test_invalid_review_rejected(self):
        with self.assertRaises(ValueError):
            Review("r1", "p1", "user", "possibly", "medium", "", "2026")
        with self.assertRaises(ValueError):
            Review("r1", "p1", "user", "contradiction", "medium", "x" * 2001, "2026")

    def test_pairwise_kappa_and_undefined_case(self):
        with tempfile.TemporaryDirectory() as temp:
            log = AnnotationLog(Path(temp) / "review.jsonl", ["p1", "p2", "p3", "p4"])
            for reviewer in ("alpha", "beta"):
                for pair_id, label in (("p1", "contradiction"), ("p2", "equivalent"),
                                       ("p3", "contradiction"), ("p4", "equivalent")):
                    log.add(pair_id, reviewer, label)
            agreement = pairwise_agreement(log.read_all())
            self.assertEqual(len(agreement), 1)
            self.assertAlmostEqual(agreement[0]["cohens_kappa"], 1.0)
            self.assertEqual(agreement[0]["shared_pairs"], 4)
            constant = [review for review in log.read_all() if review.pair_id in ("p1", "p3")]
            self.assertIsNone(pairwise_agreement(constant)[0]["cohens_kappa"])


if __name__ == "__main__":
    unittest.main()
