import tempfile
import unittest
from collections import Counter
from pathlib import Path

from norm_k.dataset import LABEL_ORDER, generate_pairs, generate_quads, split_subject
from norm_k.schema import Pair, load_pairs, save_pairs


class DatasetTests(unittest.TestCase):
    def test_generation_is_reproducible(self):
        self.assertEqual(generate_pairs(10, 42), generate_pairs(10, 42))
        self.assertNotEqual(generate_pairs(10, 42), generate_pairs(10, 43))

    def test_balanced_and_unique_at_target_size(self):
        pairs = generate_pairs(600)
        self.assertEqual(len(pairs), 3000)
        self.assertEqual(Counter(p.label for p in pairs), {label: 600 for label in LABEL_ORDER})
        self.assertEqual(len({p.id for p in pairs}), 3000)
        self.assertEqual(len({(p.text_a, p.text_b) for p in pairs}), 3000)

    def test_labels_and_temporal_contract(self):
        pairs = generate_pairs(20)
        for pair in pairs:
            if pair.label == "contradiction":
                self.assertTrue(pair.same_fact and pair.incompatible and pair.same_valid_time)
            if pair.label == "temporal_update":
                self.assertFalse(pair.same_valid_time)
            self.assertTrue(pair.generated)

    def test_jsonl_roundtrip(self):
        pairs = generate_pairs(4)
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "pairs.jsonl"
            save_pairs(file, pairs)
            self.assertEqual(load_pairs(file), pairs)
            with file.open("a", encoding="utf-8") as f:
                f.write(file.read_text(encoding="utf-8").splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate pair ID"):
                load_pairs(file)

    def test_subject_splits_are_consistent(self):
        names = [p.subject for p in generate_pairs(100)]
        self.assertEqual({split_subject(name) for name in names}, {"train", "validation", "test"})
        self.assertTrue(all(split_subject(name, 42) == split_subject(name, 42) for name in names))

    def test_quads_have_correct_subject_swaps(self):
        quads = generate_quads(25)
        self.assertEqual({q.operation_pair for q in quads},
                         {"negation_x_subject", "negation_x_value", "subject_x_value", "time_x_value"})
        for q in quads:
            self.assertNotEqual(q.subject, q.other_subject)
            self.assertIn(q.subject, q.base)
            if "subject" in q.operation_pair:
                self.assertIn(q.other_subject, q.both)
            if "negation" in q.operation_pair:
                self.assertIn("does not own", q.both)

    def test_schema_rejects_bad_label(self):
        d = generate_pairs(1)[0].__dict__.copy()
        d["label"] = "anything"
        with self.assertRaises(ValueError):
            Pair(**d)


if __name__ == "__main__":
    unittest.main()
