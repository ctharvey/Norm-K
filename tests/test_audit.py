import unittest
from dataclasses import replace

from norm_k.audit import audit_pairs, dataset_fingerprint, purged_indices
from norm_k.dataset import generate_pairs


class AuditTests(unittest.TestCase):
    def test_generated_dataset_has_no_group_leakage(self):
        pairs = generate_pairs(100)
        audit = audit_pairs(pairs)
        self.assertEqual(audit["pair_count"], 500)
        self.assertEqual(audit["duplicate_text_pairs"], 0)
        self.assertEqual(audit["same_text_pair_conflicting_labels"], [])
        for split in audit["splits"].values():
            self.assertEqual(split["group_overlap"], [])
            self.assertEqual(split["remaining_overlap_after_purge"], 0)

    def test_fingerprint_catches_order_and_label_changes(self):
        pairs = generate_pairs(2)
        self.assertNotEqual(dataset_fingerprint(pairs), dataset_fingerprint(list(reversed(pairs))))
        self.assertEqual(dataset_fingerprint(pairs), dataset_fingerprint(pairs))

    def test_purge_keeps_test_and_removes_overlapping_train_validation(self):
        a, b, c = generate_pairs(1)[:3]
        b = replace(b, text_a=a.text_a)
        c = replace(c, text_b=a.text_a)
        self.assertEqual(purged_indices([a, b, c], ["train", "validation", "test"]), [2])


if __name__ == "__main__":
    unittest.main()
