import unittest

import numpy as np

from norm_k.dataset import generate_pairs, split_template
from norm_k.embeddings import HashingEncoder
from norm_k.probes import low_rank_probe


class ProbeTests(unittest.TestCase):
    def test_low_rank_probe_uses_held_out_templates(self):
        try:
            from sklearn.decomposition import PCA  # noqa: F401
        except ImportError:
            self.skipTest("Research extras not installed")
        pairs = generate_pairs(20)
        texts = list(dict.fromkeys(t for p in pairs for t in (p.text_a, p.text_b)))
        vectors = dict(zip(texts, HashingEncoder(32).encode(texts)))
        splits = [split_template(p.template_family) for p in pairs]
        result = low_rank_probe(pairs, vectors, splits, ranks=(2, 4, 8))
        self.assertEqual([r["dimensions"] for r in result["ranks"]], [2, 4, 8])
        self.assertTrue(all(np.isfinite(r["average_precision"]) for r in result["ranks"]))


if __name__ == "__main__":
    unittest.main()
