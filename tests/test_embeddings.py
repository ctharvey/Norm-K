import tempfile
import unittest
from pathlib import Path

import numpy as np

from norm_k.embeddings import EmbeddingCache, HashingEncoder


class FailingRemoteEncoder:
    name = "test-remote-no-actual-api"
    dimensions = 3

    def encode(self, _texts):
        raise AssertionError("A remote call should never occur in this test")


class EmbeddingTests(unittest.TestCase):
    def test_hashing_encoder_deterministic(self):
        encoder = HashingEncoder(64)
        a, b = encoder.encode(["Alice owns a dog", "Alice owns a dog"])
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (64,))

    def test_cache_persists_and_is_immutable_to_caller(self):
        encoder = HashingEncoder(64)
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "cache.sqlite"
            with EmbeddingCache(db) as cache:
                vectors = cache.embed(encoder, ["A test", "A test", "Other test"])
                self.assertEqual(len(vectors), 3)
                np.testing.assert_array_equal(vectors[0], vectors[1])
                vectors[0][0] = 999
            with EmbeddingCache(db) as cache:
                cached = cache.embed(encoder, ["A test"])
                self.assertNotEqual(cached[0][0], 999)

    def test_uncached_remote_embeddings_are_opt_in_and_preflighted(self):
        with tempfile.TemporaryDirectory() as temp:
            with EmbeddingCache(Path(temp) / "db.sqlite") as cache:
                with self.assertRaisesRegex(RuntimeError, "exceeding --max-new=0"):
                    cache.embed(FailingRemoteEncoder(), ["The text"], max_new=0)
                with self.assertRaisesRegex(RuntimeError, "needs 2 new API embeddings"):
                    cache.embed(FailingRemoteEncoder(), ["A", "B"], max_new=1)

    def test_cache_separates_models(self):
        with tempfile.TemporaryDirectory() as temp:
            with EmbeddingCache(Path(temp) / "db.sqlite") as cache:
                cache.embed(HashingEncoder(64), ["Same text"])
                self.assertIsNone(cache.lookup(HashingEncoder(128), "Same text"))


if __name__ == "__main__":
    unittest.main()
