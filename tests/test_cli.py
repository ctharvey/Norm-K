import tempfile
import unittest
import importlib.util
from pathlib import Path

from norm_k.cli import main


class CLITests(unittest.TestCase):
    def test_end_to_end_offline_smoke(self):
        if importlib.util.find_spec("sklearn") is None or importlib.util.find_spec("matplotlib") is None:
            self.skipTest("Research extras not installed")
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            pairs = base / "pairs.jsonl"
            cache = base / "cache.sqlite"
            output = base / "evaluations"
            k_output = base / "k"
            self.assertEqual(main(["generate", "--per-label", "20", "--output", str(pairs)]), 0)
            self.assertEqual(main(["evaluate", "--pairs", str(pairs), "--cache", str(cache),
                                   "--output", str(output)]), 0)
            self.assertTrue((output / "baseline.json").exists())
            self.assertTrue((output / "pair_scatter.svg").exists())
            self.assertEqual(main(["k-probe", "--count", "20", "--cache", str(cache),
                                   "--output", str(k_output)]), 0)
            self.assertTrue((k_output / "k_probe.json").exists())
            self.assertTrue((k_output / "k_distribution.svg").exists())


if __name__ == "__main__":
    unittest.main()
