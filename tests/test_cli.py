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
            audit = base / "audit.json"
            self.assertEqual(main(["audit", "--pairs", str(pairs), "--output", str(audit)]), 0)
            self.assertTrue(audit.exists())
            self.assertEqual(main(["evaluate", "--pairs", str(pairs), "--cache", str(cache),
                                   "--output", str(output)]), 0)
            self.assertTrue((output / "baseline.json").exists())
            self.assertTrue((output / "pair_scatter.svg").exists())
            self.assertEqual(main(["k-probe", "--count", "20", "--cache", str(cache),
                                   "--output", str(k_output)]), 0)
            self.assertTrue((k_output / "k_probe.json").exists())
            self.assertTrue((k_output / "k_distribution.svg").exists())
            ablation_output = base / "ablation"
            self.assertEqual(main(["k-ablate", "--pairs", str(pairs), "--cache", str(cache),
                                   "--operation", "both", "--output", str(ablation_output)]), 0)
            self.assertTrue((ablation_output / "k_ablation.json").exists())
            self.assertTrue((ablation_output / "pair_k_negation.svg").exists())


if __name__ == "__main__":
    unittest.main()
