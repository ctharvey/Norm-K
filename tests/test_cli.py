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

            queue = base / "review_queue.jsonl"
            reviews = base / "reviews.jsonl"
            consensus = base / "consensus.json"
            self.assertEqual(main(["prepare-review", "--pairs", str(pairs),
                                   "--per-label", "2", "--output", str(queue)]), 0)
            from norm_k.review import AnnotationLog, load_items
            items = load_items(queue)
            self.assertEqual(len(items), 10)
            log = AnnotationLog(reviews, [item.id for item in items])
            log.add(items[0].id, "alpha", "contradiction")
            log.add(items[0].id, "beta", "contradiction")
            self.assertEqual(main(["export-reviews", "--queue", str(queue),
                                   "--annotations", str(reviews), "--output", str(consensus)]), 0)
            self.assertTrue(consensus.exists())
            with self.assertRaises(FileExistsError):
                main(["prepare-review", "--pairs", str(pairs), "--output", str(queue)])

            external = base / "external.tsv"
            external.write_text("premise\thypothesis\tgold_label\n"
                                "A door is open\tA door is closed\tcontradiction\n",
                                encoding="utf-8")
            imported = base / "imported.jsonl"
            self.assertEqual(main(["import-benchmark", "--input", str(external),
                                   "--source", "test", "--a-field", "premise",
                                   "--b-field", "hypothesis", "--label-field", "gold_label",
                                   "--output", str(imported)]), 0)
            self.assertEqual(len(load_items(imported)), 1)
            self.assertTrue((base / "imported.manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
