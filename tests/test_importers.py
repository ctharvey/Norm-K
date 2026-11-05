import tempfile
import unittest
from pathlib import Path

from norm_k.importers import import_benchmark


class ImporterTests(unittest.TestCase):
    def test_csv_requires_explicit_label_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "pairs.csv"
            source.write_text("premise,hypothesis,gold_label,split\n"
                              "Alice has a dog,Alice owns a pet,entailment,test\n"
                              "Door is open,Door is closed,contradiction,train\n", encoding="utf-8")
            queue, manifest = import_benchmark(source, source="fixture", a_field="premise",
                                               b_field="hypothesis", label_field="gold_label",
                                               split_field="split")
            self.assertEqual(len(queue), 2)
            self.assertIsNone(queue[0].reference_label)
            self.assertEqual(manifest["unmapped_raw_labels"],
                             {"entailment": 1, "contradiction": 1})
            queue, manifest = import_benchmark(source, source="fixture", a_field="premise",
                                               b_field="hypothesis", label_field="gold_label",
                                               label_map={"contradiction": "contradiction"})
            self.assertEqual(queue[1].reference_label, "contradiction")
            self.assertIsNone(queue[0].reference_label)
            self.assertEqual(len(manifest["source_sha256"]), 64)

    def test_tsv_and_jsonl_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            tsv = Path(temp) / "examples.tsv"
            tsv.write_text("a\tb\nhello\thi\n", encoding="utf-8")
            queue, _ = import_benchmark(tsv, "tsv-test", a_field="a", b_field="b")
            self.assertEqual(queue[0].text_a, "hello")
            jsonl = Path(temp) / "examples.jsonl"
            jsonl.write_text('{"x": "abc", "y": "def"}\n', encoding="utf-8")
            queue, _ = import_benchmark(jsonl, "json-test", a_field="x", b_field="y")
            self.assertEqual(queue[0].text_b, "def")

    def test_import_does_not_infer_entailment_or_neutral(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "examples.jsonl"
            source.write_text('{"a": "A", "b": "B", "label": "neutral"}\n', encoding="utf-8")
            queue, _ = import_benchmark(source, "unknown", "a", "b", "label")
            self.assertIsNone(queue[0].reference_label)
            with self.assertRaises(ValueError):
                import_benchmark(source, "unknown", "a", "b", "label",
                                 label_map={"neutral": "not_an_allowed_label"})

    def test_bad_rows_and_caps_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "examples.csv"
            source.write_text("a,b\nA,B\nC,D\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_rows"):
                import_benchmark(source, "fixture", "a", "b", max_rows=1)
            with self.assertRaisesRegex(ValueError, "missing requested column"):
                import_benchmark(source, "fixture", "unknown", "b")


if __name__ == "__main__":
    unittest.main()
