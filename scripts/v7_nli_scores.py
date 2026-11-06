"""Score candidate pairs with the pinned NLI model. Local, no API calls.

Usage: v7_nli_scores.py PAIRS_TSV OUTPUT_TSV
Needs torch, transformers and sentencepiece. Pairs already in OUTPUT_TSV are kept, so a stopped run
can resume. Prints measured seconds.
"""
import csv
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path.home() / ".claude" / "tools"))
from progress import Progress  # noqa: E402

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
BATCH = 64

source, output = Path(sys.argv[1]), Path(sys.argv[2])
csv.field_size_limit(10_000_000)
with open(source, encoding="utf-8", newline="") as stream:
    wanted = [(row["a"], row["b"]) for row in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_MINIMAL)]
done = set()
if output.exists():
    with open(output, encoding="utf-8", newline="") as stream:
        done = {(row["a"], row["b"]) for row in csv.DictReader(stream, delimiter="\t")}
todo = sorted((pair for pair in wanted if pair not in done), key=lambda pair: len(pair[0]) + len(pair[1]))

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION).eval()
index = {name.lower(): number for number, name in model.config.id2label.items()}["contradiction"]

progress = Progress(f"norm-k-{source.stem}", f"Norm-K NLI: {source.stem}", total=len(todo), unit="pairs")
start = time.time()
fresh_file = not output.exists()
try:
    with open(output, "a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        if fresh_file:
            writer.writerow(("a", "b", "contradiction_probability"))
        with torch.inference_mode():
            for at in range(0, len(todo), BATCH):
                batch = todo[at:at + BATCH]
                encoded = tokenizer([p[0] for p in batch], [p[1] for p in batch], padding=True, truncation=True,
                                    max_length=256, return_tensors="pt")
                found = torch.softmax(model(**encoded).logits, dim=1)[:, index].tolist()
                writer.writerows((a, b, value) for (a, b), value in zip(batch, found))
                if (at // BATCH) % 20 == 0:
                    stream.flush()
                    progress.update(done=at + len(batch))
except BaseException as error:
    progress.finish("failed", note=type(error).__name__)
    raise
progress.finish()
print("scored", len(todo), "seconds", round(time.time() - start, 1))
