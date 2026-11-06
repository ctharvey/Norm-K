"""Check GPU NLI scores against finished CPU scores, then score the pairs CPU hasn't reached.

Usage: v7_nli_gpu_fill.py PAIRS_TSV CPU_NLI_TSV GPU_NLI_TSV
Full precision. Exits with no new scores if the check fails. Prints measured speed for both checks.
"""
import csv
import random
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
BATCH, SAMPLE, TOLERANCE, CUTOFF = 128, 2000, 1e-3, 0.5

pairs_tsv, cpu_tsv, gpu_tsv = sys.argv[1:4]
with open(pairs_tsv, encoding="utf-8", newline="") as stream:
    wanted = [(row["a"], row["b"]) for row in csv.DictReader(stream, delimiter="\t")]
done = {}
with open(cpu_tsv, encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        try:  # The last line may be half-written while the CPU job runs.
            done[(row["a"], row["b"])] = float(row["contradiction_probability"])
        except (TypeError, ValueError):
            pass

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION).eval().to("cuda")
model.float()
index = {name.lower(): number for number, name in model.config.id2label.items()}["contradiction"]


def score(pairs):
    found = {}
    ordered = sorted(pairs, key=lambda pair: len(pair[0]) + len(pair[1]))
    with torch.inference_mode():
        for at in range(0, len(ordered), BATCH):
            batch = ordered[at:at + BATCH]
            encoded = tokenizer([p[0] for p in batch], [p[1] for p in batch], padding=True, truncation=True,
                                max_length=256, return_tensors="pt").to("cuda")
            values = torch.softmax(model(**encoded).logits.float(), dim=1)[:, index].tolist()
            found.update(zip(batch, values))
    return found


sample = random.Random(2031).sample(sorted(done), SAMPLE)
start = time.time()
checked = score(sample)
seconds = time.time() - start
gaps = [abs(checked[p] - done[p]) for p in sample]
flipped = sum((checked[p] >= CUTOFF) != (done[p] >= CUTOFF) for p in sample)
print("check pairs", SAMPLE, "max gap", max(gaps), "mean gap", sum(gaps) / len(gaps), "crossed cutoff", flipped,
      "pairs per second", round(SAMPLE / seconds, 1), flush=True)
if max(gaps) > TOLERANCE or flipped:
    raise SystemExit("GPU scores disagree with CPU scores. No new scores written.")

left = [pair for pair in wanted if pair not in done]
start = time.time()
found = score(left)
seconds = time.time() - start
with open(gpu_tsv, "w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(("a", "b", "contradiction_probability"))
    writer.writerows((a, b, value) for (a, b), value in found.items())
print("filled", len(found), "seconds", round(seconds, 1), "pairs per second", round(len(found) / max(seconds, 1e-9), 1))
