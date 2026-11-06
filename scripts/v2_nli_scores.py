"""Write the NLI baseline's contradiction probability for every selected pair.

Usage: v2_nli_scores.py <multinli zip> <wanli test.jsonl> <output tsv>
Needs torch, transformers and sentencepiece. Runs locally; no API calls.
"""
import csv
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from norm_k.public_v2 import load_public  # noqa: E402

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
REVISION = sys.argv[4] if len(sys.argv) > 4 else None

data = load_public(sys.argv[1], sys.argv[2])
pairs = data.primary + data.replication
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION).eval()
index = {name.lower(): number for number, name in model.config.id2label.items()}["contradiction"]

rows = []
with torch.no_grad():
    for start in range(0, len(pairs), 32):
        batch = pairs[start:start + 32]
        inputs = tokenizer([p.text_a for p in batch], [p.text_b for p in batch], truncation=True,
                           padding=True, return_tensors="pt")
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[:, index]
        rows += [(pair.id, float(value)) for pair, value in zip(batch, probabilities)]
        print(f"scored {len(rows)} of {len(pairs)}", flush=True)

output = Path(sys.argv[3])
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(["pair_id", "contradiction_probability"])
    writer.writerows(rows)
print(f"Wrote {len(rows)} scores to {output}")
