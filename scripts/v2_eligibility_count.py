"""Count and list the pairs selected for the public test. No embeddings, no K.

Usage: v2_eligibility_count.py <multinli zip> <wanli test.jsonl> <output ids tsv>
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from norm_k.public_v2 import load_public  # noqa: E402

data = load_public(sys.argv[1], sys.argv[2])
for name, pairs in (("primary", data.primary), ("replication", data.replication)):
    print(name, len(pairs), "pairs,", sum(p.label == "contradiction" for p in pairs), "contradictions")
print("genres:", dict(Counter(p.relation_family for p in data.primary)))
output = Path(sys.argv[3])
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(["role", "pair_id", "source", "genre"])
    for role, pairs in (("primary", data.primary), ("replication", data.replication)):
        writer.writerows((role, p.id, p.template_family, p.relation_family) for p in pairs)
