"""Score saved NLI pairs at 32-bit, beside frozen 16-bit scores. A check, never a new verdict.

nli_precision_check.py PAIRS_TSV OUT_TSV DEVICE [--batch N] [--limit N]
The pinned model loads as 16-bit by default under transformers 5, so this forces 32-bit.
--limit takes a seeded sample. Prints measured speed.
"""
import csv
import random
import sys
import time

import torch

from gpu_parity import load_nli, read_pairs, score

args = sys.argv[1:]
options = {"--batch": 32, "--limit": 0}
for flag in options:
    if flag in args:
        at = args.index(flag)
        options[flag] = int(args[at + 1])
        del args[at:at + 2]
source, output, device = args
pairs = read_pairs(source)
if options["--limit"]:
    pairs = random.Random(2031).sample(sorted(pairs), options["--limit"])
pairs = sorted(pairs, key=lambda pair: len(pair[0]) + len(pair[1]))
tokenizer, model, index = load_nli(device)
model = model.float()
print("device", device, "dtype", next(model.parameters()).dtype, "threads", torch.get_num_threads(), flush=True)
start = time.time()
found = score(tokenizer, model, index, pairs, options["--batch"], 256, device)
seconds = time.time() - start
with open(output, "w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(("a", "b", "contradiction_probability"))
    writer.writerows((a, b, value) for (a, b), value in zip(pairs, found))
print("scored", len(pairs), "seconds", round(seconds, 1), "pairs per second", round(len(pairs) / seconds, 1))
