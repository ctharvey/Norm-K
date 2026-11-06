"""GPU reruns of local-model work, beside the frozen results. Parity checks, never new verdicts.

vectors:  gpu_parity.py vectors CPU_NPZ OUT_NPZ
nli:      gpu_parity.py nli PAIRS_TSV OUT_TSV [--batch N] [--max-length N] [--keep-order]
diagnose: gpu_parity.py diagnose CPU_NLI_TSV
PAIRS_TSV has columns a and b. Models and revisions match the CPU runs. NLI loads as 16-bit here. See nli_precision_check.py.
"""
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ENCODER, ENCODER_REVISION = "sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130"
NLI, NLI_REVISION = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"


def vectors(source, output):
    stored = np.load(source, allow_pickle=False)
    texts, old = stored["texts"].tolist(), stored["vectors"]
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=ENCODER_REVISION)
    model = AutoModel.from_pretrained(ENCODER, revision=ENCODER_REVISION, use_safetensors=True).eval().to("cuda")
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # Same batches as v4_local_vectors.py.
    new = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    start = time.time()
    with torch.inference_mode():
        for at in range(0, len(order), 64):
            batch = order[at:at + 64]
            encoded = tokenizer([texts[i] for i in batch], padding=True, truncation=True, max_length=384,
                                return_tensors="pt").to("cuda")
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            new[batch] = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
    seconds = time.time() - start
    np.savez_compressed(output, texts=np.asarray(texts), vectors=new)
    similarity = (new.astype(np.float64) * old.astype(np.float64)).sum(1)
    print(Path(source).name, "texts", len(texts), "seconds", round(seconds, 1), "texts per second",
          round(len(texts) / seconds, 1), "largest cosine distance from CPU", float((1 - similarity).max()))


def load_nli(device):
    tokenizer = AutoTokenizer.from_pretrained(NLI, revision=NLI_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(NLI, revision=NLI_REVISION).eval().to(device)
    index = {name.lower(): number for number, name in model.config.id2label.items()}["contradiction"]
    return tokenizer, model, index


def score(tokenizer, model, index, pairs, batch_size, max_length, device):
    found = []
    with torch.inference_mode():
        for at in range(0, len(pairs), batch_size):
            batch = pairs[at:at + batch_size]
            options = {"max_length": max_length} if max_length else {}
            encoded = tokenizer([p[0] for p in batch], [p[1] for p in batch], padding=True, truncation=True,
                                return_tensors="pt", **options).to(device)
            found += torch.softmax(model(**encoded).logits.float(), dim=1)[:, index].tolist()
    return found


def read_pairs(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return [(row["a"], row["b"]) for row in csv.DictReader(stream, delimiter="\t")]


def nli(source, output, batch=64, max_length=256, keep_order=False):
    pairs = read_pairs(source)
    if not keep_order:
        pairs = sorted(pairs, key=lambda pair: len(pair[0]) + len(pair[1]))
    tokenizer, model, index = load_nli("cuda")
    start = time.time()
    found = score(tokenizer, model, index, pairs, batch, max_length, "cuda")
    seconds = time.time() - start
    with open(output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("a", "b", "contradiction_probability"))
        writer.writerows((a, b, value) for (a, b), value in zip(pairs, found))
    print("scored", len(pairs), "seconds", round(seconds, 1), "pairs per second", round(len(pairs) / seconds, 1))


def diagnose(cpu_tsv):
    """Separate padding from device on the pairs where GPU and the stored CPU scores differ most."""
    stored = {}
    with open(cpu_tsv, encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            try:
                stored[(row["a"], row["b"])] = float(row["contradiction_probability"])
            except (TypeError, ValueError):
                pass
    sample = sorted(random.Random(2031).sample(sorted(stored), 4000), key=lambda p: len(p[0]) + len(p[1]))
    tokenizer, gpu, index = load_nli("cuda")
    padded = score(tokenizer, gpu, index, sample, 128, 256, "cuda")
    worst = sorted(range(len(sample)), key=lambda i: -abs(padded[i] - stored[sample[i]]))[:150]
    pairs = [sample[i] for i in worst]
    alone_gpu = score(tokenizer, gpu, index, pairs, 1, 256, "cuda")
    _, cpu, _ = load_nli("cpu")
    alone_cpu = score(tokenizer, cpu, index, pairs, 1, 256, "cpu")
    old, batched = [stored[p] for p in pairs], [padded[i] for i in worst]

    def gap(first, second):
        values = np.abs(np.asarray(first) - np.asarray(second))
        return {"max": float(values.max()), "mean": float(values.mean())}

    print("tf32 matmul", torch.backends.cuda.matmul.allow_tf32, "tf32 cudnn", torch.backends.cudnn.allow_tf32)
    print("all sampled, GPU batch vs stored CPU", gap(padded, [stored[p] for p in sample]))
    print("worst 150, GPU batch vs stored CPU", gap(batched, old))
    print("worst 150, GPU alone vs GPU batch (padding on GPU)", gap(alone_gpu, batched))
    print("worst 150, CPU alone vs stored CPU (padding on CPU)", gap(alone_cpu, old))
    print("worst 150, GPU alone vs CPU alone (device, no padding)", gap(alone_gpu, alone_cpu))


if __name__ == "__main__":
    args = sys.argv[1:]
    command = args.pop(0)
    if command == "vectors":
        vectors(*args)
    elif command == "diagnose":
        diagnose(*args)
    else:
        options = {"keep_order": "--keep-order" in args}
        args = [a for a in args if a != "--keep-order"]
        for flag, name in (("--batch", "batch"), ("--max-length", "max_length")):
            if flag in args:
                at = args.index(flag)
                options[name] = int(args[at + 1])
                del args[at:at + 2]
        nli(*args, **options)
