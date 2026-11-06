"""Run the v4 placebo test. See docs/preregistration-flip-placebo-v4.md. Run with cwd = repo.

select:  v4_flip_placebo.py select MNLI_ZIP WANLI OUT_TSV TEXTS_JSON
run:     v4_flip_placebo.py run MNLI_ZIP WANLI IDS_TSV ENCODER OUTPUT_JSON [HASH_JSON] [--max-new N]
ENCODER is hash, openai, or a path to an .npz written by v4_local_vectors.py.
"""
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from norm_k.embeddings import EmbeddingCache, HashingEncoder, OpenAIEncoder
from norm_k.flip_v4 import edited_texts, score_set, select_fresh
from norm_k.public_v2 import eligible, load_public, mnli_pairs

TRAIN = "multinli_1.0/multinli_1.0_train.jsonl"


def train_lines(path):
    with zipfile.ZipFile(path) as archive:
        return archive.read(TRAIN).decode("utf-8").splitlines()


def select(mnli, wanli, out_tsv, texts_json):
    data = load_public(mnli, wanli)
    seen = {t for pairs in (data.primary, data.replication) for p in pairs for t in (p.text_a, p.text_b)}
    fresh = select_fresh(train_lines(mnli), seen)
    with open(out_tsv, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("pair_id", "genre"))
        writer.writerows((p.id, p.relation_family) for p in sorted(fresh, key=lambda item: item.id))
    texts = edited_texts([*fresh, *data.primary, *data.replication])
    Path(texts_json).write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print("fresh pairs", len(fresh), "texts", len(texts), "words", sum(len(t.split()) for t in texts))


def load_fresh(mnli, ids_tsv):
    with open(ids_tsv, encoding="utf-8", newline="") as stream:
        wanted = [row["pair_id"] for row in csv.DictReader(stream, delimiter="\t")]
    pairs, _ = mnli_pairs(train_lines(mnli), "mnli_train")
    by_id = {pair.id: pair for pair in pairs}
    fresh = [by_id[pair_id] for pair_id in wanted]
    if not all(eligible(pair) for pair in fresh):
        raise SystemExit("A stored pair ID resolved to an ineligible pair")
    return fresh


def vectors_for(encoder, texts, max_new):
    if encoder.endswith(".npz"):
        stored = np.load(encoder, allow_pickle=False)
        table = dict(zip(stored["texts"].tolist(), stored["vectors"].astype(float)))
        return {text: table[text] for text in texts}
    chosen = HashingEncoder() if encoder == "hash" else OpenAIEncoder()
    with EmbeddingCache(".norm_k_cache/embeddings.sqlite3") as cache:
        return dict(zip(texts, cache.embed(chosen, texts, max_new=max_new)))


def run(mnli, wanli, ids_tsv, encoder, output, hash_json=None, max_new=0):
    if Path(output).exists():
        raise SystemExit(f"Refusing to replace {output}")
    data, fresh = load_public(mnli, wanli), load_fresh(mnli, ids_tsv)
    sets = {"fresh_mnli_train": fresh, "v2_mnli": data.primary, "v2_wanli": data.replication}
    vectors = vectors_for(encoder, edited_texts([p for pairs in sets.values() for p in pairs]), max_new)
    hash_deltas = None if hash_json is None else json.loads(Path(hash_json).read_text(encoding="utf-8"))
    result = {name: score_set(pairs, vectors, hash_deltas=(hash_deltas or {}).get(name))
              for name, pairs in sets.items()}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, item in result.items():
        print(name, item["pairs"], item["replication"]["verdict"], item["negation_vs_placebo"]["verdict"])


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = 0
    if "--max-new" in args:
        at = args.index("--max-new")
        limit = int(args[at + 1])
        del args[at:at + 2]
    if args[0] == "select":
        select(*args[1:])
    else:
        run(*args[1:], max_new=limit)
