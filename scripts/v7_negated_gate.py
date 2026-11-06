"""Run the v7 gate test. See docs/preregistration-negated-gate-v7-draft.md. Run with cwd = repo.

select:     v7_negated_gate.py select WANLI_TRAIN WANLI_TEST MNLI_ZIP V5_IDS OUT_TSV TEXTS_JSON
candidates: v7_negated_gate.py candidates WANLI_TRAIN IDS_TSV ENCODER OUT_TSV [--max-new N]
run:        v7_negated_gate.py run WANLI_TRAIN IDS_TSV ENCODER NLI_TSV OUT_JSON [HASH_JSON] [--max-new N]
ENCODER is hash, openai, or an .npz path from v4_local_vectors.py. Hash scores only what is passed on.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from norm_k.counterfactuals import toggle_negation
from norm_k.embeddings import EmbeddingCache, HashingEncoder, OpenAIEncoder
from norm_k.gate_v7 import evaluate, needed_pairs, retrieve, select_pairs
from norm_k.public_v2 import load_public, mnli_pairs, wanli_pairs
from v5_flip_matched import V4_IDS, load_fresh, read_ids, statements, train_lines

TRAIN_SHA256 = "85058cf017a911e89242dc29fa0a4ddaad3664cb923dc0a82145fdda14b694e5"


def read_train(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != TRAIN_SHA256:
        raise SystemExit("WANLI train does not match its pinned SHA-256")
    return wanli_pairs(Path(path).read_text(encoding="utf-8").splitlines())


def load_selected(train, ids_tsv):
    by_id = {p.id: p for p in read_train(train)}
    return [by_id[i] for i in read_ids(ids_tsv)]


def texts_for(pairs):
    found = statements(pairs)
    return sorted({*found, *map(toggle_negation, found)})


def select(train, test, mnli, v5_ids, out_tsv, texts_json):
    data = load_public(mnli, test)
    lines = train_lines(mnli)
    by_id = {p.id: p for p in mnli_pairs(lines, "mnli_train")[0]}
    used = [by_id[i] for i in read_ids(V4_IDS)]
    test_pairs = wanli_pairs(Path(test).read_text(encoding="utf-8").splitlines())
    seen = set(statements([*data.primary, *data.replication, *used, *load_fresh(mnli, v5_ids), *test_pairs]))
    chosen = select_pairs(read_train(train), seen)
    with open(out_tsv, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("pair_id",))
        writer.writerows((p.id,) for p in sorted(chosen, key=lambda item: item.id))
    texts = texts_for(chosen)
    Path(texts_json).write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print("pairs", len(chosen), "distinct A", len({p.text_a for p in chosen}), "distinct B",
          len({p.text_b for p in chosen}), "texts", len(texts), "words", sum(len(t.split()) for t in texts))


def vectors_for(texts, encoder, max_new):
    if encoder.endswith(".npz"):
        stored = np.load(encoder, allow_pickle=False)
        table = dict(zip(stored["texts"].tolist(), stored["vectors"].astype(float)))
        return {text: table[text] for text in texts}, None
    if encoder == "hash":
        return dict(zip(texts, HashingEncoder().encode(texts))), None
    chosen = OpenAIEncoder()
    with EmbeddingCache(".norm_k_cache/embeddings.sqlite3") as cache:
        found = dict(zip(texts, cache.embed(chosen, texts, max_new=max_new)))
    return found, chosen.tokens_used


def candidates(train, ids_tsv, encoder, out_tsv, max_new=0):
    pairs = load_selected(train, ids_tsv)
    table, tokens = vectors_for(texts_for(pairs), encoder, max_new)
    needed = needed_pairs(retrieve(pairs, table), full=encoder != "hash")
    with open(out_tsv, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("a", "b"))
        writer.writerows(needed)
    print("pairs to read", len(needed), "tokens", tokens)


def read_nli(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return {(row["a"], row["b"]): float(row["contradiction_probability"])
                for row in csv.DictReader(stream, delimiter="\t")}


def run(train, ids_tsv, encoder, nli_tsv, output, hash_json=None, max_new=0):
    if Path(output).exists():
        raise SystemExit(f"Refusing to replace {output}")
    pairs = load_selected(train, ids_tsv)
    table, tokens = vectors_for(texts_for(pairs), encoder, max_new)
    hash_margin = None if hash_json is None else json.loads(Path(hash_json).read_text(encoding="utf-8"))["margin"]["margin"]
    result = evaluate(pairs, table, read_nli(nli_tsv), hash_margin, full=encoder != "hash")
    result["api_tokens_used"] = tokens
    result["hash_margin"] = hash_margin
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("verdict", "margin", "found", "ceiling", "room")}))


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = 0
    if "--max-new" in args:
        at = args.index("--max-new")
        limit = int(args[at + 1])
        del args[at:at + 2]
    command, rest = args[0], args[1:]
    if command == "select":
        select(*rest)
    elif command == "candidates":
        candidates(*rest, max_new=limit)
    else:
        run(*rest, max_new=limit)
