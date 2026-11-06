"""Run the v6 NaN-NLI test. See docs/preregistration-nan-nli-v6-draft.md. Run with cwd = repo.

bridge-texts: v6_nan_nli.py bridge-texts MNLI_ZIP WANLI IDS_TSV OUT_JSON
bridge:       v6_nan_nli.py bridge MNLI_ZIP WANLI IDS_TSV ENCODER OUT_JSON [--max-new N]
nan-texts:    v6_nan_nli.py nan-texts NAN_CSV OUT_JSON
run:          v6_nan_nli.py run NAN_CSV ENCODER OUT_JSON [HASH_JSON NLI_TSV] [--max-new N]
ENCODER is hash, openai, or an .npz path from v4_local_vectors.py. The bridge never reads NaN-NLI.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from norm_k.embeddings import EmbeddingCache, HashingEncoder, OpenAIEncoder
from norm_k.flip_v6 import (NLI_COLUMNS, bridge_passes, bridge_row, nan_texts, read_nan, score_nan, statements_of,
                            wrapper_texts)
from norm_k.public_v2 import load_public
from v5_flip_matched import load_fresh


def vectors_for(texts, encoder, max_new):
    """Return vectors and measured tokens."""
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


def seen_sets(mnli, wanli, ids):
    return {"fresh_mnli_train": load_fresh(mnli, ids), "v2_wanli": load_public(mnli, wanli).replication}


def bridge_texts(mnli, wanli, ids, output):
    sets = seen_sets(mnli, wanli, ids)
    texts = wrapper_texts(statements_of([p for pairs in sets.values() for p in pairs]))
    Path(output).write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print("texts", len(texts))


def bridge(mnli, wanli, ids, encoder, output, max_new=0):
    if Path(output).exists():
        raise SystemExit(f"Refusing to replace {output}")
    sets = seen_sets(mnli, wanli, ids)
    texts = wrapper_texts(statements_of([p for pairs in sets.values() for p in pairs]))
    table, tokens = vectors_for(texts, encoder, max_new)
    result = {name: bridge_row(pairs, table) for name, pairs in sets.items()}
    result["api_tokens_used"] = tokens
    result["passes"] = bridge_passes(result["fresh_mnli_train"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("tokens", tokens, "passes", result["passes"])


def read_nli(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return {row["pair_id"]: {c: float(row[c]) for c in NLI_COLUMNS} for row in csv.DictReader(stream, delimiter="\t")}


def run(nan, encoder, output, hash_json=None, nli_tsv=None, max_new=0):
    if Path(output).exists():
        raise SystemExit(f"Refusing to replace {output}")
    pairs = read_nan(nan)
    table, tokens = vectors_for(nan_texts(pairs), encoder, max_new)
    hash_delta = None if hash_json is None else json.loads(Path(hash_json).read_text(encoding="utf-8"))["delta"]
    result = score_nan(pairs, table, hash_delta=hash_delta, nli=None if nli_tsv is None else read_nli(nli_tsv))
    result["api_tokens_used"] = tokens
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("tokens", tokens, "verdict", result["primary"]["verdict"], "size gate", result["size_gate"])


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = 0
    if "--max-new" in args:
        at = args.index("--max-new")
        limit = int(args[at + 1])
        del args[at:at + 2]
    command, rest = args[0], args[1:]
    if command == "bridge-texts":
        bridge_texts(*rest)
    elif command == "bridge":
        bridge(*rest, max_new=limit)
    elif command == "nan-texts":
        pairs = read_nan(rest[0])
        Path(rest[1]).write_text(json.dumps(nan_texts(pairs), ensure_ascii=False), encoding="utf-8")
        print("texts", len(nan_texts(pairs)))
    else:
        run(*rest, max_new=limit)
