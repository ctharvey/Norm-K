"""Exploratory: is v7's margin explained by negation words? Saved vectors and saved NLI scores only.

cheap_polarity_clean.py MNLI_ZIP WANLI_TRAIN V5_IDS V7_IDS V5_MPNET_NPZ V7_MPNET_NPZ NLI_TSV OUTPUT_JSON
Adds a mismatch flag to cosine at several weights, and splits v7's contradictions by negation words.
The best weight is picked on the same data, which favors the baseline. Nothing here is a verdict.
"""
import json
import sys
from pathlib import Path

import numpy as np

from cheap_polarity import direction, has_negation, pair_scores
from norm_k.counterfactuals import toggle_negation
from norm_k.gate_v7 import hits, margin, passed, retrieve
from v5_flip_matched import load_fresh, statements
from v5_mutation_scores import openai_table, rows
from v7_negated_gate import load_selected, read_nli, texts_for

WEIGHTS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)


def score(pairs, table, v, nli):
    held = retrieve(pairs, table)
    a, b = rows(held.a, table), rows(held.b, table)
    na, nb = rows([toggle_negation(t) for t in held.a], table), rows([toggle_negation(t) for t in held.b], table)
    v_negated = pair_scores(held.a, held.b, a, b, na, nb, v)["v_negated_cosine"]
    flag = (has_negation(held.a) != has_negation(held.b)).astype(float)
    ranks = {"cosine": held.cosine, "negated_cosine": held.negated_cosine, "v_negated_cosine": v_negated}
    for weight in WEIGHTS:
        ranks[f"cosine+{weight}*mismatch"] = held.cosine + weight * flag
        ranks[f"negated_cosine+{weight}*mismatch"] = held.negated_cosine + weight * flag
    masks = {name: passed(value) for name, value in ranks.items()}
    wanted = [(p.text_a, p.text_b) for p in pairs if p.label == "contradiction"]
    found = {name: hits(held, mask, wanted, nli) for name, mask in masks.items()}
    best = max((n for n in found if n.startswith("cosine+")), key=lambda n: found[n].mean())
    best_negated = max((n for n in found if n.startswith("negated_cosine+")), key=lambda n: found[n].mean())
    a_neg, b_neg = has_negation([x for x, _ in wanted]), has_negation([y for _, y in wanted])
    groups = {"no_negation_word": ~a_neg & ~b_neg, "exactly_one": a_neg != b_neg, "both": a_neg & b_neg}
    result = {
        "candidates_with_mismatch": float(flag.mean()),
        "passed_with_mismatch": {name: float(flag[masks[name]].mean())
                                 for name in ("cosine", "negated_cosine", "v_negated_cosine", best)},
        "found": {name: float(value.mean()) for name, value in found.items()},
        "best_cosine_blend": best, "best_negated_blend": best_negated,
        "negated_minus_best_blend": margin(found["negated_cosine"], found[best]),
        "best_negated_blend_minus_negated": margin(found[best_negated], found["negated_cosine"]),
        "best_blend_minus_cosine": margin(found[best], found["cosine"]),
        "v_negated_minus_cosine": margin(found["v_negated_cosine"], found["cosine"]),
        "groups": {}}
    for name, keep in groups.items():
        result["groups"][name] = {
            "contradictions": int(keep.sum()), "cosine": float(found["cosine"][keep].mean()),
            "negated_cosine": float(found["negated_cosine"][keep].mean()),
            "best_blend": float(found[best][keep].mean()),
            "negated_minus_cosine": margin(found["negated_cosine"][keep], found["cosine"][keep])}
    return result


def main(mnli, train, v5_ids, v7_ids, v5_npz, v7_npz, nli_tsv, output):
    if Path(output).exists():
        raise SystemExit(f"Refusing to replace {output}")
    fresh, v7, nli = load_fresh(mnli, v5_ids), load_selected(train, v7_ids), read_nli(nli_tsv)
    fresh_texts = sorted({t for s in statements(fresh) for t in (s, toggle_negation(s))})
    result = {}
    for encoder in ("text-embedding-3-small", "all-mpnet-base-v2"):
        if encoder == "text-embedding-3-small":
            table = {**openai_table(fresh_texts), **openai_table(texts_for(v7))}
        else:
            table = {}
            for path in (v5_npz, v7_npz):
                stored = np.load(path, allow_pickle=False)
                table.update(zip(stored["texts"].tolist(), stored["vectors"].astype(float)))
        result[encoder] = score(v7, table, direction(fresh, table), nli)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
