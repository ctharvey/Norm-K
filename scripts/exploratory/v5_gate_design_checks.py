"""Exploratory design checks on seen v5 data: top-k gate headroom, and interval width at NaN-NLI's size.

v5_gate_design_checks.py MNLI_ZIP WANLI IDS_TSV MPNET_NPZ OUTPUT_JSON
Run with cwd = repo. Reads saved vectors only. Nothing here is a verdict.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as ap

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "scripts/exploratory"),
                 str(Path.home() / ".claude" / "tools")]

from norm_k.counterfactuals import toggle_negation
from norm_k.flip_v5 import POOL, matched_edit, pool_texts
from norm_k.public_v2 import load_public
from v5_flip_matched import load_fresh, statements
from v5_mutation_scores import openai_table, rows

mnli, wanli, ids, mpnet, output = sys.argv[1:6]
fresh, replication = load_fresh(mnli, ids), load_public(mnli, wanli).replication
sets = {"fresh": fresh, "wanli": replication}
texts = pool_texts(statements([*fresh, *replication]))


def topk(pairs, table):
    y = np.asarray([p.label == "contradiction" for p in pairs], dtype=int)
    a_t, b_t = [p.text_a for p in pairs], [p.text_b for p in pairs]
    a, b = rows(a_t, table), rows(b_t, table)
    na, nb = rows([toggle_negation(t) for t in a_t], table), rows([toggle_negation(t) for t in b_t], table)
    plain, negated = a @ b.T, (a @ nb.T + na @ b.T) / 2
    labels = np.zeros(plain.shape, dtype=int)
    labels[np.arange(len(y)), np.arange(len(y))] = y
    a_id = np.unique(a_t, return_inverse=True)[1]
    b_id = np.unique(b_t, return_inverse=True)[1]
    # Another pair with the same statements may carry a different gold label: leave those out.
    clash = (a_id[:, None] == a_id[None, :]) & (b_id[:, None] == b_id[None, :]) & ~np.eye(len(y), dtype=bool)
    out = {}
    for k in (5, 10, 20):
        near = np.argsort(-plain, axis=1)[:, :k]
        mask = np.zeros(plain.shape, dtype=bool)
        mask[np.arange(len(y))[:, None], near] = True
        mask &= ~clash
        cand_y, cos, neg = labels[mask], plain[mask], negated[mask]
        flip = neg - cos
        row = {"candidates": int(mask.sum()), "stage1_recall": round(cand_y.sum() / y.sum(), 3),
               "prevalence": round(float(cand_y.mean()), 3)}
        for share in (0.05, 0.10, 0.20):
            take = int(round(share * len(cand_y)))
            row[str(share)] = {name: round(float(cand_y[np.argsort(-s, kind="stable")[:take]].sum() / y.sum()), 3)
                               for name, s in (("cosine", cos), ("negcos", neg), ("flip", flip))}
        out[k] = row
    return out


def width(pairs, table, draws=30, boots=400):
    """Unclustered 95% interval width for flip vs matched at 258 pairs, 45% contradictions."""
    rng = np.random.default_rng(7)
    y = np.asarray([p.label == "contradiction" for p in pairs], dtype=int)
    assigned = {t: POOL[matched_edit(t, table)[0]](t) for t in statements(pairs)}
    a_t, b_t = [p.text_a for p in pairs], [p.text_b for p in pairs]
    a, b = rows(a_t, table), rows(b_t, table)

    def flip(edit):
        na, nb = rows([edit(t) for t in a_t], table), rows([edit(t) for t in b_t], table)
        return ((a * nb).sum(1) + (b * na).sum(1)) / 2 - (a * b).sum(1)

    neg, mat = flip(toggle_negation), flip(assigned.__getitem__)
    pos, other = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    widths, deltas = [], []
    for _ in range(draws):
        pick = np.r_[rng.choice(pos, 117, replace=False), rng.choice(other, 141, replace=False)]
        found = []
        for _ in range(boots):
            s = rng.choice(pick, len(pick))
            found.append(ap(y[s], neg[s]) - ap(y[s], mat[s]))
        low, high = np.percentile(found, (2.5, 97.5))
        widths.append(high - low)
        deltas.append(ap(y[pick], neg[pick]) - ap(y[pick], mat[pick]))
    return {"mean_width": round(float(np.mean(widths)), 3), "delta_sd": round(float(np.std(deltas)), 3)}


result = {}
for encoder in ("openai", "mpnet"):
    if encoder == "openai":
        table = openai_table(texts)
    else:
        stored = np.load(mpnet, allow_pickle=False)
        table = dict(zip(stored["texts"].tolist(), stored["vectors"].astype(float)))
    for name, pairs in sets.items():
        result[f"{encoder}/{name}"] = {"topk": topk(pairs, table), "width": width(pairs, table)}
        print(encoder, name, json.dumps(result[f"{encoder}/{name}"]), flush=True)
    del table
Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
