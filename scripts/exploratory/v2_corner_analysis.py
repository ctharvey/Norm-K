"""Exploratory: where does v2's transformed-sentence gain come from? Not preregistered.
Cached vectors only (max_new=0). argv: repo, mnli zip, wanli jsonl. Run with cwd = repo."""
import dataclasses
import json
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])

from sklearn.metrics import average_precision_score as ap

import norm_k.public_v2 as pv
from norm_k.ablation import BASE_FIELDS, CORNER_FIELDS, LEXICAL_FIELDS
from norm_k.counterfactuals import counterfactual_square
from norm_k.embeddings import EmbeddingCache, HashingEncoder, OpenAIEncoder
from norm_k.statistics import grouped_ap_delta_interval

SEED, REPS, SHUFFLES = 42, 2000, 5
NEGATION = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|nowhere|cannot|without)\b|n['’]t\b", re.I)
BASELINE = BASE_FIELDS + LEXICAL_FIELDS
data = pv.load_public(sys.argv[2], sys.argv[3])


def prepare(pairs, vectors):
    squares = [counterfactual_square(pair, "negation") for pair in pairs]
    rows = pv._rows(pairs, squares, vectors, {pair.id: 0.0 for pair in pairs})
    return squares, rows, pv._labels(pairs)


def interval(y, base, new, groups):
    found = grouped_ap_delta_interval(list(y), base, new, list(groups), repetitions=REPS, seed=SEED)
    return None if found is None else [round(found["ci_lower"], 4), round(found["ci_upper"], 4)]


def side_negated(text, toggled):
    return len(toggled) < len(text)  # The edit removed a written negation.


def strata(squares, rows):
    neg_a = np.asarray([side_negated(s.base, s.transformed_base) for s in squares])
    neg_b = np.asarray([side_negated(s.other, s.transformed_other) for s in squares])
    jaccard = np.asarray([row["token_jaccard"] for row in rows])
    low, high = np.quantile(jaccard, [1 / 3, 2 / 3])
    clean = np.asarray([not NEGATION.search(s.base) and not NEGATION.search(s.other) for s in squares])
    return {"no_negation_word_anywhere": clean,
            "only_B_negated": neg_b & ~neg_a, "only_A_negated": neg_a & ~neg_b,
            "both_negated": neg_a & neg_b, "neither_negated": ~neg_a & ~neg_b,
            "low_overlap": jaccard <= low, "mid_overlap": (jaccard > low) & (jaccard < high),
            "high_overlap": jaccard >= high}


def stratum_report(y, base, new, squares, rows):
    report = {}
    for name, keep in strata(squares, rows).items():
        positives = int(y[keep].sum())
        entry = {"pairs": int(keep.sum()), "positives": positives}
        if 0 < positives < keep.sum():
            entry.update(floor=round(float(y[keep].mean()), 3), base=round(float(ap(y[keep], base[keep])), 3),
                         new=round(float(ap(y[keep], new[keep])), 3))
            entry["delta"] = round(entry["new"] - entry["base"], 3)
            entry["mean_flip_ap"] = round(float(ap(y[keep], flip_scores(rows)["mean_flip"][keep])), 3)
        report[name] = entry
    return report


def flip_scores(rows):
    """Training-free: does negating one side move the pair closer together?"""
    best = np.asarray([max(r["cos_a_tb"], r["cos_b_ta"]) - r["cosine"] for r in rows])
    mean = np.asarray([(r["cos_a_tb"] + r["cos_b_ta"]) / 2 - r["cosine"] for r in rows])
    return {"best_flip": best, "mean_flip": mean, "plain_cosine": np.asarray([r["cosine"] for r in rows])}


def analyse(name, vectors):
    squares, rows, y = prepare(data.primary, vectors)
    groups = np.asarray([pair.relation_family for pair in data.primary])
    t_squares, t_rows, t_y = prepare(data.replication, vectors)

    def held(fields):
        return pv._held_out_scores("rf", pv._columns(rows, fields), y, groups, SEED)

    def transfer(fields):
        return pv._fit_score("rf", pv._columns(rows, fields), y, pv._columns(t_rows, fields), SEED)

    out = {"floor": float(y.mean()), "wanli_floor": float(t_y.mean())}
    base, full = held(BASELINE), held(BASELINE + CORNER_FIELDS)
    t_base, t_full = transfer(BASELINE), transfer(BASELINE + CORNER_FIELDS)
    out["reproduced"] = {"baseline": ap(y, base), "with_corners": ap(y, full),
                         "wanli_baseline": ap(t_y, t_base), "wanli_with_corners": ap(t_y, t_full)}

    subsets = {"A_only:norm_a_ta": ("norm_a_ta",), "B_only:norm_b_tb": ("norm_b_tb",),
               "single_statement:both_norms": ("norm_a_ta", "norm_b_tb"),
               "cross:cos_a_tb+cos_b_ta": ("cos_a_tb", "cos_b_ta"), "both_flipped:cos_ta_tb": ("cos_ta_tb",)}
    out["added_alone"] = {}
    for label, fields in subsets.items():
        score = held(BASELINE + fields)
        t_score = transfer(BASELINE + fields)
        out["added_alone"][label] = {"delta": round(ap(y, score) - ap(y, base), 4),
                                     "interval": interval(y, base, score, groups),
                                     "wanli_delta": round(ap(t_y, t_score) - ap(t_y, t_base), 4)}
    without_cross = held(BASELINE + ("cos_ta_tb", "norm_a_ta", "norm_b_tb"))
    out["removing_cross_from_full"] = round(ap(y, without_cross) - ap(y, full), 4)

    out["strata_primary"] = stratum_report(y, base, full, squares, rows)
    out["strata_wanli"] = stratum_report(t_y, t_base, t_full, t_squares, t_rows)

    out["training_free"] = {}
    for label, score in flip_scores(rows).items():
        t_score = flip_scores(t_rows)[label]
        out["training_free"][label] = {"mnli_ap": round(ap(y, score), 4), "mnli_ap_reversed": round(ap(y, -score), 4),
                                       "wanli_ap": round(ap(t_y, t_score), 4), "wanli_ap_reversed": round(ap(t_y, -t_score), 4)}

    flips, t_flips = flip_scores(rows), flip_scores(t_rows)
    out["mean_flip_over_reversed_cosine"] = {
        "mnli_interval": interval(y, -flips["plain_cosine"], flips["mean_flip"], groups),
        "wanli_interval": interval(t_y, -t_flips["plain_cosine"], t_flips["mean_flip"],
                                   [pair.id for pair in data.replication])}
    out["flip_cosine_correlation"] = round(float(np.corrcoef(flips["mean_flip"], flips["plain_cosine"])[0, 1]), 3)

    # Wrong partner: give each B the A of another pair in its genre. Any gain left
    # cannot come from the relation between the two statements.
    gains, base_aps, full_aps = [], [], []
    rng = np.random.default_rng(SEED)
    for _ in range(SHUFFLES):
        order = np.arange(len(y))
        for group in np.unique(groups):
            members = np.flatnonzero(groups == group)
            order[members] = rng.permutation(members)
        wrong_pairs = [dataclasses.replace(data.primary[i], text_a=data.primary[j].text_a)
                       for i, j in enumerate(order)]
        wrong_squares = [dataclasses.replace(squares[i], base=squares[j].base,
                                             transformed_base=squares[j].transformed_base)
                         for i, j in enumerate(order)]
        wrong_rows = pv._rows(wrong_pairs, wrong_squares, vectors, {p.id: 0.0 for p in wrong_pairs})
        b = pv._held_out_scores("rf", pv._columns(wrong_rows, BASELINE), y, groups, SEED)
        f = pv._held_out_scores("rf", pv._columns(wrong_rows, BASELINE + CORNER_FIELDS), y, groups, SEED)
        base_aps.append(ap(y, b))
        full_aps.append(ap(y, f))
        gains.append(ap(y, f) - ap(y, b))
    out["wrong_partner"] = {"mean_baseline": round(float(np.mean(base_aps)), 4),
                            "mean_with_corners": round(float(np.mean(full_aps)), 4),
                            "gains": [round(float(g), 4) for g in gains]}
    print(name, json.dumps(out, indent=1, default=float), flush=True)


texts = sorted({t for pairs in (data.primary, data.replication) for pair in pairs
                for s in [counterfactual_square(pair, "negation")]
                for t in (s.base, s.other, s.transformed_base, s.transformed_other)})
with EmbeddingCache(".norm_k_cache/embeddings.sqlite3") as cache:
    real = dict(zip(texts, cache.embed(OpenAIEncoder(), texts, max_new=0)))
    hashed = dict(zip(texts, cache.embed(HashingEncoder(), texts, max_new=0)))
analyse("text-embedding-3-small", real)
analyse("hash", hashed)
