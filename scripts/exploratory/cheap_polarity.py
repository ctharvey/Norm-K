"""Exploratory: negation-mismatch baseline, average negation vector, shifted search, polarity score.

cheap_polarity.py MNLI_ZIP WANLI WANLI_TRAIN V5_IDS V7_IDS V5_MPNET_NPZ V7_MPNET_NPZ NLI_TSV OUTPUT_JSON
Run with cwd = repo, after v7's report. Reads saved vectors only, and never calls an embedding API.
v is always estimated on a different set from the one it is scored on. Nothing here is a verdict.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as ap

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts/exploratory"),
                str(Path.home() / ".claude" / "tools")]

from norm_k.counterfactuals import toggle_negation  # noqa: E402
from norm_k.flip_v4 import NEGATION_WORD  # noqa: E402
from norm_k.gate_v7 import Candidates, hits, passed  # noqa: E402
from norm_k.public_v2 import load_public  # noqa: E402
from progress import Progress  # noqa: E402
from v5_flip_matched import load_fresh, statements  # noqa: E402
from v5_mutation_scores import openai_table, paired_interval, rows  # noqa: E402
from v7_negated_gate import load_selected, read_nli, texts_for  # noqa: E402

SEED = 42


def has_negation(texts):
    return np.asarray([bool(NEGATION_WORD.search(t)) for t in texts])


def unit(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def direction(pairs, table):
    """Mean move from a statement with no negation word to its negated form."""
    texts = [t for t in statements(pairs) if not NEGATION_WORD.search(t)]
    return (rows([toggle_negation(t) for t in texts], table) - rows(texts, table)).mean(0)


def shifted(matrix, texts, v, signed=True):
    """Stand-in for the negated form. Signed moves statements with a negation word the other way."""
    sign = np.where(has_negation(texts), -1.0, 1.0) if signed else np.ones(len(texts))
    return unit(matrix + sign[:, None] * v)


def pair_scores(a_texts, b_texts, a, b, na, nb, v):
    plain = (a * b).sum(1)
    negated = ((a * nb).sum(1) + (na * b).sum(1)) / 2
    sa, sb = shifted(a, a_texts, v), shifted(b, b_texts, v)
    ua, ub = shifted(a, a_texts, v, False), shifted(b, b_texts, v, False)
    v_negated = ((a * sb).sum(1) + (sa * b).sum(1)) / 2
    mismatch = (has_negation(a_texts) != has_negation(b_texts)).astype(float)
    gap = np.abs(a @ v - b @ v)
    return {"cosine": plain, "reversed_cosine": -plain, "flip": negated - plain, "negated_cosine": negated,
            "mismatch": mismatch, "mismatch_then_cosine": 2 * mismatch + plain,
            "mismatch_then_reversed_cosine": 2 * mismatch - plain,
            "v_flip": v_negated - plain, "v_negated_cosine": v_negated,
            "v_flip_unsigned": ((a * ub).sum(1) + (ua * b).sum(1)) / 2 - plain,
            "polarity_gap": gap, "polarity_gap_times_cosine": gap * np.maximum(plain, 0)}


def pair_table(pairs, table, v):
    rng = np.random.default_rng(SEED)
    y = np.asarray([p.label == "contradiction" for p in pairs], dtype=int)
    a_texts, b_texts = [p.text_a for p in pairs], [p.text_b for p in pairs]
    a, b = rows(a_texts, table), rows(b_texts, table)
    na, nb = rows([toggle_negation(t) for t in a_texts], table), rows([toggle_negation(t) for t in b_texts], table)
    scores = pair_scores(a_texts, b_texts, a, b, na, nb, v)
    clean = ~has_negation(a_texts) & ~has_negation(b_texts)
    result = {"pairs": len(y), "positives": int(y.sum()), "floor": float(y.mean()),
              "clean_pairs": int(clean.sum()), "clean_floor": float(y[clean].mean()), "scores": {}}
    genres = np.asarray([p.relation_family for p in pairs])
    wrong = {name: [] for name in ("flip", "v_flip", "polarity_gap")}
    for _ in range(5):
        order = np.arange(len(y))
        for genre in np.unique(genres):
            members = np.flatnonzero(genres == genre)
            order[members] = rng.permutation(members)
        shuffled = pair_scores([a_texts[i] for i in order], b_texts, a[order], b, na[order], nb, v)
        for name in wrong:
            wrong[name].append(ap(y, shuffled[name]))
    for name, score in scores.items():
        result["scores"][name] = {"ap": float(ap(y, score)), "clean_ap": float(ap(y[clean], score[clean]))}
    for name in ("mismatch_then_cosine", "mismatch_then_reversed_cosine", "v_flip", "polarity_gap"):
        result["scores"][name]["minus_flip_interval"] = paired_interval(y, scores[name], scores["flip"], rng)
    for name, values in wrong.items():
        result["scores"][name]["wrong_partner_ap"] = float(np.mean(values))
    return result


def reach(found, contradictions):
    return float(np.mean([key in found for key in contradictions]))


def retrieval(pairs, table, v, nli=None):
    """Who gets retrieved, and what each ranking passes on, over every A and B in the set."""
    a_texts, b_texts = sorted({p.text_a for p in pairs}), sorted({p.text_b for p in pairs})
    contradictions = [(p.text_a, p.text_b) for p in pairs if p.label == "contradiction"]
    a, b = rows(a_texts, table), rows(b_texts, table)
    na = rows([toggle_negation(t) for t in a_texts], table)
    nb = rows([toggle_negation(t) for t in b_texts], table)

    def neighbors(query, k):
        order = np.argsort(-(query @ b.T), axis=1, kind="stable")[:, :k]
        return {(a_texts[i], b_texts[j]) for i, line in enumerate(order) for j in line}

    normal5, normal10 = neighbors(a, 5), neighbors(a, 10)
    true5, v5 = neighbors(na, 5), neighbors(shifted(a, a_texts, v), 5)
    result = {"contradictions": len(contradictions), "reach": {
        "cosine_top5": reach(normal5, contradictions), "cosine_top10": reach(normal10, contradictions),
        "negated_query_top5": reach(true5, contradictions), "v_query_top5": reach(v5, contradictions),
        "cosine5_plus_negated5": reach(normal5 | true5, contradictions),
        "cosine5_plus_v5": reach(normal5 | v5, contradictions)},
        "candidates": {"cosine_top10": len(normal10), "cosine5_plus_negated5": len(normal5 | true5),
                       "cosine5_plus_v5": len(normal5 | v5)}}
    index_a, index_b = {t: i for i, t in enumerate(a_texts)}, {t: i for i, t in enumerate(b_texts)}
    ordered = sorted(normal5)
    i, j = [index_a[x] for x, _ in ordered], [index_b[y] for _, y in ordered]
    scores = pair_scores([x for x, _ in ordered], [y for _, y in ordered], a[i], b[j], na[i], nb[j], v)
    held = Candidates([x for x, _ in ordered], [y for _, y in ordered], scores["cosine"], scores["negated_cosine"])
    rank = {name: scores[name] for name in ("cosine", "negated_cosine", "flip", "mismatch_then_cosine", "v_flip",
                                            "v_negated_cosine", "polarity_gap_times_cosine")}
    result["recall_at_10_percent"] = {name: float(hits(held, passed(s), contradictions, None).mean())
                                      for name, s in rank.items()}
    if nli is not None:
        result["found_after_nli"] = {name: float(hits(held, passed(s), contradictions, nli).mean())
                                     for name, s in rank.items()}
    return result


def main(mnli, wanli, train, v5_ids, v7_ids, v5_npz, v7_npz, nli_tsv, output):
    if Path(output).exists():
        raise SystemExit(f"Refusing to replace {output}")
    fresh, seen_wanli = load_fresh(mnli, v5_ids), load_public(mnli, wanli).replication
    v7 = load_selected(train, v7_ids)
    nli = read_nli(nli_tsv)
    v5_texts = sorted({t for s in statements([*fresh, *seen_wanli]) for t in (s, toggle_negation(s))})
    progress = Progress("norm-k-cheap-polarity", "Norm-K cheap polarity checks", total=6, unit="runs")
    result, done = {}, 0
    try:
        for encoder in ("text-embedding-3-small", "all-mpnet-base-v2"):
            if encoder == "text-embedding-3-small":
                table = {**openai_table(v5_texts), **openai_table(texts_for(v7))}
            else:
                table = {}
                for path in (v5_npz, v7_npz):
                    stored = np.load(path, allow_pickle=False)
                    table.update(zip(stored["texts"].tolist(), stored["vectors"].astype(float)))
            from_fresh, from_wanli = direction(fresh, table), direction(seen_wanli, table)
            result[encoder] = {"v_norm": float(np.linalg.norm(from_fresh)),
                               "v_cosine_across_sets": float(from_fresh @ from_wanli / (
                                   np.linalg.norm(from_fresh) * np.linalg.norm(from_wanli)))}
            for name, pairs, v, reads in (("fresh", fresh, from_wanli, None), ("wanli", seen_wanli, from_fresh, None),
                                          ("v7", v7, from_fresh, nli)):
                result[encoder][name] = {"pairs": pair_table(pairs, table, v),
                                         "retrieval": retrieval(pairs, table, v, reads)}
                done += 1
                progress.update(done=done, note=f"{encoder} {name}")
            del table
    except BaseException as error:
        progress.finish("failed", note=type(error).__name__)
        raise
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress.finish()


if __name__ == "__main__":
    main(*sys.argv[1:])
