"""Exploratory: score mutations and gate headroom on v5's saved vectors. Run with cwd = repo.

v5_mutation_scores.py MNLI_ZIP WANLI IDS_TSV MPNET_NPZ OUTPUT_JSON

Reads saved vectors only, and never calls an embedding API. Every pair here was already seen,
so nothing from this script is a verdict.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as ap

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path.home() / ".claude" / "tools"))

from norm_k.counterfactuals import toggle_negation  # noqa: E402
from norm_k.embeddings import EmbeddingCache  # noqa: E402
from norm_k.flip_v5 import POOL, matched_edit, pool_texts  # noqa: E402
from norm_k.public_v2 import load_public  # noqa: E402
from progress import Progress  # noqa: E402
from v5_flip_matched import load_fresh, statements  # noqa: E402

SEED, REPETITIONS, FLOOR = 42, 2000, 0.01  # FLOOR matches mutation_sanity.py.
SCORES = ("flip", "minimum_gain", "normalized_flip", "anti_alignment", "pairwise_term", "individual_term")


@dataclass
class Corners:
    a: np.ndarray
    b: np.ndarray
    na: np.ndarray
    nb: np.ndarray


def rows(texts, table):
    found = np.asarray([table[text] for text in texts], dtype=float)
    return found / np.linalg.norm(found, axis=1, keepdims=True)


def mutations(c: Corners) -> dict[str, np.ndarray]:
    da, db = c.na - c.a, c.nb - c.b
    gain_a, gain_b = (da * c.b).sum(1), (c.a * db).sum(1)
    move_a, move_b = 1 - (c.a * c.na).sum(1), 1 - (c.b * c.nb).sum(1)
    size = np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1)
    flip = (gain_a + gain_b) / 2
    pairwise, individual = ((da - db) * (c.b - c.a)).sum(1) / 2, -(move_a + move_b) / 2
    assert np.abs(flip - pairwise - individual).max() < 1e-9
    return {"flip": flip, "minimum_gain": np.minimum(gain_a, gain_b),
            "normalized_flip": (gain_a / np.maximum(move_a, FLOOR) + gain_b / np.maximum(move_b, FLOOR)) / 2,
            "anti_alignment": -(da * db).sum(1) / np.maximum(size, 1e-12),
            "pairwise_term": pairwise, "individual_term": individual}


def paired_interval(y, first, second, rng):
    found = []
    for _ in range(REPETITIONS):
        sample = rng.integers(0, len(y), len(y))
        if 0 < y[sample].sum() < len(y):
            found.append(ap(y[sample], first[sample]) - ap(y[sample], second[sample]))
    return [float(v) for v in np.percentile(found, (2.5, 97.5))]


def wrong_partner(y, genres, negated: Corners, matched: Corners, rng, shuffles=5):
    found = {name: {"negation": [], "matched": []} for name in SCORES}
    for _ in range(shuffles):
        order = np.arange(len(y))
        for genre in np.unique(genres):
            members = np.flatnonzero(genres == genre)
            order[members] = rng.permutation(members)
        for label, c in (("negation", negated), ("matched", matched)):
            scores = mutations(Corners(c.a[order], c.b, c.na[order], c.nb))
            for name in SCORES:
                found[name][label].append(ap(y, scores[name]))
    return {name: {label: float(np.mean(values)) for label, values in value.items()} for name, value in found.items()}


def unrelated_opposition(y, c: Corners, anti):
    """Measured stand-in for the synthetic reflection's 50.59%: anti-alignment on every cross pair."""
    da, db = c.na - c.a, c.nb - c.b
    da /= np.maximum(np.linalg.norm(da, axis=1, keepdims=True), 1e-12)
    db /= np.maximum(np.linalg.norm(db, axis=1, keepdims=True), 1e-12)
    random = -(da @ db.T)[~np.eye(len(y), dtype=bool)]

    def describe(values):
        return {"mean": float(values.mean()), "above_0.5": float((values > 0.5).mean()),
                "above_0.99": float((values > 0.99).mean())}

    return {"random_pairs": describe(random), "contradictions": describe(anti[y == 1]),
            "other_real_pairs": describe(anti[y == 0])}


def recall(labels, score, budget, allowed=None):
    """Recall of the top budget share of the whole pool, optionally only from allowed rows."""
    index = np.arange(len(labels)) if allowed is None else np.flatnonzero(allowed)
    take = int(round(budget * len(labels)))
    top = index[np.argsort(-score[index], kind="stable")[:take]]
    return float(labels[top].sum() / labels.sum())


def pool_recalls(c: Corners, y, a_texts, b_texts, rng, hard: bool):
    """1% contradiction pool. hard takes each A's nearest other B; otherwise random, as in v5."""
    size = len(y)
    plain = c.a @ c.b.T
    negated = (c.a @ c.nb.T + c.na @ c.b.T) / 2
    a_id, b_id = np.unique(a_texts, return_inverse=True)[1], np.unique(b_texts, return_inverse=True)[1]
    related = (a_id[:, None] == a_id[None, :]) | (b_id[:, None] == b_id[None, :])
    wanted = int(round(y.sum() / 0.01)) - size
    if hard:
        each = int(np.ceil(wanted / size))
        masked = np.where(related, -np.inf, plain)
        near = np.argpartition(-masked, each, axis=1)[:, :each]
        cross = (np.arange(size)[:, None] * size + near).ravel()
        cross = cross[np.isfinite(masked.ravel()[cross])]
        cross = rng.choice(cross, min(wanted, len(cross)), replace=False)
    else:
        cross = rng.choice(np.flatnonzero(~related.ravel()), wanted, replace=False)
    chosen = np.concatenate([np.arange(size) * (size + 1), cross])
    labels = np.concatenate([y, np.zeros(len(cross), dtype=int)])
    cosine, negcos = plain.ravel()[chosen], negated.ravel()[chosen]
    flip = negcos - cosine
    found = {"pool": len(labels), "prevalence": float(labels.mean()),
             "real_pairs_in_cosine_top": {str(s): recall(np.r_[np.ones(size), np.zeros(len(cross))].astype(int),
                                                         cosine, s) for s in (0.05, 0.10, 0.20)}}
    for budget in (0.01, 0.02):
        row = {"cosine": recall(labels, cosine, budget), "negated_cosine": recall(labels, negcos, budget),
               "flip": recall(labels, flip, budget)}
        for stage in (0.05, 0.10, 0.20):
            cut = np.quantile(cosine, 1 - stage)
            row[f"cosine_top_{stage}_then_flip"] = recall(labels, flip, budget, cosine >= cut)
        found[f"budget_{budget}"] = row
    return found


def evaluate(pairs, table):
    rng = np.random.default_rng(SEED)
    y = np.asarray([p.label == "contradiction" for p in pairs], dtype=int)
    genres = np.asarray([p.relation_family for p in pairs])
    a_texts, b_texts = [p.text_a for p in pairs], [p.text_b for p in pairs]
    assigned = {t: POOL[matched_edit(t, table)[0]](t) for t in statements(pairs)}
    a, b = rows(a_texts, table), rows(b_texts, table)
    negated = Corners(a, b, rows([toggle_negation(t) for t in a_texts], table),
                      rows([toggle_negation(t) for t in b_texts], table))
    matched = Corners(a, b, rows([assigned[t] for t in a_texts], table), rows([assigned[t] for t in b_texts], table))
    with_negation, with_matched = mutations(negated), mutations(matched)
    plain = (a * b).sum(1)
    result = {"pairs": len(y), "positives": int(y.sum()), "floor": float(y.mean()),
              "cosine_ap": float(ap(y, plain)), "reversed_cosine_ap": float(ap(y, -plain)), "scores": {}}
    wrong = wrong_partner(y, genres, negated, matched, rng)
    for name in SCORES:
        negation_ap, matched_ap = float(ap(y, with_negation[name])), float(ap(y, with_matched[name]))
        result["scores"][name] = {
            "negation_ap": negation_ap, "matched_ap": matched_ap, "delta": negation_ap - matched_ap,
            "delta_interval": paired_interval(y, with_negation[name], with_matched[name], rng),
            "minus_flip": negation_ap - float(ap(y, with_negation["flip"])),
            "minus_flip_interval": None if name == "flip" else
            paired_interval(y, with_negation[name], with_negation["flip"], rng),
            "wrong_partner": wrong[name],
            "wrong_partner_delta": wrong[name]["negation"] - wrong[name]["matched"]}
    result["spearman_with_flip"] = {
        name: float(np.corrcoef(np.argsort(np.argsort(with_negation[name])),
                                np.argsort(np.argsort(with_negation["flip"])))[0, 1]) for name in SCORES}
    result["unrelated_opposition"] = unrelated_opposition(y, negated, with_negation["anti_alignment"])
    result["random_pool"] = pool_recalls(negated, y, a_texts, b_texts, rng, hard=False)
    result["hard_pool"] = pool_recalls(negated, y, a_texts, b_texts, rng, hard=True)
    return result


def openai_table(texts):
    class Identity:  # Cache lookups only. This has no encode method, so it can't reach the API.
        name, dimensions = "text-embedding-3-small", 1536

    with EmbeddingCache(ROOT / ".norm_k_cache/embeddings.sqlite3") as cache:
        table = {text: cache.lookup(Identity, text) for text in texts}
    missing = sum(value is None for value in table.values())
    if missing:
        raise SystemExit(f"{missing} of {len(texts)} texts are missing from the cache")
    return table


def main(mnli, wanli, ids_tsv, mpnet, output):
    if Path(output).exists():
        raise SystemExit(f"Refusing to replace {output}")
    fresh, replication = load_fresh(mnli, ids_tsv), load_public(mnli, wanli).replication
    sets = {"fresh_mnli_train": fresh, "v2_wanli": replication}
    texts = pool_texts(statements([*fresh, *replication]))
    progress = Progress("norm-k-mutations", "Norm-K mutation scores", total=4, unit="runs")
    result, done = {}, 0
    try:
        for encoder in ("text-embedding-3-small", "all-mpnet-base-v2"):
            if encoder == "all-mpnet-base-v2":
                stored = np.load(mpnet, allow_pickle=False)
                table = dict(zip(stored["texts"].tolist(), stored["vectors"].astype(float)))
            else:
                table = openai_table(texts)
            result[encoder] = {}
            for name, pairs in sets.items():
                result[encoder][name] = evaluate(pairs, table)
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
