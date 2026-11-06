"""Size-matched test of the flip score. See docs/preregistration-flip-matched-v5.md."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Sequence

import numpy as np

from .counterfactuals import toggle_negation
from .flip_v4 import _VERB_PATTERNS, NEGATION_WORD, Edit, _measure, placebo_edit
from .geometry import cosine
from .schema import Pair

# Meaning-preserving edits, smallest to largest in intent. Fixed before any test pair is embedded.
POOL: dict[str, Edit] = {
    "in_fact": lambda text: placebo_edit(text, "in fact"),
    "note": lambda text: f"Note: {text}",
    "as_it_happens": lambda text: placebo_edit(text, "as it happens"),
    "plainly": lambda text: f"To put it plainly: {text}",
    "for_consideration": lambda text: f"Here is a statement for your consideration: {text}",
    "read_carefully": lambda text: f"{text} Please read this statement carefully.",
    "record": lambda text: f"{text} This sentence is part of a longer written record.",
    "record_office": lambda text: f"{text} This sentence is part of a longer written record kept in an office.",
    "copied_line": lambda text: f"Copied line: {text} End of line.",
    "wrapper": lambda text: f"The following line was copied without changes from a document: {text} End of the copied line.",
}
VERB_EDITS = ("in_fact", "as_it_happens")  # Pool edits that sit where negation does.
SET_BUDGETS = (0.10, 0.20, 0.50)  # Share of a test set passed on.
POOL_BUDGETS = (0.01, 0.02, 0.05, 0.10)  # Share of a diluted pool passed on.
PREVALENCES = (0.01, 0.001)
GATE_PREVALENCE, GATE_BUDGET, GATE_MIN_MARGIN = 0.01, 0.05, 0.05


def tense_edit(text: str) -> str | None:
    """Meaning-changing, non-negating, reported only: swap the verb's tense."""
    for pattern in _VERB_PATTERNS:
        result, count = re.subn(pattern, lambda found: {"is": "was", "was": "is"}[found.group(0)], text, count=1)
        if count:
            return result
    return None


def pool_texts(statements: Sequence[str]) -> list[str]:
    texts: set[str] = set()
    for text in statements:
        texts.update((text, toggle_negation(text), tense_edit(text)))
        texts.update(edit(text) for edit in POOL.values())
    return sorted(texts)


def distance(text: str, edited: str, vectors: dict[str, np.ndarray]) -> float:
    return 1 - cosine(vectors[text], vectors[edited])


def matched_edit(text: str, vectors: dict[str, np.ndarray]) -> tuple[str, float, float]:
    """Pool edit closest in size to negation on this statement. Uses no labels or partner."""
    target = distance(text, toggle_negation(text), vectors)
    name = min(POOL, key=lambda item: (abs(distance(text, POOL[item](text), vectors) - target), item))
    return name, distance(text, POOL[name](text), vectors), target


def _flip(pair: Pair, a_edit: str, b_edit: str, vectors: dict[str, np.ndarray]) -> float:
    a, b = vectors[pair.text_a], vectors[pair.text_b]
    return (cosine(a, vectors[b_edit]) + cosine(b, vectors[a_edit])) / 2 - cosine(a, b)


def match_quality(statements: Sequence[str], vectors: dict[str, np.ndarray]) -> dict:
    rows = [matched_edit(text, vectors) for text in statements]
    gaps = np.asarray([row[1] - row[2] for row in rows])
    names, counts = np.unique([row[0] for row in rows], return_counts=True)
    return {"statements": len(rows), "negation_mean": float(np.mean([row[2] for row in rows])),
            "matched_mean": float(np.mean([row[1] for row in rows])), "mean_gap": float(gaps.mean()),
            "median_abs_gap": float(np.median(np.abs(gaps))),
            "chosen": {str(n): int(c) for n, c in zip(names, counts)}}


def gate_passes(quality: dict) -> bool:
    return abs(quality["mean_gap"]) <= 0.01 and quality["median_abs_gap"] <= 0.02


def _top(scores: np.ndarray, share: float) -> np.ndarray:
    """Mask of the highest-scoring share of items, which is what a gate would pass on."""
    keep = int(np.ceil(share * len(scores)))
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:keep]] = True
    return mask


def recall_at(y: np.ndarray, scores: np.ndarray, share: float) -> float:
    return float(y[_top(scores, share)].sum() / y.sum())


def _rows(texts: Sequence[str], vectors: dict[str, np.ndarray]) -> np.ndarray:
    rows = np.asarray([vectors[text] for text in texts], dtype=float)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def diluted_pool(pairs: Sequence[Pair], y: np.ndarray, assigned: dict[str, str], vectors: dict[str, np.ndarray],
                 prevalence: float, seed: int) -> dict:
    """Real pairs plus random A-with-other-B pairs, which stand in for unrelated negatives."""
    a, b = [p.text_a for p in pairs], [p.text_b for p in pairs]
    A, B = _rows(a, vectors), _rows(b, vectors)
    plain = A @ B.T
    negated = (A @ _rows([toggle_negation(t) for t in b], vectors).T
               + _rows([toggle_negation(t) for t in a], vectors) @ B.T) / 2
    matched = (A @ _rows([assigned[t] for t in b], vectors).T + _rows([assigned[t] for t in a], vectors) @ B.T) / 2
    size = len(pairs)
    cross = np.flatnonzero(~np.eye(size, dtype=bool).ravel())
    wanted = min(int(round(y.sum() / prevalence)) - size, len(cross))
    chosen = np.concatenate([np.arange(size) * (size + 1),
                             np.random.default_rng(seed).choice(cross, max(wanted, 0), replace=False)])
    labels = np.concatenate([y, np.zeros(len(chosen) - size, dtype=int)])
    return {"y": labels, "cosine": plain.ravel()[chosen], "negated_cosine": negated.ravel()[chosen],
            "flip_negation": (negated - plain).ravel()[chosen], "flip_matched": (matched - plain).ravel()[chosen]}


def gate_margin(pool: dict, score: str, seed: int, repetitions: int) -> dict:
    """Recall over the stronger cosine direction at the fixed budget. Resamples positives; cutoffs stay fixed."""
    y = pool["y"]
    found = _top(pool[score], GATE_BUDGET)[y == 1].astype(float)
    baseline = max((_top(pool["cosine"], GATE_BUDGET)[y == 1], _top(-pool["cosine"], GATE_BUDGET)[y == 1]),
                   key=lambda hits: hits.sum()).astype(float)
    gaps = found - baseline
    draws = np.random.default_rng(seed).integers(0, len(gaps), size=(repetitions, len(gaps)))
    means = gaps[draws].mean(axis=1)
    return {"margin": float(gaps.mean()), "recall": float(found.mean()), "cosine_recall": float(baseline.mean()),
            "ci_lower": float(np.quantile(means, 0.025)), "ci_upper": float(np.quantile(means, 0.975))}


def gate_verdict(primary: dict, replication_margin: float, hash_margin: float | None) -> str:
    """The gate rule. A missing hash margin means this is the hash dry run."""
    if (primary["margin"] >= GATE_MIN_MARGIN and primary["ci_lower"] > 0 and replication_margin > 0
            and (hash_margin is None or primary["margin"] > hash_margin)):
        return "GATE-PASS"
    return "GATE-FAIL" if primary["ci_upper"] < GATE_MIN_MARGIN else "UNCLEAR"


def gate_measures(pairs: Sequence[Pair], y: np.ndarray, assigned: dict[str, str], vectors: dict[str, np.ndarray],
                  seed: int, repetitions: int) -> dict:
    pools = {prevalence: diluted_pool(pairs, y, assigned, vectors, prevalence, seed) for prevalence in PREVALENCES}
    names = ("flip_negation", "negated_cosine", "flip_matched", "cosine")
    primary = pools[GATE_PREVALENCE]
    return {
        "primary": gate_margin(primary, "flip_negation", seed, repetitions),
        "negated_cosine": gate_margin(primary, "negated_cosine", seed, repetitions),
        "diluted": {str(prevalence): {
            "pool": len(pool["y"]), "prevalence": float(pool["y"].mean()),
            "recall": {name: {str(share): recall_at(pool["y"], pool[name], share) for share in POOL_BUDGETS}
                       for name in names},
            "reversed_cosine": {str(share): recall_at(pool["y"], -pool["cosine"], share) for share in POOL_BUDGETS},
        } for prevalence, pool in pools.items()},
    }


def wrong_partner(pairs: Sequence[Pair], y: np.ndarray, assigned: dict[str, str], vectors: dict[str, np.ndarray],
                  seed: int, shuffles: int = 5) -> dict:
    """Mean AP when each B gets the A of another pair in its genre, as in v4."""
    from sklearn.metrics import average_precision_score as ap

    rng, genres = np.random.default_rng(seed), np.asarray([p.relation_family for p in pairs])
    found: dict[str, list[float]] = {"negation": [], "matched": []}
    for _ in range(shuffles):
        order = np.arange(len(pairs))
        for genre in np.unique(genres):
            members = np.flatnonzero(genres == genre)
            order[members] = rng.permutation(members)
        wrong = [replace(pair, text_a=pairs[j].text_a) for pair, j in zip(pairs, order)]
        for name, edit in (("negation", toggle_negation), ("matched", assigned.__getitem__)):
            found[name].append(ap(y, np.asarray([_flip(p, edit(p.text_a), edit(p.text_b), vectors) for p in wrong])))
    means = {name: float(np.mean(values)) for name, values in found.items()}
    return {**means, "delta": means["negation"] - means["matched"]}


def score_set(pairs: Sequence[Pair], vectors: dict[str, np.ndarray], seed: int = 42, repetitions: int = 2000,
              hash_delta: float | None = None) -> dict:
    from sklearn.metrics import average_precision_score as ap

    y = np.asarray([pair.label == "contradiction" for pair in pairs], dtype=int)
    ids = [pair.id for pair in pairs]
    statements = sorted({t for p in pairs for t in (p.text_a, p.text_b)})
    chosen = {text: matched_edit(text, vectors)[0] for text in statements}
    assigned = {text: POOL[name](text) for text, name in chosen.items()}

    def scores(edit) -> np.ndarray:
        return np.asarray([_flip(p, edit(p.text_a), edit(p.text_b), vectors) for p in pairs])

    negation, matched = scores(toggle_negation), scores(assigned.__getitem__)
    fixed = {name: scores(edit) for name, edit in POOL.items()}
    tense = scores(tense_edit)
    moved = {name: float(np.mean([distance(t, edit(t), vectors) for t in statements]))
             for name, edit in {**POOL, "negation": toggle_negation, "tense": tense_edit}.items()}
    largest = max(POOL, key=lambda name: moved[name])
    clean = np.asarray([not NEGATION_WORD.search(p.text_a) and not NEGATION_WORD.search(p.text_b) for p in pairs])
    quality = match_quality(statements, vectors)
    measure = _measure(y, negation, matched, ids, seed, repetitions, hash_delta, "SURVIVES-MATCHING", "SIZE-EXPLAINED")
    if not gate_passes(quality):
        measure["verdict"] = "UNCLEAR"
        measure["gate_failed"] = True
    plain = np.asarray([cosine(vectors[p.text_a], vectors[p.text_b]) for p in pairs])
    a_size, b_size = ([distance(t, toggle_negation(t), vectors) for t in side]
                      for side in ([p.text_a for p in pairs], [p.text_b for p in pairs]))
    size, b_size = np.asarray(a_size) + np.asarray(b_size), np.asarray(b_size)
    at_verb = [[chosen[t] in VERB_EDITS for t in (p.text_a, p.text_b)] for p in pairs]

    def subset(mask: np.ndarray) -> dict:
        found = {"pairs": int(mask.sum()), "positives": int(y[mask].sum())}
        if 0 < y[mask].sum() < mask.sum():
            found.update(flip_negation=float(ap(y[mask], negation[mask])), flip_matched=float(ap(y[mask], matched[mask])))
        return found

    return {
        "wrong_partner_ap": wrong_partner(pairs, y, assigned, vectors, seed),
        "statement_only_ap": {"edit_size": float(ap(y, size)), "reversed_edit_size": float(ap(y, -size)),
                              "b_edit_size": float(ap(y, b_size)), "reversed_b_edit_size": float(ap(y, -b_size))},
        "matched_at_verb": {"both_statements": subset(np.asarray([all(row) for row in at_verb])),
                            "either_statement": subset(np.asarray([any(row) for row in at_verb]))},
        "recall_at_budget": {name: {str(share): recall_at(y, score, share) for share in SET_BUDGETS}
                             for name, score in {"flip_negation": negation, "flip_matched": matched,
                                                 "cosine": plain, "reversed_cosine": -plain}.items()},
        "gate": gate_measures(pairs, y, assigned, vectors, seed, repetitions),
        "pairs": len(y), "positives": int(y.sum()), "floor": float(y.mean()),
        "ap": {"flip_negation": float(ap(y, negation)), "flip_matched": float(ap(y, matched)),
               "flip_tense": float(ap(y, tense)),
               **{f"flip_{name}": float(ap(y, score)) for name, score in fixed.items()}},
        "negation_vs_matched": measure, "match_quality": quality, "gate_passed": gate_passes(quality),
        "mean_cosine_distance_moved": moved, "largest_edit": largest,
        "no_negation_words": {"pairs": int(clean.sum()), "floor": float(y[clean].mean()),
                              "flip_negation": float(ap(y[clean], negation[clean])),
                              "flip_matched": float(ap(y[clean], matched[clean]))},
    }
