"""Size-matched test of the flip score. See docs/preregistration-flip-matched-v5.md."""

from __future__ import annotations

import re
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


def score_set(pairs: Sequence[Pair], vectors: dict[str, np.ndarray], seed: int = 42, repetitions: int = 2000,
              hash_delta: float | None = None) -> dict:
    from sklearn.metrics import average_precision_score as ap

    y = np.asarray([pair.label == "contradiction" for pair in pairs], dtype=int)
    ids = [pair.id for pair in pairs]
    statements = sorted({t for p in pairs for t in (p.text_a, p.text_b)})
    assigned = {text: POOL[matched_edit(text, vectors)[0]](text) for text in statements}

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
    return {
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
