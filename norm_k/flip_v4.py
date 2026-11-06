"""Placebo test of the flip score. See docs/preregistration-flip-placebo-v4.md."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import replace
from typing import Callable, Iterable, Sequence

import numpy as np

from .ablation import MIN_USEFUL_DELTA_AP
from .counterfactuals import toggle_negation
from .geometry import cosine
from .public_v2 import eligible, mnli_pairs
from .schema import Pair
from .statistics import grouped_ap_delta_interval

NEUTRAL_EDITS = ("in_fact", "note")  # Primary comparators.
MEANING_EDIT = "possibly"  # Changes meaning without negating. Reported only.
FRESH_PAIRS = 1500
SELECTION_SEED = "2029"
NEGATION_WORD = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|nowhere|cannot|without)\b"
                           r"|n['’]t\b", re.I)
# Same order as the negation rule, so a placebo lands on the verb negation would edit.
_VERB_PATTERNS = (r"\bis(?= not\b)", r"\bwas(?= not\b)", r"\bis\b", r"\bwas\b")

Edit = Callable[[str], "str | None"]


def placebo_edit(text: str, word: str) -> str | None:
    """Insert a word right after the is or was that the negation rule would edit."""
    for pattern in _VERB_PATTERNS:
        result, count = re.subn(pattern, lambda found: f"{found.group(0)} {word}", text, count=1)
        if count:
            return result
    return None


def edits() -> dict[str, Edit]:
    return {"negation": toggle_negation,
            "in_fact": lambda text: placebo_edit(text, "in fact"),
            "note": lambda text: f"Note: {text}",
            MEANING_EDIT: lambda text: placebo_edit(text, MEANING_EDIT)}


def edited_texts(pairs: Iterable[Pair]) -> list[str]:
    """Every text the test embeds, in a stable order."""
    texts: set[str] = set()
    for pair in pairs:
        for text in (pair.text_a, pair.text_b):
            texts.add(text)
            for edit in edits().values():
                changed = edit(text)
                if changed is None:
                    raise ValueError(f"No edit for selected statement {pair.id}")
                texts.add(changed)
    return sorted(texts)


def flip_score(pair: Pair, edit: Edit, vectors: dict[str, np.ndarray]) -> float:
    a, b = vectors[pair.text_a], vectors[pair.text_b]
    toward_a = cosine(a, vectors[edit(pair.text_b)])
    toward_b = cosine(b, vectors[edit(pair.text_a)])
    return (toward_a + toward_b) / 2 - cosine(a, b)


def _rank(pair_id: str, seed: str = SELECTION_SEED) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).hexdigest()


def select_fresh(lines: Iterable[str], seen: set[str], count: int = FRESH_PAIRS,
                 seed: str = SELECTION_SEED) -> list[Pair]:
    """Label-blind sample. One pair per premise, since train reuses each premise."""
    pairs, _ = mnli_pairs(lines, "mnli_train")
    # Train repeats some pair IDs, and selected pairs are stored by ID.
    uses = Counter(pair.id for pair in pairs)
    usable = [pair for pair in pairs if uses[pair.id] == 1 and eligible(pair)
              and pair.text_a not in seen and pair.text_b not in seen]
    chosen: dict[str, Pair] = {}
    for pair in sorted(usable, key=lambda item: _rank(item.id, seed)):
        chosen.setdefault(pair.text_a, pair)
    picked, statements = [], set()
    for pair in sorted(chosen.values(), key=lambda item: _rank(item.id, seed)):
        if pair.text_b not in statements and pair.text_a not in statements:
            statements.update((pair.text_a, pair.text_b))
            picked.append(pair)
    return picked[:count]


def wrong_partner_ap(pairs: Sequence[Pair], y: np.ndarray, vectors: dict[str, np.ndarray], name: str,
                     seed: int, shuffles: int = 5) -> float:
    """Mean AP when each B gets the A of another pair in its genre."""
    from sklearn.metrics import average_precision_score as ap

    rng, genres, found = np.random.default_rng(seed), np.asarray([p.relation_family for p in pairs]), []
    for _ in range(shuffles):
        order = np.arange(len(pairs))
        for genre in np.unique(genres):
            members = np.flatnonzero(genres == genre)
            order[members] = rng.permutation(members)
        wrong = [replace(pair, text_a=pairs[j].text_a) for pair, j in zip(pairs, order)]
        found.append(ap(y, np.asarray([flip_score(pair, edits()[name], vectors) for pair in wrong])))
    return float(np.mean(found))


def _percentile(scores: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(scores)) / max(len(scores) - 1, 1)


def swap_null(y: np.ndarray, first: np.ndarray, second: np.ndarray, seed: int,
              shuffles: int = 200) -> float:
    """97.5th percentile of the AP delta when each pair's two scores swap at random."""
    from sklearn.metrics import average_precision_score as ap

    rng, first, second, deltas = np.random.default_rng(seed), _percentile(first), _percentile(second), []
    for _ in range(shuffles):
        swap = rng.random(len(y)) < 0.5
        deltas.append(ap(y, np.where(swap, second, first)) - ap(y, np.where(swap, first, second)))
    return float(np.quantile(deltas, 0.975))


def verdict(delta: float, interval: dict | None, null_upper: float, hash_delta: float | None,
            positive: str, negative: str) -> str:
    """The frozen rule. A missing hash delta means this is the hash dry run."""
    if interval is None:
        return "UNCLEAR"
    if (delta >= MIN_USEFUL_DELTA_AP and interval["ci_lower"] > 0 and delta > null_upper
            and (hash_delta is None or delta > hash_delta)):
        return positive
    return negative if interval["ci_upper"] < MIN_USEFUL_DELTA_AP else "UNCLEAR"


def _measure(y: np.ndarray, top: np.ndarray, comparator: np.ndarray, ids: Sequence[str], seed: int,
             repetitions: int, hash_delta: float | None, positive: str, negative: str) -> dict:
    from sklearn.metrics import average_precision_score as ap

    delta = float(ap(y, top) - ap(y, comparator))
    found = grouped_ap_delta_interval(list(y), comparator, top, list(ids), repetitions=repetitions, seed=seed)
    interval = None if found is None else {"ci_lower": found["ci_lower"], "ci_upper": found["ci_upper"]}
    null_upper = swap_null(y, top, comparator, seed)
    return {"delta": delta, "interval": interval, "null_p97_5": null_upper, "hash_delta": hash_delta,
            "verdict": verdict(delta, interval, null_upper, hash_delta, positive, negative)}


def score_set(pairs: Sequence[Pair], vectors: dict[str, np.ndarray], seed: int = 42,
              repetitions: int = 2000, hash_deltas: dict[str, float] | None = None) -> dict:
    from sklearn.metrics import average_precision_score as ap

    y = np.asarray([pair.label == "contradiction" for pair in pairs], dtype=int)
    ids = [pair.id for pair in pairs]
    plain = np.asarray([cosine(vectors[p.text_a], vectors[p.text_b]) for p in pairs])
    flips = {name: np.asarray([flip_score(pair, edit, vectors) for pair in pairs])
             for name, edit in edits().items()}
    moved = {name: float(np.mean([1 - cosine(vectors[t], vectors[edit(t)])
                                  for p in pairs for t in (p.text_a, p.text_b)]))
             for name, edit in edits().items()}
    cos_name, cos_score = max((("cosine", plain), ("reversed_cosine", -plain)), key=lambda c: ap(y, c[1]))
    best = max(NEUTRAL_EDITS, key=lambda name: ap(y, flips[name]))
    clean = np.asarray([not NEGATION_WORD.search(p.text_a) and not NEGATION_WORD.search(p.text_b)
                        for p in pairs])
    genres = np.asarray([pair.relation_family for pair in pairs])
    hd = hash_deltas or {}
    return {
        "pairs": len(y), "positives": int(y.sum()), "floor": float(y.mean()),
        "ap": {**{f"flip_{name}": float(ap(y, score)) for name, score in flips.items()},
               "cosine": float(ap(y, plain)), "reversed_cosine": float(ap(y, -plain)),
               "net_flip_vs_best_placebo": float(ap(y, flips["negation"] - flips[best]))},
        "wrong_partner_ap": {name: wrong_partner_ap(pairs, y, vectors, name, seed) for name in ("negation", best)},
        "best_cosine": cos_name, "best_placebo": best, "mean_cosine_distance_moved": moved,
        "replication": _measure(y, flips["negation"], cos_score, ids, seed, repetitions,
                                hd.get("replication"), "REPLICATES", "FAILS"),
        "negation_vs_placebo": _measure(y, flips["negation"], flips[best], ids, seed, repetitions,
                                        hd.get("negation_vs_placebo"), "NEGATION-SPECIFIC", "EDIT-GENERAL"),
        "no_negation_words": {
            "pairs": int(clean.sum()), "positives": int(y[clean].sum()), "floor": float(y[clean].mean()),
            **{f"flip_{name}": float(ap(y[clean], score[clean])) for name, score in flips.items()}},
        "by_genre": {str(genre): {f"flip_{name}": float(ap(y[genres == genre], score[genres == genre]))
                                  for name, score in flips.items()}
                     for genre in np.unique(genres) if 0 < y[genres == genre].sum() < (genres == genre).sum()},
    }
