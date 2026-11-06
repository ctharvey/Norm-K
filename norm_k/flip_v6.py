"""Flip score on NaN-NLI with a whole-clause wrapper edit. See docs/preregistration-nan-nli-v6-draft.md."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .counterfactuals import toggle_negation
from .flip_v4 import _measure, swap_null, verdict
from .flip_v5 import _flip, distance, gate_passes
from .geometry import cosine
from .public_v2 import _pair, eligible
from .schema import Pair
from .statistics import grouped_ap_delta_interval

NAN_SHA256 = "533cd0a169662834a5bcbb6c064472e52b549424f43164cd7eabc087bfd6e89c"
OWN_ROW_CONSTRUCTIONS = ("not + quantifier", "not in coordination", "not + focus particle",
                         "not + degree expression")  # At least 20 pairs each.
NLI_COLUMNS = ("c_ab", "e_ab", "n_ab", "c_ba", "e_ba", "n_ba")


def negated(text: str) -> str:
    return f"It is not the case that {text}"


def control(text: str) -> str:
    return f"It is the case that {text}"


def wrapper_texts(statements: Sequence[str]) -> list[str]:
    return sorted({*statements, *map(negated, statements), *map(control, statements)})


def statements_of(pairs: Sequence[Pair]) -> list[str]:
    return sorted({t for p in pairs for t in (p.text_a, p.text_b)})


def read_nan(path: str | Path) -> list[Pair]:
    """Every row of the pinned file. Row numbers are the IDs, so no text needs committing."""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != NAN_SHA256:
        raise ValueError("NaN-NLI file does not match its pinned SHA-256")
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return [_pair(f"nan-{index}", row["premise"], row["hypothesis"], row["label"], row["Construction"], "nan_nli")
                for index, row in enumerate(csv.DictReader(stream))]


def size_quality(statements: Sequence[str], vectors: dict[str, np.ndarray]) -> dict:
    """Same gate fields as v5, with the control in place of the matched edit."""
    away = np.asarray([distance(t, negated(t), vectors) for t in statements])
    held = np.asarray([distance(t, control(t), vectors) for t in statements])
    gaps = held - away
    return {"statements": len(statements), "negation_mean": float(away.mean()), "control_mean": float(held.mean()),
            "mean_gap": float(gaps.mean()), "median_abs_gap": float(np.median(np.abs(gaps)))}


def flips(pairs: Sequence[Pair], edit, vectors: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([_flip(p, edit(p.text_a), edit(p.text_b), vectors) for p in pairs])


def minimum_gain(pairs: Sequence[Pair], edit, vectors: dict[str, np.ndarray]) -> np.ndarray:
    found = []
    for p in pairs:
        a, b, plain = vectors[p.text_a], vectors[p.text_b], cosine(vectors[p.text_a], vectors[p.text_b])
        found.append(min(cosine(vectors[edit(p.text_a)], b), cosine(a, vectors[edit(p.text_b)])) - plain)
    return np.asarray(found)


def labels(pairs: Sequence[Pair]) -> np.ndarray:
    return np.asarray([p.label == "contradiction" for p in pairs], dtype=int)


def bridge_row(pairs: Sequence[Pair], vectors: dict[str, np.ndarray], seed: int = 42, repetitions: int = 2000) -> dict:
    from sklearn.metrics import average_precision_score as ap

    y, statements = labels(pairs), statements_of(pairs)
    negation, held = flips(pairs, negated, vectors), flips(pairs, control, vectors)
    quality = size_quality(statements, vectors)
    measure = _measure(y, negation, held, [p.id for p in pairs], seed, repetitions, None, "CARRIES", "NO-SIGNAL")
    return {"pairs": len(y), "positives": int(y.sum()), "floor": float(y.mean()), "negation_ap": float(ap(y, negation)),
            "control_ap": float(ap(y, held)), "measure": measure, "quality": quality,
            "size_gate": gate_passes(quality)}


def bridge_passes(row: dict) -> bool:
    measure = row["measure"]
    return bool(row["size_gate"] and measure["delta"] >= 0.03 and measure["interval"]
                and measure["interval"]["ci_lower"] > 0)


def shuffle_null(y: np.ndarray, first: np.ndarray, second: np.ndarray, groups: Sequence[str], seed: int,
                 shuffles: int = 2000) -> float:
    """97.5th percentile of the AP delta with labels shuffled inside each group."""
    from sklearn.metrics import average_precision_score as ap

    rng, names = np.random.default_rng(seed), np.asarray(groups)
    members = [np.flatnonzero(names == name) for name in np.unique(names)]
    found = []
    for _ in range(shuffles):
        shuffled = y.copy()
        for index in members:
            shuffled[index] = rng.permutation(y[index])
        found.append(ap(shuffled, first) - ap(shuffled, second))
    return float(np.quantile(found, 0.975))


def wrong_partner(pairs: Sequence[Pair], y: np.ndarray, vectors: dict[str, np.ndarray], seed: int,
                  shuffles: int = 5) -> dict:
    """Each B gets the A of a pair in its construction with another premise, where one exists."""
    from sklearn.metrics import average_precision_score as ap

    rng = np.random.default_rng(seed)
    usable = [i for i, p in enumerate(pairs)
              if any(q.relation_family == p.relation_family and q.text_a != p.text_a for q in pairs)]
    found: dict[str, list[float]] = {"negation": [], "control": []}
    for _ in range(shuffles):
        wrong = []
        for i in usable:
            options = [q for q in pairs if q.relation_family == pairs[i].relation_family and q.text_a != pairs[i].text_a]
            wrong.append(replace(pairs[i], text_a=options[rng.integers(len(options))].text_a))
        for name, edit in (("negation", negated), ("control", control)):
            found[name].append(ap(y[usable], flips(wrong, edit, vectors)))
    means = {name: float(np.mean(values)) for name, values in found.items()}
    return {**means, "delta": means["negation"] - means["control"], "pairs": len(usable),
            "positives": int(y[usable].sum())}


def nli_scores(y: np.ndarray, table: dict[str, dict[str, float]], ids: Sequence[str]) -> dict:
    from sklearn.metrics import average_precision_score as ap

    logits = np.asarray([[table[i][c] for c in NLI_COLUMNS] for i in ids], dtype=float)

    def contradiction(block: np.ndarray) -> np.ndarray:
        shifted = np.exp(block - block.max(1, keepdims=True))
        return shifted[:, 0] / shifted.sum(1)

    ab, ba = contradiction(logits[:, :3]), contradiction(logits[:, 3:])
    scores = {"pc_ab": ab, "pc_ba": ba, "mean": (ab + ba) / 2, "minimum": np.minimum(ab, ba),
              "maximum": np.maximum(ab, ba), "c_minus_e": logits[:, 0] - logits[:, 1]}
    return {name: float(ap(y, score)) for name, score in scores.items()}


def nan_texts(pairs: Sequence[Pair]) -> list[str]:
    """The wrapper set plus v5's edit on the pairs it covers."""
    covered = [t for p in pairs if eligible(p) for t in (p.text_a, p.text_b)]
    return sorted({*wrapper_texts(statements_of(pairs)), *covered, *map(toggle_negation, covered)})


def score_nan(pairs: Sequence[Pair], vectors: dict[str, np.ndarray], seed: int = 42, repetitions: int = 2000,
              hash_delta: float | None = None, nli: dict[str, dict[str, float]] | None = None) -> dict:
    from sklearn.metrics import average_precision_score as ap

    y, ids = labels(pairs), [p.id for p in pairs]
    premises = [p.text_a for p in pairs]
    negation, held = flips(pairs, negated, vectors), flips(pairs, control, vectors)
    quality = size_quality(statements_of(pairs), vectors)
    delta = float(ap(y, negation) - ap(y, held))
    found = grouped_ap_delta_interval(list(y), held, negation, premises, repetitions=repetitions, seed=seed)
    interval = None if found is None else {"ci_lower": found["ci_lower"], "ci_upper": found["ci_upper"]}
    null_upper = shuffle_null(y, negation, held, premises, seed, repetitions)
    result = verdict(delta, interval, null_upper, hash_delta, "TRANSFERS", "NO-TRANSFER")
    if not gate_passes(quality):
        result = "UNCLEAR"
    plain = np.asarray([cosine(vectors[p.text_a], vectors[p.text_b]) for p in pairs])
    gold = np.asarray([p.attribute.removeprefix("nli_") for p in pairs])
    family = np.asarray([p.relation_family if p.relation_family in OWN_ROW_CONSTRUCTIONS else "other"
                         for p in pairs])

    def subset(mask: np.ndarray) -> dict:
        row = {"pairs": int(mask.sum()), "positives": int(y[mask].sum())}
        if 0 < y[mask].sum() < mask.sum():
            row.update(floor=float(y[mask].mean()), negation_ap=float(ap(y[mask], negation[mask])),
                       control_ap=float(ap(y[mask], held[mask])))
        return row

    covered = [p for p in pairs if eligible(p)]
    y_covered = labels(covered)
    return {
        "pairs": len(y), "positives": int(y.sum()), "premises": len(set(premises)), "floor": float(y.mean()),
        "primary": {"negation_ap": float(ap(y, negation)), "control_ap": float(ap(y, held)), "delta": delta,
                    "interval": interval, "null_p97_5": null_upper, "hash_delta": hash_delta, "verdict": result},
        "quality": quality, "size_gate": gate_passes(quality),
        "swap_null_p97_5": swap_null(y, negation, held, seed),
        "cosine_ap": float(ap(y, plain)), "reversed_cosine_ap": float(ap(y, -plain)),
        "against": {name: subset(np.isin(gold, ["contradiction", name])) for name in ("entailment", "neutral")},
        "by_construction": {name: subset(family == name) for name in (*OWN_ROW_CONSTRUCTIONS, "other")},
        "minimum_gain_ap": {"negation": float(ap(y, minimum_gain(pairs, negated, vectors))),
                            "control": float(ap(y, minimum_gain(pairs, control, vectors)))},
        "wrong_partner": wrong_partner(pairs, y, vectors, seed),
        "v5_edit": {"pairs": len(covered), "positives": int(y_covered.sum()),
                    "premises": len({p.text_a for p in covered}),
                    "ap": float(ap(y_covered, flips(covered, toggle_negation, vectors)))
                    if 0 < y_covered.sum() < len(covered) else None},
        "nli_ap": None if nli is None else nli_scores(y, nli, ids),
    }
