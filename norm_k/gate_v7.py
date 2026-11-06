"""Negated-cosine gate in front of NLI. See docs/preregistration-negated-gate-v7-draft.md."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .counterfactuals import toggle_negation
from .public_v2 import eligible
from .schema import Pair

K, PASS_SHARE, CONFIRM_AT = 5, 0.10, 0.5
MIN_MARGIN, MIN_ROOM = 0.05, 0.10
REPORT_KS, REPORT_SHARES = (5, 10, 20), (0.05, 0.10, 0.20)
SEED, LIMIT = "2031", 5000


def select_pairs(pairs: Sequence[Pair], seen: set[str], seed: str = SEED, limit: int = LIMIT) -> list[Pair]:
    """Label-blind: eligibility, unseen statements, one pair per statement pair, then a seeded sample."""
    unique: dict[tuple[str, str], Pair] = {}
    for pair in sorted(pairs, key=lambda item: item.id):
        if eligible(pair) and pair.text_a not in seen and pair.text_b not in seen:
            unique.setdefault((pair.text_a, pair.text_b), pair)
    kept = sorted(unique.values(), key=lambda item: item.id)
    return random.Random(seed).sample(kept, min(limit, len(kept)))


def _rows(texts: Sequence[str], vectors: Mapping[str, np.ndarray]) -> np.ndarray:
    found = np.asarray([vectors[text] for text in texts], dtype=float)
    return found / np.linalg.norm(found, axis=1, keepdims=True)


@dataclass
class Candidates:
    a: list[str]
    b: list[str]
    cosine: np.ndarray
    negated_cosine: np.ndarray

    @property
    def flip(self) -> np.ndarray:
        return self.negated_cosine - self.cosine

    def pair(self, index: int) -> tuple[str, str]:
        return self.a[index], self.b[index]


def retrieve(pairs: Sequence[Pair], vectors: Mapping[str, np.ndarray], k: int = K) -> Candidates:
    """Each distinct A's k highest-cosine Bs, over every distinct B in the set."""
    a_texts = sorted({p.text_a for p in pairs})
    b_texts = sorted({p.text_b for p in pairs})
    a, b = _rows(a_texts, vectors), _rows(b_texts, vectors)
    na = _rows([toggle_negation(t) for t in a_texts], vectors)
    nb = _rows([toggle_negation(t) for t in b_texts], vectors)
    plain = a @ b.T
    near = np.argsort(-plain, axis=1, kind="stable")[:, :k]
    i = np.repeat(np.arange(len(a_texts)), near.shape[1])
    j = near.ravel()
    return Candidates([a_texts[n] for n in i], [b_texts[n] for n in j], plain[i, j],
                      ((a[i] * nb[j]).sum(1) + (na[i] * b[j]).sum(1)) / 2)


def passed(score: np.ndarray, share: float = PASS_SHARE) -> np.ndarray:
    """Mask of the top share. Stable, so ties break the same way each run."""
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score, kind="stable")[:int(round(share * len(score)))]] = True
    return mask


def hits(candidates: Candidates, mask: np.ndarray, contradictions: Sequence[tuple[str, str]],
         nli: Mapping[tuple[str, str], float] | None) -> np.ndarray:
    """One entry per labeled contradiction: 1 if it passed, and NLI confirmed it when NLI is given."""
    through = {candidates.pair(i) for i in np.flatnonzero(mask)}
    return np.asarray([key in through and (nli is None or nli[key] >= CONFIRM_AT) for key in contradictions],
                      dtype=float)


def margin(first: np.ndarray, second: np.ndarray, seed: int = 42, repetitions: int = 2000) -> dict:
    """First minus second, resampling contradictions with cutoffs fixed, as in v5."""
    gaps = first - second
    draws = np.random.default_rng(seed).integers(0, len(gaps), size=(repetitions, len(gaps)))
    means = gaps[draws].mean(axis=1)
    return {"margin": float(gaps.mean()), "ci_lower": float(np.quantile(means, 0.025)),
            "ci_upper": float(np.quantile(means, 0.975))}


def gate_verdict(found: dict, room: float | None, hash_margin: float | None) -> str:
    """The rule. room is None only on the hash dry run, which has no ceiling."""
    if room is not None and room < MIN_ROOM:
        return "UNCLEAR"
    if (found["margin"] >= MIN_MARGIN and found["ci_lower"] > 0
            and (hash_margin is None or found["margin"] > hash_margin)):
        return "GATE-PASS"
    return "GATE-FAIL" if found["ci_upper"] < MIN_MARGIN else "UNCLEAR"


def rankings(candidates: Candidates, seed: int = 42) -> dict[str, np.ndarray]:
    size = len(candidates.a)
    return {"cosine": candidates.cosine, "negated_cosine": candidates.negated_cosine, "flip": candidates.flip,
            "random": np.random.default_rng(seed).random(size)}


def needed_pairs(candidates: Candidates, full: bool) -> list[tuple[str, str]]:
    """Pairs the NLI model must read: all candidates, or only what the two gated rankings pass."""
    if full:
        return sorted(set(zip(candidates.a, candidates.b)))
    keep = passed(candidates.cosine) | passed(candidates.negated_cosine)
    return sorted({candidates.pair(i) for i in np.flatnonzero(keep)})


def evaluate(pairs: Sequence[Pair], vectors: Mapping[str, np.ndarray], nli: Mapping[tuple[str, str], float],
             hash_margin: float | None = None, full: bool = True, seed: int = 42, repetitions: int = 2000) -> dict:
    contradictions = [(p.text_a, p.text_b) for p in pairs if p.label == "contradiction"]
    labeled = {(p.text_a, p.text_b) for p in pairs}
    candidates = retrieve(pairs, vectors)
    ranks = rankings(candidates, seed)
    masks = {name: passed(score) for name, score in ranks.items()}
    gated = {name: hits(candidates, masks[name], contradictions, nli) for name in ("cosine", "negated_cosine")}
    found = margin(gated["negated_cosine"], gated["cosine"], seed, repetitions)
    result = {
        "pairs": len(pairs), "contradictions": len(contradictions), "candidates": len(candidates.a),
        "passed_to_nli": int(masks["cosine"].sum()),
        "found": {name: float(value.mean()) for name, value in gated.items()}, "margin": found,
    }
    room = None
    if full:
        everything = np.ones(len(candidates.a), dtype=bool)
        ceiling = hits(candidates, everything, contradictions, nli).mean()
        room = float(ceiling - gated["cosine"].mean())
        result["ceiling"], result["room"] = float(ceiling), room
        result["found"]["flip"] = float(hits(candidates, masks["flip"], contradictions, nli).mean())
        result["found"]["random"] = float(hits(candidates, masks["random"], contradictions, nli).mean())
        result["reachable"] = {
            str(k): float(hits(retrieve(pairs, vectors, k), np.ones(len(candidates.a) // K * k, dtype=bool),
                               contradictions, None).mean()) for k in REPORT_KS}
        result["recall_before_nli"] = {
            str(share): {name: float(hits(candidates, passed(score, share), contradictions, None).mean())
                         for name, score in ranks.items()} for share in REPORT_SHARES}
        calls = set(zip(candidates.a, candidates.b))
        result["unlabeled_nli_contradictions"] = sum(key not in labeled and nli[key] >= CONFIRM_AT for key in calls)
        result["distinct_candidates"] = len(calls)
    result["verdict"] = gate_verdict(found, room, hash_margin)
    return result
