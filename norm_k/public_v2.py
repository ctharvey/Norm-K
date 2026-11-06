"""Fair K test on public human-labeled pairs. See docs/preregistration-public-v2.md."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .ablation import (BASE_FIELDS, CORNER_FIELDS, K_FIELDS, LEXICAL_FIELDS, MIN_USEFUL_DELTA_AP,
                       corner_features, counterfactual_features)
from .counterfactuals import CounterfactualSquare, counterfactual_square, toggle_negation
from .evaluation import token_features
from .schema import Pair
from .statistics import grouped_ap_delta_interval

MNLI_SHA256 = "049f507b9e36b1fcb756cfd5aeb3b7a0cfcb84bf023793652987f7e7e0957822"
WANLI_SHA256 = "4276e0af7fcdf657d1ab7beb54eaf025fda592a76c9ee86b63b7871953fc74fd"
MAX_WORDS = 25
FULL_FIELDS = BASE_FIELDS + CORNER_FIELDS + LEXICAL_FIELDS
NLI_FIELD = "nli"
# name: (model kind, baseline fields, fields added to the baseline)
COMPARISONS = {
    "k_over_full": ("lr", FULL_FIELDS, K_FIELDS),
    "k_over_rf": ("rf", FULL_FIELDS, K_FIELDS),
    "k_over_nli_full": ("lr", FULL_FIELDS + (NLI_FIELD,), K_FIELDS),
    "corners_over_rf_base_lexical": ("rf", BASE_FIELDS + LEXICAL_FIELDS, CORNER_FIELDS),
}
PRIMARY_COMPARISONS = ("k_over_full", "k_over_rf")

_VERB_WORDS = re.compile(r"\b(own|owns|live|lives|work|works|reside|resides)\b", re.I)
_BEFORE_EDIT = re.compile(r",|;|:|\b(which|that|who|whom|whose|what|when|where|while|if|because"
                          r"|although|though|since|until|unless|after|before|as|so|but|and|or"
                          r"|knew|said|say|says|think|thinks|believe|believes|yes|well|uh|um)\b", re.I)

# Optional hook for long runs: called as progress_hook(comparison, done, total).
progress_hook: Callable[[str, int, int], None] | None = None


def short_single(text: str) -> bool:
    return len(text.split()) <= MAX_WORDS and len(re.findall(r"[.!?](\s|$)", text.strip())) <= 1


def main_clause(text: str) -> bool:
    """The negation edit must land on a main-clause is or was. Uses no labels."""
    if _VERB_WORDS.search(text):  # The rule edits these first, and they are often nouns.
        return False
    toggled = toggle_negation(text)
    if not toggled:
        return False
    prefix = os.path.commonprefix([text, toggled])
    return not _BEFORE_EDIT.search(prefix) and not re.match(r"\s*there\b", prefix, re.I)


def eligible(pair: Pair) -> bool:
    square = counterfactual_square(pair, "negation")
    if square is None or (square.transformed_base == pair.text_b
                          and square.transformed_other == pair.text_a):
        return False
    return all(short_single(text) and main_clause(text) for text in (pair.text_a, pair.text_b))


def _pair(pair_id: str, a: str, b: str, gold: str, genre: str, source: str) -> Pair:
    contradiction = gold == "contradiction"
    return Pair(id=pair_id, text_a=a, text_b=b, label="contradiction" if contradiction else "compatible",
                subject="unknown", attribute=f"nli_{gold}", relation_family=genre,
                template_family=source, transformation="public_nli", cardinality="unknown",
                same_fact=contradiction, incompatible=contradiction, same_valid_time=True,
                rationale=f"Public {source} gold label: {gold}.", generated=False)


def mnli_pairs(lines: Iterable[str], source: str) -> tuple[list[Pair], dict[str, int]]:
    """Parse MultiNLI JSONL. Returns pairs and how many of 5 annotators chose gold."""
    pairs, agreement = [], {}
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        gold = row["gold_label"]
        if gold == "-":
            continue
        pair_id = f"mnli-{row['pairID']}"
        pairs.append(_pair(pair_id, row["sentence1"], row["sentence2"], gold, row["genre"], source))
        agreement[pair_id] = sum(label == gold for label in row["annotator_labels"])
    return pairs, agreement


def wanli_pairs(lines: Iterable[str]) -> list[Pair]:
    pairs = []
    for line in lines:
        if line.strip():
            row = json.loads(line)
            pairs.append(_pair(f"wanli-{row['id']}", row["premise"], row["hypothesis"], row["gold"],
                               "wanli", "wanli"))
    return pairs


def _verify(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{path.name} does not match its pinned SHA-256")


@dataclass(frozen=True)
class PublicData:
    primary: list[Pair]
    replication: list[Pair]
    agreement: dict[str, int]


def select_public(mnli: Sequence[Pair], agreement: dict[str, int],
                  wanli: Sequence[Pair]) -> PublicData:
    primary = [pair for pair in mnli if eligible(pair)]
    seen = {text for pair in primary for text in (pair.text_a, pair.text_b)}
    # Replication trains on the primary set, so it can't share a statement with it.
    replication = [pair for pair in wanli if eligible(pair)
                   and pair.text_a not in seen and pair.text_b not in seen]
    return PublicData(primary, replication, {pair.id: agreement[pair.id] for pair in primary})


def load_public(mnli_zip: str | Path, wanli_path: str | Path) -> PublicData:
    mnli_zip, wanli_path = Path(mnli_zip), Path(wanli_path)
    _verify(mnli_zip, MNLI_SHA256)
    _verify(wanli_path, WANLI_SHA256)
    pairs, agreement = [], {}
    with zipfile.ZipFile(mnli_zip) as archive:
        for split in ("matched", "mismatched"):
            name = f"multinli_1.0/multinli_1.0_dev_{split}.jsonl"
            lines = archive.read(name).decode("utf-8").splitlines()
            found, votes = mnli_pairs(lines, f"mnli_{split}")
            pairs += found
            agreement.update(votes)
    wanli = wanli_pairs(wanli_path.read_text(encoding="utf-8").splitlines())
    return select_public(pairs, agreement, wanli)


def _model(kind: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=seed))


def _fit_score(kind: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
               seed: int) -> np.ndarray:
    model = _model(kind, seed)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def _held_out_scores(kind: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     seed: int) -> np.ndarray:
    scores = np.zeros(len(y), dtype=float)
    for group in np.unique(groups):
        held_out = groups == group
        if len(set(y[~held_out])) < 2:
            raise ValueError(f"Training folds for {group} contain one class")
        scores[held_out] = _fit_score(kind, x[~held_out], y[~held_out], x[held_out], seed)
    return scores


def clears_bar(delta: float, interval: dict | None, null_upper: float,
               hash_delta: float | None) -> bool:
    """All frozen conditions. A missing hash delta means this is the hash dry run."""
    if interval is None or delta < MIN_USEFUL_DELTA_AP or interval["ci_lower"] <= 0:
        return False
    return delta > null_upper and (hash_delta is None or delta > hash_delta)


def public_decision(comparisons: dict[str, dict]) -> str:
    primary = [comparisons[name] for name in PRIMARY_COMPARISONS]
    if all(item["clears_bar"] for item in primary):
        return "PASS"
    if any(item["interval"] and item["interval"]["ci_upper"] < MIN_USEFUL_DELTA_AP for item in primary):
        return "STOP"
    return "INCONCLUSIVE"


def _rows(pairs: Sequence[Pair], squares: Sequence[CounterfactualSquare],
          vectors: dict[str, np.ndarray], nli: dict[str, float]) -> list[dict[str, float]]:
    return [{**counterfactual_features(square, vectors), **corner_features(square, vectors),
             **token_features(pair.text_a, pair.text_b), NLI_FIELD: nli[pair.id]}
            for pair, square in zip(pairs, squares)]


def _columns(rows: Sequence[dict[str, float]], fields: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[row[field] for field in fields] for row in rows])


def _labels(pairs: Sequence[Pair]) -> np.ndarray:
    return np.asarray([pair.label == "contradiction" for pair in pairs], dtype=int)


def _hypothesis_only(pairs: Sequence[Pair], y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = np.asarray([pair.text_b for pair in pairs], dtype=object)
    scores = np.zeros(len(y), dtype=float)
    for group in np.unique(groups):
        held_out = groups == group
        vectorizer = CountVectorizer(binary=True, min_df=2)
        model = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
        model.fit(vectorizer.fit_transform(texts[~held_out]), y[~held_out])
        scores[held_out] = model.predict_proba(vectorizer.transform(texts[held_out]))[:, 1]
    return scores


def _compare(name: str, rows: Sequence[dict[str, float]], y: np.ndarray, groups: np.ndarray,
             seed: int, repetitions: int, permutations: int, hash_delta: float | None) -> tuple[dict, np.ndarray, np.ndarray]:
    from sklearn.metrics import average_precision_score as ap

    kind, baseline_fields, added_fields = COMPARISONS[name]
    baseline, added = _columns(rows, baseline_fields), _columns(rows, added_fields)
    base_scores = _held_out_scores(kind, baseline, y, groups, seed)
    new_scores = _held_out_scores(kind, np.column_stack([baseline, added]), y, groups, seed)
    base_ap, new_ap = float(ap(y, base_scores)), float(ap(y, new_scores))
    delta = new_ap - base_ap
    found = grouped_ap_delta_interval(list(y), base_scores, new_scores, list(groups),
                                      repetitions=repetitions, seed=seed)
    interval = None if found is None else {
        "ci_lower": found["ci_lower"], "ci_upper": found["ci_upper"],
        "resamples_accepted": found["resamples_accepted"], "clusters": found["subject_clusters"]}
    rng, null = np.random.default_rng(seed), []
    for step in range(permutations):
        shuffled = added.copy()
        for group in np.unique(groups):
            members = np.flatnonzero(groups == group)
            shuffled[members] = added[rng.permutation(members)]
        score = _held_out_scores(kind, np.column_stack([baseline, shuffled]), y, groups, seed)
        null.append(float(ap(y, score)) - base_ap)
        if progress_hook:
            progress_hook(name, step + 1, permutations)
    null_upper = float(np.quantile(null, 0.975)) if null else float("inf")
    return ({"model": kind, "baseline_ap": base_ap, "candidate_ap": new_ap, "delta": delta,
             "interval": interval, "null_mean": float(np.mean(null)) if null else None,
             "null_p97_5": null_upper, "hash_delta": hash_delta,
             "clears_bar": clears_bar(delta, interval, null_upper, hash_delta)},
            base_scores, new_scores)


def _subset_delta(y: np.ndarray, base: np.ndarray, new: np.ndarray, keep: np.ndarray) -> float | None:
    from sklearn.metrics import average_precision_score as ap

    if len(set(y[keep])) < 2:
        return None
    return float(ap(y[keep], new[keep]) - ap(y[keep], base[keep]))


def _replication(name: str, train_rows: Sequence[dict[str, float]], y_train: np.ndarray,
                 test_rows: Sequence[dict[str, float]], y_test: np.ndarray,
                 ids: Sequence[str], seed: int, repetitions: int) -> dict:
    from sklearn.metrics import average_precision_score as ap

    kind, baseline_fields, added_fields = COMPARISONS[name]
    candidate_fields = baseline_fields + added_fields
    base = _fit_score(kind, _columns(train_rows, baseline_fields), y_train,
                      _columns(test_rows, baseline_fields), seed)
    new = _fit_score(kind, _columns(train_rows, candidate_fields), y_train,
                     _columns(test_rows, candidate_fields), seed)
    found = grouped_ap_delta_interval(list(y_test), base, new, list(ids), repetitions=repetitions, seed=seed)
    return {"baseline_ap": float(ap(y_test, base)), "candidate_ap": float(ap(y_test, new)),
            "delta": float(ap(y_test, new) - ap(y_test, base)),
            "interval": None if found is None else {"ci_lower": found["ci_lower"], "ci_upper": found["ci_upper"]}}


def run_public_test(data: PublicData, vectors: dict[str, np.ndarray], nli: dict[str, float],
                    hash_deltas: dict[str, float] | None, seed: int = 42, repetitions: int = 2000,
                    permutations: int = 200) -> dict:
    from sklearn.metrics import average_precision_score as ap

    squares = [counterfactual_square(pair, "negation") for pair in data.primary]
    if any(square is None for square in squares):
        raise ValueError("Every selected pair must have a negation square")
    rows = _rows(data.primary, squares, vectors, nli)
    y = _labels(data.primary)
    groups = np.asarray([pair.relation_family for pair in data.primary])
    sources = np.asarray([pair.template_family for pair in data.primary])
    votes = np.asarray([data.agreement[pair.id] for pair in data.primary])

    comparisons, robustness = {}, {}
    for name in COMPARISONS:
        result, base, new = _compare(name, rows, y, groups, seed, repetitions, permutations,
                                     None if hash_deltas is None else hash_deltas[name])
        comparisons[name] = result
        robustness[name] = {
            "agreement_4_of_5": _subset_delta(y, base, new, votes >= 4),
            "mismatched_genres": _subset_delta(y, base, new, sources == "mnli_mismatched"),
            "without_genre": {str(group): _subset_delta(y, base, new, groups != group)
                              for group in np.unique(groups)},
        }

    replication_squares = [counterfactual_square(pair, "negation") for pair in data.replication]
    replication_rows = _rows(data.replication, replication_squares, vectors, nli)
    y_test = _labels(data.replication)
    replication = {name: _replication(name, rows, y, replication_rows, y_test,
                                      [pair.id for pair in data.replication], seed, repetitions)
                   for name in COMPARISONS}

    reference = {
        "lexical_lr": _held_out_scores("lr", _columns(rows, LEXICAL_FIELDS), y, groups, seed),
        "base_lr": _held_out_scores("lr", _columns(rows, BASE_FIELDS), y, groups, seed),
        "nli_alone": _columns(rows, (NLI_FIELD,))[:, 0],
        "hypothesis_only": _hypothesis_only(data.primary, y, groups, seed),
    }
    return {
        "pairs": len(y), "positives": int(y.sum()), "genres": len(set(groups)),
        "prevalence_ap_floor": float(y.mean()),
        "genre_counts": {str(g): int((groups == g).sum()) for g in np.unique(groups)},
        "replication_pairs": len(y_test), "replication_positives": int(y_test.sum()),
        "decision": None if hash_deltas is None else public_decision(comparisons),
        "hash_dry_run_decision": public_decision(comparisons) if hash_deltas is None else None,
        "comparisons": comparisons, "robustness": robustness, "replication": replication,
        "reference_ap": {name: float(ap(y, score)) for name, score in reference.items()},
        "replication_nli_alone_ap": float(ap(y_test, _columns(replication_rows, (NLI_FIELD,))[:, 0])),
    }
