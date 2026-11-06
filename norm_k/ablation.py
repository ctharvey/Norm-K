"""Paired, same-cohort ablations of K-derived features.

Only available on examples with a supported counterfactual transformation.
Both models use the exact same rows/splits to avoid coverage confounding.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from .counterfactuals import CounterfactualSquare, is_mirror
from .audit import purged_indices
from .evaluation import review_budget_metrics, token_features
from .geometry import cosine, k_norm, k_relative, pair_features
from .schema import Pair
from .statistics import grouped_ap_delta_interval


BASE_FIELDS = ("cosine", "euclidean", "hoyer_delta", "mean_abs_delta", "mean_product")
# With cosine(A, B), these hold every pairwise score ||K|| is built from on unit
# vectors. Without them, a K gain may only reflect the two extra embedded texts.
CORNER_FIELDS = ("cos_ta_tb", "cos_a_tb", "cos_b_ta", "norm_a_ta", "norm_b_tb")
K_FIELDS = ("k_norm", "k_relative")
LEXICAL_FIELDS = ("token_jaccard", "negation_disagreement", "number_disagreement",
                  "token_symmetric_difference")
BENCHMARK_CELLS = {"HC": "high_overlap_contradiction", "HN": "high_overlap_other",
                   "LC": "low_overlap_contradiction", "LN": "low_overlap_other"}
# Frozen in docs/preregistration-matched-v1.md before any real-embedding run.
MIN_USEFUL_DELTA_AP = 0.03


def load_matched_benchmark(path: str | Path) -> list[Pair]:
    """Load the hand-written 2x2 TSV. Labels are author-assigned, not human-audited."""
    pairs = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for number, row in enumerate(csv.DictReader(stream, delimiter="\t"), 1):
            label = row["label"]
            pairs.append(Pair(
                id=f"matched-v1-{number:04d}", text_a=row["text_a"], text_b=row["text_b"],
                label=label, subject=row["subject"], attribute=row["family"],
                relation_family=row["family"], template_family=BENCHMARK_CELLS[row["cell"]],
                transformation="hand_written", cardinality="unknown",
                same_fact=label in {"contradiction", "equivalent", "temporal_update"},
                incompatible=label in {"contradiction", "temporal_update"},
                same_valid_time=label != "temporal_update",
                rationale="Author-assigned provisional label; needs independent review.",
                generated=False))
    return pairs


def corner_features(square: CounterfactualSquare,
                    vectors: dict[str, np.ndarray]) -> dict[str, float]:
    a, b = vectors[square.base], vectors[square.other]
    ta, tb = vectors[square.transformed_base], vectors[square.transformed_other]
    return {"cos_ta_tb": cosine(ta, tb), "cos_a_tb": cosine(a, tb), "cos_b_ta": cosine(b, ta),
            "norm_a_ta": float(np.linalg.norm(ta - a)), "norm_b_tb": float(np.linalg.norm(tb - b))}


def _out_of_family_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          seed: int) -> np.ndarray:
    """Score each relation family with a model that never saw that family."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scores = np.zeros(len(y), dtype=float)
    for family in np.unique(groups):
        held_out = groups == family
        if len(set(y[~held_out])) < 2:
            raise ValueError(f"Training folds for {family} contain one class")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000,
                                                                   random_state=seed))
        model.fit(x[~held_out], y[~held_out])
        scores[held_out] = model.predict_proba(x[held_out])[:, 1]
    return scores


def preregistered_decision(delta: float, interval: dict | None, null_upper: float) -> str:
    """Apply the frozen K rule. A wide interval is INCONCLUSIVE, never a pass.

    The protocol adds one check outside this function: a real encoder's delta must
    also beat the hash control's, which passed an earlier, weaker version of this rule.
    """
    if interval is None:
        return "INCONCLUSIVE"
    if delta >= MIN_USEFUL_DELTA_AP and interval["ci_lower"] > 0 and delta > null_upper:
        return "PASS"
    if interval["ci_upper"] < MIN_USEFUL_DELTA_AP:
        return "STOP"
    return "INCONCLUSIVE"


def run_fair_k_test(pairs: Sequence[Pair], squares: Sequence[CounterfactualSquare],
                    vectors: dict[str, np.ndarray], seed: int = 42,
                    repetitions: int = 2000, permutations: int = 200) -> dict:
    """Does K add anything once the baseline holds the scores K is built from?"""
    from sklearn.metrics import average_precision_score

    if len(pairs) != len(squares) or not pairs:
        raise ValueError("Pairs and squares must align and be nonempty")
    if any(pair.id != square.pair_id for pair, square in zip(pairs, squares)):
        raise ValueError("Pair/square order mismatch")
    kept = [i for i, square in enumerate(squares) if not is_mirror(square)]
    mirrors = len(pairs) - len(kept)
    pairs, squares = [pairs[i] for i in kept], [squares[i] for i in kept]
    rows = [{**counterfactual_features(square, vectors), **corner_features(square, vectors),
             **token_features(pair.text_a, pair.text_b)} for pair, square in zip(pairs, squares)]
    y = np.asarray([pair.label == "contradiction" for pair in pairs], dtype=int)
    groups = np.asarray([pair.relation_family for pair in pairs])

    def columns(fields: tuple[str, ...]) -> np.ndarray:
        return np.asarray([[row[field] for field in fields] for row in rows])

    models = {"lexical": LEXICAL_FIELDS, "base": BASE_FIELDS, "base_k": BASE_FIELDS + K_FIELDS,
              "base_corners": BASE_FIELDS + CORNER_FIELDS,
              "base_corners_k": BASE_FIELDS + CORNER_FIELDS + K_FIELDS,
              "full": BASE_FIELDS + CORNER_FIELDS + LEXICAL_FIELDS,
              "full_k": BASE_FIELDS + CORNER_FIELDS + LEXICAL_FIELDS + K_FIELDS}
    scores = {name: _out_of_family_scores(columns(fields), y, groups, seed)
              for name, fields in models.items()}
    ap = {name: float(average_precision_score(y, score)) for name, score in scores.items()}

    def interval(first: str, second: str) -> dict | None:
        found = grouped_ap_delta_interval(list(y), scores[first], scores[second], list(groups),
                                          repetitions=repetitions, seed=seed)
        if found is None:
            return None
        return {"ci_lower": found["ci_lower"], "ci_upper": found["ci_upper"],
                "resamples_accepted": found["resamples_accepted"],
                "clusters": found["subject_clusters"], "cluster_unit": "relation_family"}

    primary = ap["full_k"] - ap["full"]
    primary_interval = interval("full", "full_k")

    # Null: keep K's values and each family's mix, break the link to each pair.
    rng = np.random.default_rng(seed)
    fixed, k_values = columns(models["full"]), columns(K_FIELDS)
    null = []
    for _ in range(permutations):
        shuffled = k_values.copy()
        for family in np.unique(groups):
            members = np.flatnonzero(groups == family)
            shuffled[members] = k_values[rng.permutation(members)]
        score = _out_of_family_scores(np.column_stack([fixed, shuffled]), y, groups, seed)
        null.append(float(average_precision_score(y, score)) - ap["full"])
    null_upper = float(np.quantile(null, 0.975))

    cells: dict[str, list[float]] = {}
    for pair, row in zip(pairs, rows):
        cells.setdefault(pair.template_family, []).append(row["k_norm"])
    return {
        "pairs_scored": len(pairs), "mirror_pairs_excluded": mirrors,
        "positives": int(y.sum()), "families": len(set(groups)),
        "prevalence_ap_floor": float(y.mean()), "average_precision": ap,
        "primary_delta_ap_k_over_full": primary,
        "primary_interval": primary_interval,
        "decision": preregistered_decision(primary, primary_interval, null_upper),
        "secondary_delta_ap_corners_over_base": ap["base_corners"] - ap["base"],
        "secondary_interval_corners_over_base": interval("base", "base_corners"),
        "secondary_delta_ap_k_over_corners": ap["base_corners_k"] - ap["base_corners"],
        "secondary_delta_ap_k_over_base": ap["base_k"] - ap["base"],
        "shuffled_k_null": {"permutations": permutations, "mean": float(np.mean(null)),
                            "p2_5": float(np.quantile(null, 0.025)), "p97_5": null_upper},
        "k_norm_mean_by_cell": {cell: float(np.mean(values)) for cell, values in sorted(cells.items())},
    }


def counterfactual_features(square: CounterfactualSquare,
                            vectors: dict[str, np.ndarray]) -> dict[str, float]:
    a, b = vectors[square.base], vectors[square.other]
    transformed_a, transformed_b = vectors[square.transformed_base], vectors[square.transformed_other]
    return {**pair_features(a, b),
            "k_norm": k_norm(a, transformed_a, b, transformed_b),
            "k_relative": k_relative(a, transformed_a, b, transformed_b)}


def _probe(pairs: Sequence[Pair], features: Sequence[dict[str, float]],
           splits: Sequence[str], fields: tuple[str, ...], seed: int) -> tuple[dict, np.ndarray | None]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([[row[field] for field in fields] for row in features])
    y = np.asarray([pair.label == "contradiction" for pair in pairs], dtype=int)
    train = np.asarray([s == "train" for s in splits])
    test = np.asarray([s == "test" for s in splits])
    if not train.any() or not test.any() or len(set(y[train])) < 2 or len(set(y[test])) < 2:
        return {"skipped": "Insufficient train/test classes among eligible examples"}, None
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000,
                                                                      random_state=seed))
    classifier.fit(x[train], y[train])
    prediction = classifier.predict_proba(x[test])[:, 1]
    report = {
        "train": int(train.sum()), "test": int(test.sum()),
        "average_precision": float(average_precision_score(y[test], prediction)),
        "roc_auc": float(roc_auc_score(y[test], prediction)),
        "review_budgets": review_budget_metrics(list(y[test]), list(prediction)),
    }
    return report, prediction


def run_k_ablation(pairs: Sequence[Pair], squares: Sequence[CounterfactualSquare],
                   vectors: dict[str, np.ndarray], splits: Sequence[str], seed: int = 42) -> dict:
    if len(pairs) != len(squares) or len(pairs) != len(splits) or not pairs:
        raise ValueError("Pairs, squares, and split lists must align and be nonempty")
    if any(pair.id != square.pair_id for pair, square in zip(pairs, squares)):
        raise ValueError("Pair/square order mismatch")
    original_count = len(pairs)
    kept = purged_indices(pairs, splits)
    pairs = [pairs[i] for i in kept]
    squares = [squares[i] for i in kept]
    splits = [splits[i] for i in kept]
    if not pairs:
        return {"skipped": "Every eligible example was purged due to statement overlap",
                "eligible_count": 0, "purged_exact_statement_overlap": original_count}
    features = [counterfactual_features(square, vectors) for square in squares]
    base, base_prediction = _probe(pairs, features, splits, BASE_FIELDS, seed)
    augmented, augmented_prediction = _probe(pairs, features, splits,
                                             BASE_FIELDS + ("k_norm", "k_relative"), seed)
    classes: dict[str, list[float]] = {}
    for pair, row in zip(pairs, features):
        classes.setdefault(pair.label, []).append(row["k_norm"])
    summary = {label: {"count": len(vals), "k_mean": float(np.mean(vals)),
                       "k_median": float(np.median(vals))} for label, vals in classes.items()}
    result = {"eligible_count": len(pairs), "purged_exact_statement_overlap": original_count - len(pairs),
              "class_counts": dict(Counter(p.label for p in pairs)),
              "k_descriptives": summary, "baseline": base, "with_k": augmented}
    if "average_precision" in base and "average_precision" in augmented:
        result["delta_average_precision"] = augmented["average_precision"] - base["average_precision"]
        result["delta_recall_at_10pct_review"] = (
            augmented["review_budgets"][1]["recall"] - base["review_budgets"][1]["recall"])
        test_pairs = [pair for pair, split in zip(pairs, splits) if split == "test"]
        result["delta_ap_subject_cluster_ci_95"] = grouped_ap_delta_interval(
            [int(pair.label == "contradiction") for pair in test_pairs],
            base_prediction, augmented_prediction, [pair.subject for pair in test_pairs], seed=seed)
    return result
