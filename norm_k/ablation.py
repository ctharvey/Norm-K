"""Paired, same-cohort ablations of K-derived features.

Only available on examples with a supported counterfactual transformation.
Both models use the exact same rows/splits to avoid coverage confounding.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from .counterfactuals import CounterfactualSquare
from .audit import purged_indices
from .evaluation import review_budget_metrics
from .geometry import k_norm, k_relative, pair_features
from .schema import Pair
from .statistics import grouped_ap_delta_interval


BASE_FIELDS = ("cosine", "euclidean", "hoyer_delta", "mean_abs_delta", "mean_product")


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
