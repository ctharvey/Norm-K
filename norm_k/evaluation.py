"""Transparent baseline evaluation; synthetic labels are not scientific evidence."""

from __future__ import annotations

import re
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from .geometry import pair_features
from .schema import Pair


def token_features(a: str, b: str) -> dict[str, float]:
    tokens_a = re.findall(r"[a-z0-9]+", a.lower())
    tokens_b = re.findall(r"[a-z0-9]+", b.lower())
    set_a, set_b = set(tokens_a), set(tokens_b)
    union = set_a | set_b
    a_neg, b_neg = "not" in set_a or "never" in set_a, "not" in set_b or "never" in set_b
    numbers_a = Counter(t for t in tokens_a if t.isdecimal())
    numbers_b = Counter(t for t in tokens_b if t.isdecimal())
    return {
        "token_jaccard": len(set_a & set_b) / len(union) if union else 1.0,
        "negation_disagreement": float(a_neg != b_neg),
        "number_disagreement": float(numbers_a != numbers_b),
        "token_symmetric_difference": float(len(set_a ^ set_b)),
    }


def features_for_pairs(pairs: Sequence[Pair],
                       vectors: dict[str, np.ndarray]) -> list[dict[str, float]]:
    result = []
    for pair in pairs:
        result.append({**pair_features(vectors[pair.text_a], vectors[pair.text_b]),
                       **token_features(pair.text_a, pair.text_b)})
    return result


def review_budget_metrics(labels: Sequence[bool], scores: Sequence[float],
                          budgets: Sequence[float] = (0.05, 0.1, 0.2)) -> list[dict[str, float | int]]:
    """How many contradictions are found if we review only the top-scored pairs?"""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        raise ValueError("Cannot evaluate an empty set")
    if any(not np.isfinite(s) for s in scores):
        raise ValueError("Scores must be finite")
    if any(not 0 < budget <= 1 for budget in budgets):
        raise ValueError("Budgets must lie in (0, 1]")
    positives = sum(bool(label) for label in labels)
    ranking = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    result = []
    for budget in budgets:
        count = max(1, int(np.ceil(len(labels) * budget)))
        hits = sum(bool(labels[index]) for index in ranking[:count])
        result.append({"budget": float(budget), "reviewed": count, "hits": hits,
                       "precision": hits / count,
                       "recall": hits / positives if positives else 0.0})
    return result


def baseline_report(pairs: Sequence[Pair], feature_rows: Sequence[dict[str, float]]) -> dict:
    if len(pairs) != len(feature_rows) or not pairs:
        raise ValueError("Expected matching nonempty pairs and feature rows")
    y = [pair.label == "contradiction" for pair in pairs]
    scorers = {
        "cosine_high": lambda f: f["cosine"],
        "cosine_low": lambda f: -f["cosine"],
        "distance_high": lambda f: f["euclidean"],
        "hoyer_high": lambda f: f["hoyer_delta"],
        "lexical_negation": lambda f: f["negation_disagreement"],
        "lexical_number": lambda f: f["number_disagreement"],
    }
    return {
        "count": len(pairs),
        "class_counts": dict(Counter(pair.label for pair in pairs)),
        "scores": {name: review_budget_metrics(y, [scorer(f) for f in feature_rows])
                   for name, scorer in scorers.items()},
    }


def train_linear_probe(pairs: Sequence[Pair], features: Sequence[dict[str, float]],
                       splits: Sequence[str], seed: int = 42) -> dict:
    """Small supervised control, trained only on subject-held-out train records."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Linear probe requires pip install -e '.[research]'") from exc
    if not len(pairs) == len(features) == len(splits):
        raise ValueError("Mismatched evaluation arrays")
    # No metadata or synthetic label-derived features are allowed in the model.
    fields = ("cosine", "euclidean", "hoyer_delta", "mean_abs_delta", "mean_product",
              "token_jaccard", "negation_disagreement", "number_disagreement",
              "token_symmetric_difference")
    x = np.asarray([[f[field] for field in fields] for f in features])
    y = np.asarray([p.label == "contradiction" for p in pairs], dtype=int)
    train = np.asarray([s == "train" for s in splits])
    test = np.asarray([s == "test" for s in splits])
    if not train.any() or not test.any() or len(set(y[train])) != 2 or len(set(y[test])) != 2:
        return {"skipped": "Insufficient positive/negative examples in subject-held-out train/test groups"}
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1,
                                                                      random_state=seed))
    classifier.fit(x[train], y[train])
    probabilities = classifier.predict_proba(x[test])[:, 1]
    return {
        "train_count": int(train.sum()), "test_count": int(test.sum()),
        "feature_names": list(fields),
        "average_precision": float(average_precision_score(y[test], probabilities)),
        "roc_auc": float(roc_auc_score(y[test], probabilities)),
        "review_budgets": review_budget_metrics(list(y[test]), list(probabilities)),
        "scope": "subject-held-out synthetic generated pairs; not a validated NLI result",
    }


def plot_scatter(pairs: Sequence[Pair], features: Sequence[dict[str, float]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(path).parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {
        "contradiction": "#d62728", "equivalent": "#2ca02c", "compatible": "#1f77b4",
        "temporal_update": "#ff7f0e", "unrelated": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, color in palette.items():
        indices = [i for i, pair in enumerate(pairs) if pair.label == label]
        ax.scatter([features[i]["cosine"] for i in indices],
                   [features[i]["hoyer_delta"] for i in indices],
                   c=color, alpha=0.45, s=18, label=f"{label} ({len(indices)})")
    ax.set(xlabel="Cosine similarity", ylabel="Hoyer sparsity of E(B) − E(A)",
           title="Embedding pair geometry (encoder and data labels must be disclosed)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_k_distribution(groups: dict[str, Sequence[float]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(path).parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in sorted(groups.items()):
        ax.hist(values, bins=20, alpha=0.5, label=name)
    ax.set(xlabel="Four-point residual norm ‖K‖", ylabel="Example count",
           title="Interaction residual distribution (NOT contradiction probability)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_dimension_curve(results: dict[str, dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(path).parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, result in results.items():
        runs = result.get("ranks", [])
        if runs:
            ax.plot([row["dimensions"] for row in runs],
                    [row["review_budgets"][1]["recall"] for row in runs],
                    marker="o", label=label)
    ax.set(xlabel="PCA projection dimension", ylabel="Contradiction recall @ 10% review budget",
           ylim=(0, 1), title="Low-rank synthetic smoke only — not semantic validation")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_pair_k_scatter(rows: Sequence[tuple[str, float, float]], path: str) -> None:
    """Render cosine vs pair-level K for the *eligible* transformation cohort."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(path).parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "contradiction": "#d62728", "equivalent": "#2ca02c", "compatible": "#1f77b4",
        "temporal_update": "#ff7f0e", "unrelated": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, color in colors.items():
        selected = [(c, k) for t, c, k in rows if t == label]
        if selected:
            ax.scatter([c for c, _ in selected], [k for _, k in selected],
                       c=color, alpha=0.4, s=15, label=f"{label} ({len(selected)})")
    ax.set(xlabel="Cosine(E(A), E(B))", ylabel="‖E(A) + E(T(B)) − E(T(A)) − E(B)‖",
           title="Pair-level counterfactual K (supported transformations only)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
