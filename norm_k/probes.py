"""Small, supervised probes over frozen difference vectors.

Only training data is used to fit scaling and PCA. Results on generated/hash
data are tests of methodology, not evidence of recovered language semantics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .evaluation import review_budget_metrics
from .schema import Pair


def low_rank_probe(pairs: Sequence[Pair], vectors: dict[str, np.ndarray],
                   splits: Sequence[str], seed: int = 42,
                   ranks: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128)) -> dict:
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Low-rank probes require pip install -e '.[research]'") from exc
    if len(pairs) != len(splits) or not pairs:
        raise ValueError("Expected matching nonempty pair/split sequences")
    x = np.vstack([np.abs(vectors[p.text_b] - vectors[p.text_a]) for p in pairs])
    labels = np.asarray([p.label == "contradiction" for p in pairs], dtype=int)
    train = np.asarray([s == "train" for s in splits])
    test = np.asarray([s == "test" for s in splits])
    if len(set(labels[train])) < 2 or len(set(labels[test])) < 2 or train.sum() < 4:
        return {"skipped": "Insufficient split classes for supervised low-rank probe"}
    usable = sorted({r for r in ranks if 1 <= r < min(int(train.sum()), x.shape[1])})
    if not usable:
        return {"skipped": "No usable projection dimensions"}
    # Fit PCA once on the training split, then compare prefix ranks.
    scaler = StandardScaler().fit(x[train])
    x_train = scaler.transform(x[train])
    x_test = scaler.transform(x[test])
    pca = PCA(n_components=max(usable), svd_solver="randomized", random_state=seed)
    projected_train = pca.fit_transform(x_train)
    projected_test = pca.transform(x_test)
    results = []
    for rank in usable:
        classifier = LogisticRegression(max_iter=1000, C=0.1, random_state=seed)
        classifier.fit(projected_train[:, :rank], labels[train])
        score = classifier.predict_proba(projected_test[:, :rank])[:, 1]
        results.append({
            "dimensions": rank,
            "explained_variance_fraction": float(np.sum(pca.explained_variance_ratio_[:rank])),
            "average_precision": float(average_precision_score(labels[test], score)),
            "roc_auc": float(roc_auc_score(labels[test], score)),
            "review_budgets": review_budget_metrics(list(labels[test]), list(score)),
        })
    return {
        "train_count": int(train.sum()), "test_count": int(test.sum()),
        "method": "PCA(|E(B)-E(A)|) + regularized logistic regression",
        "scope": "frozen hash-control and synthetic labels; NOT semantic contradiction validation",
        "ranks": results,
    }
