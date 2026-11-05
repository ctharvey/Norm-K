"""Nonparametric uncertainty estimates for preregistered research comparisons."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_mean_interval(values: Sequence[float], confidence: float = 0.95,
                            repetitions: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of independent observations.

    This does not correct for dependence from shared templates or subjects; a
    real experiment must bootstrap at the proper grouping level instead.
    """
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or len(sample) < 2 or not np.isfinite(sample).all():
        raise ValueError("At least two finite observations required")
    if not 0 < confidence < 1 or repetitions < 100:
        raise ValueError("Invalid confidence or repetitions")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(repetitions, len(sample)))
    means = sample[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def grouped_ap_delta_interval(labels: Sequence[int], baseline_scores: Sequence[float],
                              augmented_scores: Sequence[float], groups: Sequence[str],
                              confidence: float = 0.95, repetitions: int = 300,
                              seed: int = 42) -> dict | None:
    """Paired bootstrap of AP improvement, resampling *subjects* as clusters.

    Correlated templates and relations remain potential confounders. The CI is
    descriptive and cannot establish semantic novelty on synthetic labels.
    """
    from sklearn.metrics import average_precision_score

    y = np.asarray(labels, dtype=int)
    base = np.asarray(baseline_scores, dtype=float)
    augmented = np.asarray(augmented_scores, dtype=float)
    names = np.asarray(groups)
    if not len(y) == len(base) == len(augmented) == len(names) or not len(y):
        raise ValueError("Bootstrap inputs must have equal nonzero lengths")
    if any(not np.isfinite(array).all() for array in (base, augmented)):
        raise ValueError("Invalid predicted scores")
    if not 0 < confidence < 1 or repetitions < 100:
        raise ValueError("Invalid confidence or repetitions")
    unique = np.unique(names)
    # Extremely few independent subject clusters yield a deceptively precise
    # interval. Return no interval rather than overstate certainty.
    if len(unique) < 10 or len(np.unique(y)) != 2:
        return None
    group_indices = [np.flatnonzero(names == name) for name in unique]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(unique), size=len(unique))
        selected = np.concatenate([group_indices[index] for index in sampled])
        if len(np.unique(y[selected])) != 2:
            continue
        difference = (average_precision_score(y[selected], augmented[selected]) -
                      average_precision_score(y[selected], base[selected]))
        deltas.append(difference)
    if len(deltas) < repetitions // 2:
        return None
    tail = (1 - confidence) / 2
    return {"ci_lower": float(np.quantile(deltas, tail)),
            "ci_upper": float(np.quantile(deltas, 1 - tail)),
            "resamples_accepted": len(deltas), "subject_clusters": len(unique),
            "method": "paired subject-cluster percentile bootstrap"}
