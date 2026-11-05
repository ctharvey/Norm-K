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
