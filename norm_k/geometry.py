"""Frozen-embedding geometry and four-point interaction residuals.

These operations describe vectors, not the truth of propositions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(value: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not result.size:
        raise ValueError("Expected a nonempty one-dimensional embedding")
    if not np.isfinite(result).all():
        raise ValueError("Embedding contains NaN or infinity")
    return result


def _pair(a: ArrayLike, b: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left, right = _vector(a), _vector(b)
    if left.shape != right.shape:
        raise ValueError("Cannot compare embeddings of different dimensions")
    return left, right


def difference(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Directed displacement from a to b: E(b) - E(a)."""
    left, right = _pair(a, b)
    return right - left


def cosine(a: ArrayLike, b: ArrayLike) -> float:
    left, right = _pair(a, b)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def euclidean(a: ArrayLike, b: ArrayLike) -> float:
    return float(np.linalg.norm(difference(a, b)))


def hoyer_sparsity(vector: ArrayLike) -> float:
    """0 for equal-magnitude dense coordinates, 1 for a one-hot vector.

    Zero vectors return 0 as a declared convention; they are not evidence of
    low sparsity and should be inspected separately in any classifier.
    """
    v = _vector(vector)
    if v.size == 1:
        return 0.0
    l2 = float(np.linalg.norm(v))
    if l2 == 0:
        return 0.0
    return float(np.clip((np.sqrt(v.size) - np.linalg.norm(v, ord=1) / l2) /
                         (np.sqrt(v.size) - 1.0), 0.0, 1.0))


def four_point(base: ArrayLike, first: ArrayLike, second: ArrayLike,
               both: ArrayLike) -> NDArray[np.float64]:
    """K = E(s) + E(second(first(s))) - E(first(s)) - E(second(s)).

    The caller must ensure that the four texts implement the intended
    transformation square; this function checks numerical validity only.
    """
    b, f = _pair(base, first)
    s, fs = _pair(second, both)
    if b.shape != s.shape:
        raise ValueError("All four embeddings must have the same dimension")
    return b + fs - f - s


def k_norm(base: ArrayLike, first: ArrayLike, second: ArrayLike,
           both: ArrayLike) -> float:
    return float(np.linalg.norm(four_point(base, first, second, both)))


def k_relative(base: ArrayLike, first: ArrayLike, second: ArrayLike,
               both: ArrayLike, epsilon: float = 1e-9) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    base_v, first_v = _pair(base, first)
    _, second_v = _pair(base, second)
    denominator = (np.linalg.norm(first_v - base_v) +
                   np.linalg.norm(second_v - base_v) + epsilon)
    return k_norm(base, first, second, both) / float(denominator)


def pair_features(a: ArrayLike, b: ArrayLike) -> dict[str, float]:
    delta = difference(a, b)
    return {
        "cosine": cosine(a, b),
        "euclidean": float(np.linalg.norm(delta)),
        "hoyer_delta": hoyer_sparsity(delta),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "mean_product": float(np.mean(_vector(a) * _vector(b))),
        "delta_zero": float(not np.any(delta)),
    }

