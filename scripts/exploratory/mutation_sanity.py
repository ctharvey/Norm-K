"""Math-only exploratory score mutation checks; NOT a contradiction benchmark.

No external model, data, API, or benchmark is accessed. Assumes unit vectors.
Run: python scripts/exploratory/mutation_sanity.py
"""
from __future__ import annotations
import json
import numpy as np


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / n) if n else 0.0


def score_mutations(a: np.ndarray, b: np.ndarray, neg_a: np.ndarray,
                    neg_b: np.ndarray, floor: float = 1e-2) -> dict[str, float]:
    """No metric claims: algebraic hypotheses for fixed 4-corner embeddings."""
    a, b, neg_a, neg_b = (unit(x) for x in (a, b, neg_a, neg_b))
    da, db = neg_a - a, neg_b - b
    gain_a = float(np.dot(da, b))
    gain_b = float(np.dot(a, db))
    move_a = 1 - float(np.dot(a, neg_a))
    move_b = 1 - float(np.dot(b, neg_b))
    flip = (gain_a + gain_b) / 2
    interaction = float(np.dot(da-db, b-a)) / 2
    individual = -(move_a + move_b) / 2
    norm_k = float(np.linalg.norm(db-da))
    return {
        "flip": flip,
        "minimum_direction_gain": min(gain_a, gain_b),
        "normalized_flip": 0.5*(gain_a/max(move_a, floor)+gain_b/max(move_b, floor)),
        "anti_alignment": -cosine(da, db),
        "norm_k": norm_k,
        "pairwise_interaction": interaction,
        "individual_response": individual,
        "decomposition_residual": abs(flip-(interaction+individual)),
        "k_dot_identity_residual": abs(norm_k**2-(float(da@da+db@db-2*da@db))),
    }


def main() -> None:
    rng = np.random.default_rng(7)
    dims, n = 32, 10_000
    axis, anchor = (unit(rng.standard_normal(dims)) for _ in range(2))
    def reflect(x: np.ndarray) -> np.ndarray:
        return x-2*np.dot(x, axis)*axis
    strangers = rng.standard_normal((n, dims))
    strangers /= np.linalg.norm(strangers, axis=1)[:, None]
    anti = []
    for b in strangers:
        vals = score_mutations(anchor, b, reflect(anchor), reflect(b))
        anti.append(vals["anti_alignment"])
        assert vals["decomposition_residual"] < 1e-12
        assert vals["k_dot_identity_residual"] < 1e-12
    anti = np.asarray(anti)
    sanity = {
        "purpose": "Synthetic global-axis reflection is a counterexample, not real embedding evidence",
        "n_unrelated_pairs": n,
        "dimensions": dims,
        "maximal_false_anti_alignment_rate": float(np.mean(anti > 0.99)),
        "min_anti_alignment": float(anti.min()),
        "max_anti_alignment": float(anti.max()),
        "flip_same": score_mutations(anchor, anchor, reflect(anchor), reflect(anchor))["flip"],
        "flip_negated": score_mutations(anchor, reflect(anchor), reflect(anchor), anchor)["flip"],
        "algebraic_identities": "pass for all 10000 samples",
    }
    print(json.dumps(sanity, indent=2))

if __name__ == "__main__":
    main()
