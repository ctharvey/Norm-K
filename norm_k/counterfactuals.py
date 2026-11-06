"""Explicit, restricted text transformations for pair-level K experiments.

This is NOT a general language parser. Unsupported sentences return None.
The transformation must be counterfactually valid for both pair members;
the syntax checks below are necessary, not sufficient, for semantic validity.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .dataset import NAMES
from .schema import Pair


@dataclass(frozen=True)
class CounterfactualSquare:
    pair_id: str
    operation: str
    label: str
    base: str
    other: str
    transformed_base: str
    transformed_other: str


_TOGGLE_PATTERNS = (
    (r"\bdoes not own\b", "owns"),
    (r"\bdoes not live\b", "lives"),
    (r"\bdoes not work\b", "works"),
    (r"\bdoes not reside\b", "resides"),
    (r"\bis not\b", "is"),
    (r"\bwas not\b", "was"),
    (r"\bowns\b", "does not own"),
    (r"\blives\b", "does not live"),
    (r"\bworks\b", "does not work"),
    (r"\bresides\b", "does not reside"),
    (r"\bis\b", "is not"),
    (r"\bwas\b", "was not"),
)


def toggle_negation(text: str) -> str | None:
    """Invert one supported finite clause; avoid guessing at other grammar."""
    for pattern, replacement in _TOGGLE_PATTERNS:
        result, count = re.subn(pattern, replacement, text, count=1)
        if count:
            return result
    return None


def replace_subject(text: str, old: str, new: str) -> str | None:
    """Whole-token substitution. Only apply where the named subject exists."""
    if not old or not new or old == new:
        return None
    result, count = re.subn(r"\b" + re.escape(old) + r"\b", new, text)
    return result if count else None


def _new_subject(pair: Pair) -> str:
    candidates = [name for name in NAMES if name not in
                  re.findall(r"\b[A-Z][a-z]+\b", pair.text_a + " " + pair.text_b)]
    if not candidates:
        raise ValueError("No safe replacement subject available")
    digest = hashlib.sha256(pair.id.encode("utf-8")).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def is_mirror(square: CounterfactualSquare) -> bool:
    """True when T swaps the pair, so K = 2(A - B) for any encoder."""
    return square.transformed_base == square.other and square.transformed_other == square.base


def counterfactual_square(pair: Pair, operation: str) -> CounterfactualSquare | None:
    """Construct A, B, T(A), T(B) for a valid *syntactic* counterfactual.

    The pair-level residual is K(A,B;T) = E(A) + E(T(B)) - E(T(A)) - E(B).
    This checks whether the A→B displacement commutes with transformation T.
    """
    if operation == "negation":
        transformed_a, transformed_b = toggle_negation(pair.text_a), toggle_negation(pair.text_b)
    elif operation == "subject_swap":
        # Do not apply a one-sided substitution to different-subject pairs.
        replacement = _new_subject(pair)
        transformed_a = replace_subject(pair.text_a, pair.subject, replacement)
        transformed_b = replace_subject(pair.text_b, pair.subject, replacement)
    else:
        raise ValueError(f"Unknown counterfactual operation: {operation}")
    if not transformed_a or not transformed_b:
        return None
    if transformed_a == pair.text_a or transformed_b == pair.text_b:
        return None
    return CounterfactualSquare(pair.id, operation, pair.label, pair.text_a, pair.text_b,
                                transformed_a, transformed_b)
