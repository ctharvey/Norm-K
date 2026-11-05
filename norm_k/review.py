"""Append-only human review and consensus, separate from synthetic oracle labels."""

from __future__ import annotations

import json
import hashlib
import random
import threading
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from .schema import LABELS, Pair

REVIEW_LABELS = LABELS | {"ambiguous"}
CONFIDENCES = {"low", "medium", "high"}


@dataclass(frozen=True)
class ReviewItem:
    id: str
    text_a: str
    text_b: str
    source: str
    source_row: int
    reference_label: str | None = None
    reference_split: str | None = None
    source_url: str | None = None
    license: str = "UNVERIFIED"
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.text_a.strip() or not self.text_b.strip() or not self.source:
            raise ValueError("Review item is missing identity, text, or source")
        if self.reference_label is not None and self.reference_label not in LABELS:
            raise ValueError("Reference label is not in the Norm-K taxonomy")
        if self.source_row < 1:
            raise ValueError("Source row must be a positive integer")

    def for_reviewer(self) -> dict[str, str]:
        """Never expose source/reference labels to the human reviewer UI."""
        return {"id": self.id, "text_a": self.text_a, "text_b": self.text_b}


@dataclass(frozen=True)
class Review:
    review_id: str
    pair_id: str
    reviewer: str
    label: str
    confidence: str
    notes: str
    timestamp_utc: str

    def __post_init__(self) -> None:
        if not self.review_id or not self.pair_id or not 1 <= len(self.reviewer) <= 80:
            raise ValueError("Review requires ID, pair ID, and reviewer (1–80 characters)")
        if self.label not in REVIEW_LABELS or self.confidence not in CONFIDENCES:
            raise ValueError("Invalid review label or confidence")
        if len(self.notes) > 2000 or any(ord(c) < 32 and c not in "\n\t" for c in self.notes):
            raise ValueError("Review note is too long or contains control characters")


def save_items(path: str | Path, items: Iterable[ReviewItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for item in items:
            output.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")


def load_items(path: str | Path) -> list[ReviewItem]:
    items, seen = [], set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = ReviewItem(**json.loads(line))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Review queue line {line_no}: {exc}") from exc
            if item.id in seen:
                raise ValueError(f"Duplicate review queue ID: {item.id}")
            items.append(item)
            seen.add(item.id)
    return items


def queue_from_pairs(pairs: Sequence[Pair], per_label: int = 20,
                     seed: int = 42) -> list[ReviewItem]:
    if per_label < 1:
        raise ValueError("per_label must be positive")
    grouped: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.label].append(pair)
    rng = random.Random(seed)
    selected = []
    for label in sorted(grouped):
        pool = grouped[label]
        selected.extend(rng.sample(pool, min(per_label, len(pool))))
    rng.shuffle(selected)
    return [ReviewItem(id="pair-" + hashlib.sha256(f"norm-k-blind-v1:{seed}:{pair.id}".encode()).hexdigest()[:20],
                       text_a=pair.text_a, text_b=pair.text_b,
                       source="norm-k-synthetic", source_row=i + 1,
                       reference_label=pair.label,
                       metadata={"generated": "true", "template": pair.template_family,
                                 "source_pair_id": pair.id})
            for i, pair in enumerate(selected)]


class AnnotationLog:
    """Immutable append-only JSONL evidence; repeated reviews supersede in views only."""

    def __init__(self, path: str | Path, queue_ids: Iterable[str] | None = None):
        self.path = Path(path)
        self.allowed_ids = set(queue_ids) if queue_ids is not None else None
        self.lock = threading.Lock()

    def read_all(self) -> list[Review]:
        if not self.path.exists():
            return []
        with self.lock:
            with self.path.open("r", encoding="utf-8") as stream:
                return [Review(**json.loads(line)) for line in stream if line.strip()]

    def add(self, pair_id: str, reviewer: str, label: str,
            confidence: str = "medium", notes: str = "") -> Review:
        if self.allowed_ids is not None and pair_id not in self.allowed_ids:
            raise ValueError("Pair ID not found in the review queue")
        review = Review(review_id=str(uuid.uuid4()), pair_id=pair_id,
                        reviewer=reviewer.strip(), label=label, confidence=confidence,
                        notes=notes, timestamp_utc=datetime.now(timezone.utc).isoformat())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(review), ensure_ascii=False) + "\n")
                stream.flush()
        return review


def latest_reviews(reviews: Iterable[Review]) -> dict[tuple[str, str], Review]:
    """Collapse repeated edits from the *same* reviewer without deleting history."""
    latest = {}
    for review in reviews:
        latest[(review.pair_id, review.reviewer)] = review
    return latest


def consensus_report(items: Sequence[ReviewItem], reviews: Sequence[Review],
                     minimum_reviewers: int = 2) -> dict:
    if minimum_reviewers < 2:
        raise ValueError("At least two independent reviewers required for consensus")
    ids = {item.id for item in items}
    latest = latest_reviews(reviews)
    by_pair: dict[str, list[Review]] = defaultdict(list)
    for (pair_id, _), review in latest.items():
        if pair_id not in ids:
            raise ValueError(f"Annotation references unknown pair {pair_id}")
        by_pair[pair_id].append(review)
    resolved, disagreement, insufficient, ambiguous = [], [], [], []
    for item in items:
        records = by_pair.get(item.id, [])
        if len(records) < minimum_reviewers:
            insufficient.append(item.id)
        elif any(review.label == "ambiguous" for review in records):
            ambiguous.append(item.id)
        elif len({review.label for review in records}) == 1:
            resolved.append({"pair_id": item.id, "label": records[0].label,
                             "reviewers": sorted(r.reviewer for r in records),
                             "review_count": len(records)})
        else:
            disagreement.append({"pair_id": item.id,
                                 "votes": dict(Counter(r.label for r in records))})
    return {
        "queue_size": len(items), "review_records": len(reviews),
        "distinct_reviewers": sorted({r.reviewer for r in reviews}),
        "minimum_reviewers": minimum_reviewers,
        "resolved": resolved, "disagreement": disagreement,
        "ambiguous": ambiguous, "insufficient_review_ids": insufficient,
        "pairwise_agreement": pairwise_agreement(list(latest.values())),
        "warning": "Unanimity is not proof of correctness; source metadata and fact scope still require audit.",
    }


def pairwise_agreement(reviews: Sequence[Review]) -> list[dict]:
    """Observed agreement and Cohen's kappa on overlapping reviewer pairs.

    Kappa is undefined when chance agreement is 100%, or with fewer than two
    shared judgments. 'ambiguous' is a category, not silently discarded.
    """
    latest = latest_reviews(reviews)
    by_reviewer: dict[str, dict[str, str]] = defaultdict(dict)
    for (pair_id, reviewer), review in latest.items():
        by_reviewer[reviewer][pair_id] = review.label
    result = []
    for first, second in combinations(sorted(by_reviewer), 2):
        shared = sorted(set(by_reviewer[first]) & set(by_reviewer[second]))
        if not shared:
            continue
        left = [by_reviewer[first][pair_id] for pair_id in shared]
        right = [by_reviewer[second][pair_id] for pair_id in shared]
        observed = sum(a == b for a, b in zip(left, right)) / len(shared)
        left_freq = Counter(left)
        right_freq = Counter(right)
        expected = sum(left_freq[label] * right_freq[label] for label in REVIEW_LABELS) / (len(shared) ** 2)
        kappa = ((observed - expected) / (1.0 - expected)
                 if len(shared) >= 2 and expected < 1.0 else None)
        result.append({"reviewers": [first, second], "shared_pairs": len(shared),
                       "observed_agreement": observed, "expected_by_marginals": expected,
                       "cohens_kappa": kappa})
    return result
