"""Dataset and split-integrity auditing, independent of any model/API."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Sequence

from .dataset import split_relation, split_subject, split_template
from .schema import Pair


def dataset_fingerprint(pairs: Sequence[Pair]) -> str:
    """Content hash in dataset order, including labels and all schema fields."""
    canonical = json.dumps([asdict(pair) for pair in pairs], sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_pairs(pairs: Sequence[Pair], seed: int = 42) -> dict:
    if not pairs:
        raise ValueError("Cannot audit an empty dataset")
    schemas = {
        "subject": ("subject", lambda p: split_subject(p.subject, seed)),
        "template": ("template_family", lambda p: split_template(p.template_family, seed)),
        "relation": ("relation_family", lambda p: split_relation(p.relation_family, seed)),
    }
    reports = {}
    for mode, (group_field, assign) in schemas.items():
        seen_groups: dict[str, set[str]] = defaultdict(set)
        class_support: dict[str, Counter] = defaultdict(Counter)
        full_statements: dict[str, set[str]] = defaultdict(set)
        for pair in pairs:
            split = assign(pair)
            seen_groups[getattr(pair, group_field)].add(split)
            class_support[split][pair.label] += 1
            full_statements[split].update((pair.text_a, pair.text_b))
        leakage = [key for key, assignments in seen_groups.items() if len(assignments) > 1]
        assigned = [assign(pair) for pair in pairs]
        retained = purged_indices(pairs, assigned)
        retained_texts: dict[str, set[str]] = defaultdict(set)
        for index in retained:
            retained_texts[assigned[index]].update((pairs[index].text_a, pairs[index].text_b))
        remaining_overlap = sum(len(retained_texts[a] & retained_texts[b]) for a, b in
                                (("train", "test"), ("train", "validation"), ("validation", "test")))
        reports[mode] = {
            "group_field": group_field, "unique_groups": len(seen_groups),
            "group_overlap": leakage,
            "class_support": {split: dict(counts) for split, counts in class_support.items()},
            "exact_statement_overlap_train_test": len(full_statements["train"] & full_statements["test"]),
            "exact_statement_overlap_train_validation": len(full_statements["train"] & full_statements["validation"]),
            "exact_statement_overlap_validation_test": len(full_statements["validation"] & full_statements["test"]),
            "purged_records": len(pairs) - len(retained),
            "remaining_overlap_after_purge": remaining_overlap,
        }
    normalized_pairs = defaultdict(list)
    for pair in pairs:
        normalized_pairs[tuple(sorted((pair.text_a, pair.text_b)))].append(pair)
    duplicates = sum(max(0, len(group) - 1) for group in normalized_pairs.values())
    conflicting_labels = [sorted({p.id for p in group}) for group in normalized_pairs.values()
                          if len({p.label for p in group}) > 1]
    return {
        "pair_count": len(pairs), "fingerprint_sha256": dataset_fingerprint(pairs),
        "label_counts": dict(Counter(p.label for p in pairs)),
        "generated_count": sum(pair.generated for pair in pairs),
        "duplicate_text_pairs": duplicates,
        "same_text_pair_conflicting_labels": conflicting_labels,
        "splits": reports,
        "interpretation": "Group-overlap checks are mechanical; they do not establish label validity or generalization.",
    }


def purged_indices(pairs: Sequence[Pair], splits: Sequence[str]) -> list[int]:
    """Remove exact statement overlap, preserving test > validation > train.

    This protects the measured test set from seeing exact text during
    training. It does NOT solve semantic paraphrase/subject/template leakage.
    """
    if len(pairs) != len(splits):
        raise ValueError("Pairs and splits must align")
    if any(split not in {"train", "validation", "test"} for split in splits):
        raise ValueError("Unknown split name")
    test_texts = {text for pair, split in zip(pairs, splits) if split == "test"
                  for text in (pair.text_a, pair.text_b)}
    validation = [i for i, (pair, split) in enumerate(zip(pairs, splits))
                  if split == "validation" and pair.text_a not in test_texts
                  and pair.text_b not in test_texts]
    validation_set = set(validation)
    validation_texts = {text for i in validation for text in (pairs[i].text_a, pairs[i].text_b)}
    blocked_texts = test_texts | validation_texts
    kept = []
    for i, (pair, split) in enumerate(zip(pairs, splits)):
        if split == "test":
            kept.append(i)
        elif split == "validation":
            if i in validation_set:
                kept.append(i)
        elif pair.text_a not in blocked_texts and pair.text_b not in blocked_texts:
            kept.append(i)
    return kept
