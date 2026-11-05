"""Provenance-preserving, offline-only benchmark CSV/TSV/JSONL importer.

No implicit NLI label mapping: 'neutral' is not automatically 'compatible',
and directional entailment is not generally paraphrase equivalence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .review import ReviewItem
from .schema import LABELS


def _records(path: Path, fmt: str) -> Iterable[dict]:
    if fmt in ("csv", "tsv"):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream, delimiter="," if fmt == "csv" else "\t")
    elif fmt == "jsonl":
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} must be an object")
                yield value
    else:
        raise ValueError(f"Unsupported benchmark format {fmt}")


def import_benchmark(path: str | Path, source: str, a_field: str = "text_a",
                     b_field: str = "text_b", label_field: str | None = None,
                     label_map: dict[str, str] | None = None,
                     split_field: str | None = None, source_url: str | None = None,
                     license_name: str = "UNVERIFIED", max_rows: int = 100_000) -> tuple[list[ReviewItem], dict]:
    """Import local data without guessing truth conditions or licensing status."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not source.strip() or not a_field or not b_field or max_rows < 1:
        raise ValueError("Source name, column names, and positive max_rows are required")
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in ("csv", "tsv", "jsonl"):
        raise ValueError("Expected .csv, .tsv, or .jsonl input")
    mapping = label_map or {}
    if any(target not in LABELS for target in mapping.values()):
        raise ValueError("Every explicit mapped label must be in the Norm-K taxonomy")
    items = []
    unmapped = Counter()
    seen = set()
    for index, row in enumerate(_records(path, fmt), 1):
        if index > max_rows:
            raise ValueError(f"Input exceeds max_rows={max_rows}; refusing oversized import")
        try:
            a, b = row[a_field], row[b_field]
        except KeyError as exc:
            raise ValueError(f"Row {index} missing requested column {exc}") from exc
        if not isinstance(a, str) or not isinstance(b, str) or not a.strip() or not b.strip():
            raise ValueError(f"Row {index} contains missing/non-text statements")
        raw_label = None if label_field is None else row.get(label_field)
        if label_field is not None and label_field not in row:
            raise ValueError(f"Row {index} missing label column {label_field}")
        normalized = mapping.get(str(raw_label)) if raw_label is not None else None
        if raw_label is not None and normalized is None:
            unmapped[str(raw_label)] += 1
        # Row-indexed identity preserves repeated raw examples, revealing any
        # duplication in later audits rather than silently merging records.
        id_digest = hashlib.sha256(f"{source}:{index}:{a}:{b}".encode("utf-8")).hexdigest()[:16]
        identifier = f"{source}-{index:06d}-{id_digest}"
        if identifier in seen:
            raise ValueError("Generated a duplicate source ID")
        seen.add(identifier)
        metadata = {"raw_label": str(raw_label)} if raw_label is not None else {}
        items.append(ReviewItem(
            id=identifier, text_a=a, text_b=b, source=source, source_row=index,
            reference_label=normalized,
            reference_split=str(row[split_field]) if split_field and row.get(split_field) is not None else None,
            source_url=source_url, license=license_name, metadata=metadata,
        ))
    if not items:
        raise ValueError("No benchmark examples were imported")
    manifest = {
        "source": source, "source_file": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_url": source_url, "license_claim_unverified": license_name,
        "format": fmt, "text_fields": [a_field, b_field],
        "label_field": label_field, "explicit_label_map": mapping,
        "split_field": split_field, "record_count": len(items),
        "mapped_labels": dict(Counter(item.reference_label for item in items if item.reference_label)),
        "unmapped_raw_labels": dict(unmapped),
        "warning": "No semantic label mapping is inferred; provenance/license must be independently verified.",
    }
    return items, manifest
