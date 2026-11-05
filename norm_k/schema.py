"""Labeled-pair schema and explicit semantics for evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

LABELS = frozenset({"contradiction", "equivalent", "compatible", "temporal_update", "unrelated"})
CARDINALITIES = frozenset({"single", "multiple", "unknown"})


@dataclass(frozen=True)
class Pair:
    id: str
    text_a: str
    text_b: str
    label: str
    subject: str
    attribute: str
    relation_family: str
    template_family: str
    transformation: str
    cardinality: str
    same_fact: bool
    incompatible: bool
    same_valid_time: bool
    rationale: str
    generated: bool = True

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"Invalid label: {self.label}")
        if self.cardinality not in CARDINALITIES:
            raise ValueError(f"Invalid cardinality: {self.cardinality}")
        if not all((self.id, self.text_a.strip(), self.text_b.strip(), self.subject,
                    self.attribute, self.relation_family, self.template_family, self.rationale)):
            raise ValueError("A required pair field is empty")
        if self.label == "contradiction" and not (self.same_fact and self.incompatible and self.same_valid_time):
            raise ValueError("Contradictions must share fact and time and be incompatible")
        if self.label == "temporal_update" and self.same_valid_time:
            raise ValueError("Temporal updates cannot have identical valid time")


def save_pairs(path: str | Path, pairs: Iterable[Pair]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as output:
        for pair in pairs:
            output.write(json.dumps(asdict(pair), ensure_ascii=False, sort_keys=True) + "\n")


def load_pairs(path: str | Path) -> list[Pair]:
    result = []
    ids = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                pair = Pair(**json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid record on line {number}: {exc}") from exc
            if pair.id in ids:
                raise ValueError(f"Duplicate pair ID: {pair.id}")
            ids.add(pair.id)
            result.append(pair)
    return result
