"""Deterministic synthetic examples for engineering tests, NOT a gold benchmark.

Templates are deliberately simple and require an independent human audit before
their labels can be interpreted as experimental evidence.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .schema import Pair

NAMES = (
    "Alice", "Bob", "Charlie", "Diana", "Elena", "Farah", "George", "Hannah",
    "Isaac", "Jamal", "Kira", "Luis", "Maya", "Nolan", "Olivia", "Priya",
    "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe", "Arjun", "Beatrice", "Carmen", "Daniel", "Eli", "Fatima",
)
CITIES = ("Austin", "Denver", "Seattle", "Boston", "Chicago", "Phoenix", "Atlanta", "Miami", "Portland", "Dallas", "Houston", "Nashville")
ITEMS = ("bicycle", "guitar", "camera", "scooter", "laptop", "telescope", "skateboard", "watch")
JOBS = ("teacher", "engineer", "pilot", "editor", "designer", "chemist", "librarian", "carpenter")
LABEL_ORDER = ("contradiction", "equivalent", "compatible", "temporal_update", "unrelated")


def _different(rng: random.Random, items: tuple[str, ...], previous: str) -> str:
    return rng.choice([v for v in items if v != previous])


def _make(label: str, index: int, rng: random.Random) -> Pair:
    name = rng.choice(NAMES)
    other = _different(rng, NAMES, name)
    city = rng.choice(CITIES)
    city2 = _different(rng, CITIES, city)
    item = rng.choice(ITEMS)
    item2 = _different(rng, ITEMS, item)
    job = rng.choice(JOBS)
    job2 = _different(rng, JOBS, job)
    age = rng.randrange(20, 75)
    mode = index % 4
    meta: dict[str, object] = {
        "id": f"generated-{label}-{index:05d}", "label": label, "subject": name,
        "same_valid_time": True, "cardinality": "single", "same_fact": True,
        "incompatible": False, "generated": True,
    }

    if label == "contradiction":
        meta.update(incompatible=True, rationale="Same subject, exclusive attribute, and valid time; incompatible values.")
        if mode == 0:
            a, b = (f"In 2026, {name}'s primary home city is {city}.",
                    f"In 2026, {name}'s primary home city is {city2}.")
            relation, attribute, transform = "residence", "primary_home_city", "value_swap"
        elif mode == 1:
            a, b = (f"At noon on November 5, {name}'s office door at the {city} branch is open.",
                    f"At noon on November 5, {name}'s office door at the {city} branch is closed.")
            relation, attribute, transform = "state", "office_door_state", "antonym"
        elif mode == 2:
            a, b = (f"In 2026, {name} owns the {item} with serial number 417.",
                    f"In 2026, {name} does not own the {item} with serial number 417.")
            relation, attribute, transform = "ownership", "owns_serial_417", "negation"
        else:
            a, b = (f"On January 1, 2026, {name} was {age} years old.",
                    f"On January 1, 2026, {name} was {age + rng.randrange(1, 9)} years old.")
            relation, attribute, transform = "numeric", "age", "number_swap"
    elif label == "equivalent":
        meta["rationale"] = "Paraphrase with same fact, value, and effective time."
        if mode == 0:
            a, b = (f"In 2026, {name} bought a {item}.", f"In 2026, {name} purchased a {item}.")
            relation, attribute, transform = "purchase", "purchase_item", "synonym"
        elif mode == 1:
            a, b = (f"At noon, {name}'s office door at the {city} branch was shut.",
                    f"At noon, {name}'s office door at the {city} branch was closed.")
            relation, attribute, transform = "state", "office_door_state", "synonym"
        elif mode == 2:
            a, b = (f"In 2026, {name} resides in {city}.", f"In 2026, {name} lives in {city}.")
            relation, attribute, transform = "residence", "residence", "paraphrase"
        else:
            a, b = (f"As of January 1, 2026, {name} is {age} years old.",
                    f"{name}'s age on January 1, 2026 is {age}.")
            relation, attribute, transform = "numeric", "age", "paraphrase"
    elif label == "compatible":
        meta.update(same_fact=False, cardinality="multiple", rationale="Assertions can both hold simultaneously.")
        if mode == 0:
            a, b = (f"In 2026, {name} lives in {city}.", f"In 2026, {name} works in {city2}.")
            relation, attribute, transform = "residence_work", "different_attributes", "attribute_swap"
        elif mode == 1:
            a, b = (f"In 2026, {name} owns a {item}.", f"In 2026, {name} owns a {item2}.")
            relation, attribute, transform = "ownership", "owned_items", "additional_value"
            meta["same_fact"] = True
        elif mode == 2:
            a, b = (f"In 2026, {name} is friends with {other}.",
                    f"In 2026, {name} works with {other}.")
            relation, attribute, transform = "relationship", "different_relationships", "attribute_swap"
        else:
            a, b = (f"At noon, the light in {name}'s {city} office is on.",
                    f"At noon, the window in {name}'s {city} office is open.")
            relation, attribute, transform = "state", "different_objects", "attribute_swap"
    elif label == "temporal_update":
        meta.update(same_valid_time=False, incompatible=True,
                    rationale="Different explicit valid times allow a legitimate change; not a simultaneous contradiction.")
        if mode == 0:
            a, b = (f"In 2024, {name}'s primary home city was {city}.",
                    f"In 2026, {name}'s primary home city was {city2}.")
            relation, attribute, transform = "residence", "primary_home_city", "time_value_swap"
        elif mode == 1:
            a, b = (f"In 2024, {name} worked as a {job}.",
                    f"In 2026, {name} worked as a {job2}.")
            relation, attribute, transform = "occupation", "primary_job", "time_value_swap"
        elif mode == 2:
            a, b = (f"At 9 AM on November 5, {name}'s office door at the {city} branch was open.",
                    f"At 5 PM on November 5, {name}'s office door at the {city} branch was closed.")
            relation, attribute, transform = "state", "office_door_state", "time_antonym"
        else:
            a, b = (f"On January 1, 2024, {name} was {age} years old.",
                    f"On January 1, 2026, {name} was {age + 2} years old.")
            relation, attribute, transform = "numeric", "age", "time_numeric"
    else:  # unrelated
        meta.update(same_fact=False, cardinality="unknown", rationale="Different subjects and/or unrelated attributes.")
        if mode == 0:
            a, b = (f"In 2026, {name} lived in {city}.", f"In 2026, {other} lived in {city2}.")
            relation, attribute, transform = "residence", "different_subjects", "subject_swap"
        elif mode == 1:
            a, b = (f"In 2026, {name} owned a {item}.",
                    f"In 2026, {other} worked as a {job}.")
            relation, attribute, transform = "cross_relation", "unrelated", "subject_attribute_swap"
        elif mode == 2:
            a, b = (f"{name}'s favorite instrument is a {item}.",
                    f"{other}'s primary home city is {city}.")
            relation, attribute, transform = "cross_relation", "unrelated", "subject_attribute_swap"
        else:
            a, b = (f"{name} is {age} years old on January 1, 2026.",
                    f"{other} has a {item} in the garage.")
            relation, attribute, transform = "cross_relation", "unrelated", "subject_attribute_swap"

    return Pair(text_a=a, text_b=b, relation_family=relation, attribute=attribute,
                template_family=f"{label}-{mode}", transformation=transform, **meta)  # type: ignore[arg-type]


def generate_pairs(per_label: int = 600, seed: int = 42) -> list[Pair]:
    if per_label < 1:
        raise ValueError("per_label must be >= 1")
    rng = random.Random(seed)
    pairs: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    for label in LABEL_ORDER:
        count, attempts = 0, 0
        while count < per_label:
            if attempts > per_label * 100:
                raise RuntimeError(f"Exhausted unique synthetic examples for {label}")
            pair = _make(label, attempts, rng)
            attempts += 1
            texts = (pair.text_a, pair.text_b)
            if texts in seen:
                continue
            seen.add(texts)
            # IDs are ordinal, regardless of duplicate-generation retries.
            pair = Pair(**{**pair.__dict__, "id": f"generated-{label}-{count:05d}"})
            pairs.append(pair)
            count += 1
    rng.shuffle(pairs)
    return pairs


def split_subject(subject: str, seed: int = 42) -> str:
    """Deterministic group split: approx. 70/15/15 on known synthetic names.

    This is a subject holdout, *not* a template or relation-family holdout.
    """
    if subject in NAMES:
        names = sorted(NAMES, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).digest())
        position = names.index(subject)
        return "train" if position < 22 else "validation" if position < 27 else "test"
    digest = hashlib.sha256(f"{seed}:{subject}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def split_template(template_family: str, seed: int = 42) -> str:
    """Hold out an entire template mode across every synthetic label class.

    Generated families end with -0 through -3. The map is intentionally
    deterministic and **does not** constitute a relation-family holdout.
    """
    try:
        family, number = template_family.rsplit("-", 1)
        if family and number in {"0", "1", "2", "3"}:
            return {"0": "test", "1": "validation", "2": "train", "3": "train"}[number]
    except ValueError:
        pass
    return split_subject(template_family, seed)


@dataclass(frozen=True)
class Quad:
    base: str
    first: str
    second: str
    both: str
    subject: str
    other_subject: str
    operation_pair: str


def generate_quads(count: int = 100, seed: int = 42) -> list[Quad]:
    """Controlled transformation squares for four families of interactions."""
    if count < 1:
        raise ValueError("count must be >= 1")
    rng = random.Random(seed)
    result: list[Quad] = []
    for i in range(count):
        subject = rng.choice(NAMES)
        other = _different(rng, NAMES, subject)
        item = rng.choice(ITEMS)
        item2 = _different(rng, ITEMS, item)
        stem = f"In 2026, {subject} owns a {item}."
        mode = i % 4
        if mode == 0:
            first = f"In 2026, {subject} does not own a {item}."
            second = f"In 2026, {other} owns a {item}."
            both = f"In 2026, {other} does not own a {item}."
            label = "negation_x_subject"
        elif mode == 1:
            first = f"In 2026, {subject} does not own a {item}."
            second = f"In 2026, {subject} owns a {item2}."
            both = f"In 2026, {subject} does not own a {item2}."
            label = "negation_x_value"
        elif mode == 2:
            first = f"In 2026, {other} owns a {item}."
            second = f"In 2026, {subject} owns a {item2}."
            both = f"In 2026, {other} owns a {item2}."
            label = "subject_x_value"
        else:
            stem = f"In 2024, {subject} owns a {item}."
            first = f"In 2026, {subject} owns a {item}."
            second = f"In 2024, {subject} owns a {item2}."
            both = f"In 2026, {subject} owns a {item2}."
            label = "time_x_value"
        result.append(Quad(
            base=stem, first=first, second=second, both=both,
            subject=subject, other_subject=other, operation_pair=label,
        ))
    return result
