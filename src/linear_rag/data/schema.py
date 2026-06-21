from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

DOC_FIELDS = [
    "person",
    "location",
    "object",
    "color",
    "date",
    "event",
    "organization",
    "numeric_code",
]

QUERY_TYPES = [
    "two-condition",
    "three-condition",
    "four-condition",
    "code-based",
    "organization+event",
]

DIFFICULTIES = ["easy", "medium", "hard", "adversarial"]

HARD_NEG_TYPES = [
    "single-field-overlap",
    "two-field-overlap",
    "near-synonym-rewrite",
    "swapped-entity",
    "high-overlap-distractor",
]


@dataclass
class Doc:
    doc_id: int
    person: str
    location: str
    object: str
    color: str
    date: str
    event: str
    organization: str
    numeric_code: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Query:
    query_id: int
    query_text: str
    gold_doc_id: int
    query_type: str
    difficulty: str
    conditions: dict[str, str]
    n_conditions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
