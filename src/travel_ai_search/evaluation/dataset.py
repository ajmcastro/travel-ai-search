"""Golden relevance dataset — domain model and loader.

A GoldenDataset is the ground truth used to evaluate retrieval quality.
Each GoldenQuery holds a natural-language query, an optional set of
structured filters, and a list of RelevanceJudgments (doc_id → grade 0–3).

File format: JSON Lines — one GoldenQuery per line.

Example line:
    {
      "query_id": "q001",
      "query_text": "family beach holiday Tenerife",
      "query_class": "destination_family",
      "filters": {},
      "judgments": [
        {"doc_id": "hotel_000042", "grade": 3},
        {"doc_id": "hotel_000017", "grade": 2}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RelevanceJudgment:
    """Grade for a single document relative to a query.

    grade 0 = irrelevant (usually omitted from the file)
    grade 1 = marginally relevant
    grade 2 = relevant
    grade 3 = highly relevant
    """

    doc_id: str
    grade: int


@dataclass(frozen=True)
class GoldenQuery:
    """One query in the golden evaluation set."""

    query_id: str
    query_text: str
    query_class: str
    judgments: tuple[RelevanceJudgment, ...]
    # Structured filters to pass to the retrieval function alongside the text query.
    # Mirrors the LexicalSearchParams optional fields; future strategies may ignore them.
    filters: dict[str, Any] = field(default_factory=dict)

    @property
    def relevant_ids(self) -> frozenset[str]:
        """Doc IDs with grade ≥ 1 (binary relevant set)."""
        return frozenset(j.doc_id for j in self.judgments if j.grade >= 1)

    @property
    def graded_relevance(self) -> dict[str, int]:
        """Mapping from doc_id to grade — used by DCG / NDCG."""
        return {j.doc_id: j.grade for j in self.judgments if j.grade > 0}

    @property
    def has_relevant(self) -> bool:
        """True if at least one judgment has grade ≥ 1."""
        return any(j.grade >= 1 for j in self.judgments)


@dataclass(frozen=True)
class GoldenDataset:
    """Collection of golden queries."""

    queries: tuple[GoldenQuery, ...]

    def by_class(self) -> dict[str, list[GoldenQuery]]:
        """Group queries by query_class."""
        groups: dict[str, list[GoldenQuery]] = {}
        for q in self.queries:
            groups.setdefault(q.query_class, []).append(q)
        return groups

    def __len__(self) -> int:
        return len(self.queries)


def load_dataset(path: Path) -> GoldenDataset:
    """Load a GoldenDataset from a JSON Lines file."""
    queries: list[GoldenQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data: dict[str, Any] = json.loads(line)
        judgments = tuple(
            RelevanceJudgment(doc_id=j["doc_id"], grade=int(j["grade"]))
            for j in data.get("judgments", [])
        )
        queries.append(
            GoldenQuery(
                query_id=data["query_id"],
                query_text=data["query_text"],
                query_class=data["query_class"],
                judgments=judgments,
                filters=data.get("filters", {}),
            )
        )
    return GoldenDataset(queries=tuple(queries))
