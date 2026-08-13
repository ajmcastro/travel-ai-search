"""Unit tests for the golden dataset domain model and loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from travel_ai_search.evaluation.dataset import (
    GoldenDataset,
    GoldenQuery,
    RelevanceJudgment,
    load_dataset,
)

# ── RelevanceJudgment ─────────────────────────────────────────────────────────


def test_relevance_judgment_is_frozen() -> None:
    j = RelevanceJudgment(doc_id="hotel_001", grade=3)
    with pytest.raises(Exception):
        j.grade = 2  # type: ignore[misc]


# ── GoldenQuery properties ────────────────────────────────────────────────────


def _make_query(judgments: list[tuple[str, int]]) -> GoldenQuery:
    return GoldenQuery(
        query_id="q001",
        query_text="test query",
        query_class="test",
        judgments=tuple(RelevanceJudgment(doc_id=d, grade=g) for d, g in judgments),
    )


def test_relevant_ids_excludes_grade_zero() -> None:
    q = _make_query([("a", 3), ("b", 2), ("c", 1), ("d", 0)])
    assert q.relevant_ids == frozenset({"a", "b", "c"})


def test_relevant_ids_empty_when_all_zero() -> None:
    q = _make_query([("a", 0), ("b", 0)])
    assert q.relevant_ids == frozenset()


def test_graded_relevance_excludes_grade_zero() -> None:
    q = _make_query([("a", 3), ("b", 0), ("c", 2)])
    assert q.graded_relevance == {"a": 3, "c": 2}


def test_graded_relevance_empty_when_all_zero() -> None:
    q = _make_query([("a", 0)])
    assert q.graded_relevance == {}


def test_has_relevant_true() -> None:
    q = _make_query([("a", 1), ("b", 0)])
    assert q.has_relevant is True


def test_has_relevant_false_all_zero() -> None:
    q = _make_query([("a", 0), ("b", 0)])
    assert q.has_relevant is False


def test_has_relevant_false_empty_judgments() -> None:
    q = _make_query([])
    assert q.has_relevant is False


def test_filters_default_to_empty_dict() -> None:
    q = _make_query([("a", 3)])
    assert q.filters == {}


def test_filters_can_be_set() -> None:
    q = GoldenQuery(
        query_id="q002",
        query_text="family beach",
        query_class="family",
        judgments=(RelevanceJudgment(doc_id="h1", grade=3),),
        filters={"family_friendly": True},
    )
    assert q.filters == {"family_friendly": True}


def test_golden_query_is_frozen() -> None:
    q = _make_query([("a", 3)])
    with pytest.raises(Exception):
        q.query_text = "changed"  # type: ignore[misc]


# ── GoldenDataset ─────────────────────────────────────────────────────────────


def _make_dataset(*classes: tuple[str, str]) -> GoldenDataset:
    """Build a minimal dataset with one judgment per query."""
    queries = []
    for i, (qid, cls) in enumerate(classes):
        queries.append(
            GoldenQuery(
                query_id=qid,
                query_text=f"query {i}",
                query_class=cls,
                judgments=(RelevanceJudgment(doc_id="h1", grade=3),),
            )
        )
    return GoldenDataset(queries=tuple(queries))


def test_dataset_len() -> None:
    ds = _make_dataset(("q1", "family"), ("q2", "luxury"))
    assert len(ds) == 2


def test_dataset_by_class_groups_correctly() -> None:
    ds = _make_dataset(("q1", "family"), ("q2", "luxury"), ("q3", "family"))
    groups = ds.by_class()
    assert set(groups.keys()) == {"family", "luxury"}
    assert len(groups["family"]) == 2
    assert len(groups["luxury"]) == 1


def test_dataset_by_class_single_query_per_class() -> None:
    ds = _make_dataset(("q1", "a"), ("q2", "b"), ("q3", "c"))
    groups = ds.by_class()
    assert all(len(v) == 1 for v in groups.values())


# ── load_dataset ──────────────────────────────────────────────────────────────


def _write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "golden.jsonl"
    p.write_text(
        "\n".join(json.dumps(r) for r in records),
        encoding="utf-8",
    )
    return p


def test_load_dataset_basic(tmp_path: Path) -> None:
    records = [
        {
            "query_id": "q001",
            "query_text": "family beach",
            "query_class": "family",
            "filters": {},
            "judgments": [
                {"doc_id": "hotel_001", "grade": 3},
                {"doc_id": "hotel_002", "grade": 1},
            ],
        }
    ]
    ds = load_dataset(_write_jsonl(tmp_path, records))
    assert len(ds) == 1
    q = ds.queries[0]
    assert q.query_id == "q001"
    assert q.query_text == "family beach"
    assert q.query_class == "family"
    assert q.relevant_ids == frozenset({"hotel_001", "hotel_002"})
    assert q.graded_relevance == {"hotel_001": 3, "hotel_002": 1}


def test_load_dataset_multiple_queries(tmp_path: Path) -> None:
    records = [
        {
            "query_id": f"q{i:03d}",
            "query_text": f"query {i}",
            "query_class": "test",
            "filters": {},
            "judgments": [{"doc_id": "h1", "grade": 2}],
        }
        for i in range(5)
    ]
    ds = load_dataset(_write_jsonl(tmp_path, records))
    assert len(ds) == 5


def test_load_dataset_preserves_filters(tmp_path: Path) -> None:
    records = [
        {
            "query_id": "q001",
            "query_text": "family resort",
            "query_class": "family",
            "filters": {"family_friendly": True, "max_price": 1000.0},
            "judgments": [{"doc_id": "h1", "grade": 3}],
        }
    ]
    ds = load_dataset(_write_jsonl(tmp_path, records))
    assert ds.queries[0].filters == {"family_friendly": True, "max_price": 1000.0}


def test_load_dataset_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    ds = load_dataset(p)
    assert len(ds) == 0


def test_load_dataset_skips_blank_lines(tmp_path: Path) -> None:
    row1 = json.dumps(
        {
            "query_id": "q1",
            "query_text": "a",
            "query_class": "c",
            "filters": {},
            "judgments": [{"doc_id": "h1", "grade": 2}],
        }
    )
    row2 = json.dumps(
        {
            "query_id": "q2",
            "query_text": "b",
            "query_class": "c",
            "filters": {},
            "judgments": [{"doc_id": "h2", "grade": 1}],
        }
    )
    p = tmp_path / "blanks.jsonl"
    p.write_text(f"{row1}\n\n{row2}\n", encoding="utf-8")
    ds = load_dataset(p)
    assert len(ds) == 2


def test_load_dataset_missing_filters_defaults_to_empty(tmp_path: Path) -> None:
    records = [
        {
            "query_id": "q1",
            "query_text": "test",
            "query_class": "test",
            "judgments": [{"doc_id": "h1", "grade": 3}],
        }
    ]
    ds = load_dataset(_write_jsonl(tmp_path, records))
    assert ds.queries[0].filters == {}
