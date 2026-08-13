"""Integration smoke tests for the evaluation framework.

Runs the evaluator against the curated 6-hotel test index using a hand-crafted
mini golden dataset whose expected outcomes are known from the indexed data.

These tests verify end-to-end plumbing (search_fn → evaluator → report),
not metric arithmetic (unit-tested in test_metrics.py).
"""

from __future__ import annotations

from typing import Any

import pytest
from opensearchpy import OpenSearch

from travel_ai_search.evaluation.dataset import (
    GoldenDataset,
    GoldenQuery,
    RelevanceJudgment,
)
from travel_ai_search.evaluation.evaluator import EvaluationReport, evaluate
from travel_ai_search.retrieval.lexical import LexicalSearchParams, lexical_search

# ── Mini golden dataset ───────────────────────────────────────────────────────
# Hotel IDs in the curated index:
#   test_hotel_001  Sunny Beach Family Resort  — Spain, family, £699
#   test_hotel_002  La Vie Luxury Adults Retreat — Spain, adults-only, 5★, £2499
#   test_hotel_003  Mykonos Party and Beach Hotel — Greece, nightlife
#   test_hotel_004  Algarve Ocean Family Resort — Portugal, family, £849
#   test_hotel_005  Menorca Quiet Retreat — Spain, peaceful, boutique, £649
#   test_hotel_006  Marrakech Heritage Riad — Morocco, family, culture, £449

_MINI_DATASET = GoldenDataset(
    queries=tuple(
        [
            GoldenQuery(
                query_id="mq001",
                query_text="family beach resort",
                query_class="family",
                judgments=(
                    RelevanceJudgment(doc_id="test_hotel_001", grade=3),
                    RelevanceJudgment(doc_id="test_hotel_004", grade=3),
                    RelevanceJudgment(doc_id="test_hotel_006", grade=1),
                ),
            ),
            GoldenQuery(
                query_id="mq002",
                query_text="luxury adults retreat spa",
                query_class="adults_couples",
                judgments=(RelevanceJudgment(doc_id="test_hotel_002", grade=3),),
            ),
            GoldenQuery(
                query_id="mq003",
                query_text="peaceful boutique quiet nature",
                query_class="quiet_peaceful",
                judgments=(RelevanceJudgment(doc_id="test_hotel_005", grade=3),),
            ),
        ]
    )
)


# ── Helper ────────────────────────────────────────────────────────────────────


def _make_search_fn(client: OpenSearch, index: str):  # type: ignore[no-untyped-def]
    def search(query_text: str, top_k: int, filters: dict[str, Any]) -> list[str]:
        params = LexicalSearchParams(query=query_text, top_k=top_k, **filters)
        result = lexical_search(client, params, index=index)
        return [hit.id for hit in result.hits]

    return search


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_evaluate_returns_report(opensearch_client: OpenSearch, lexical_test_index: str) -> None:
    search_fn = _make_search_fn(opensearch_client, lexical_test_index)
    report = evaluate(search_fn, _MINI_DATASET, strategy="bm25", k=6)
    assert isinstance(report, EvaluationReport)
    assert report.strategy == "bm25"
    assert report.k == 6


@pytest.mark.integration
def test_evaluate_covers_all_queries(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    search_fn = _make_search_fn(opensearch_client, lexical_test_index)
    report = evaluate(search_fn, _MINI_DATASET, strategy="bm25", k=6)
    assert report.n_queries == 3


@pytest.mark.integration
def test_evaluate_metrics_bounded(opensearch_client: OpenSearch, lexical_test_index: str) -> None:
    search_fn = _make_search_fn(opensearch_client, lexical_test_index)
    report = evaluate(search_fn, _MINI_DATASET, strategy="bm25", k=6)

    for metric in (
        report.overall_precision,
        report.overall_recall,
        report.overall_hit_rate,
        report.mrr,
        report.map_score,
        report.overall_ndcg,
    ):
        assert 0.0 <= metric <= 1.0, f"Metric out of [0,1]: {metric}"


@pytest.mark.integration
def test_evaluate_luxury_query_hits_correct_hotel(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    """BM25 should return test_hotel_002 for 'luxury adults retreat spa'."""
    search_fn = _make_search_fn(opensearch_client, lexical_test_index)
    report = evaluate(search_fn, _MINI_DATASET, strategy="bm25", k=6)

    luxury_query = next(q for q in report.per_query if q.query_id == "mq002")
    # With only one grade-3 doc and a highly specific query, hit rate must be 1.0
    assert luxury_query.hit_rate == pytest.approx(1.0), (
        "BM25 failed to retrieve test_hotel_002 for 'luxury adults retreat spa'"
    )


@pytest.mark.integration
def test_evaluate_family_query_positive_ndcg(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    """At least one family hotel should rank in top-6 for 'family beach resort'."""
    search_fn = _make_search_fn(opensearch_client, lexical_test_index)
    report = evaluate(search_fn, _MINI_DATASET, strategy="bm25", k=6)

    family_query = next(q for q in report.per_query if q.query_id == "mq001")
    assert family_query.ndcg > 0.0, "No family hotel ranked in top-6 for 'family beach resort'"


@pytest.mark.integration
def test_evaluate_by_class_groups_match_mini_dataset(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    search_fn = _make_search_fn(opensearch_client, lexical_test_index)
    report = evaluate(search_fn, _MINI_DATASET, strategy="bm25", k=6)

    classes = {s.query_class for s in report.by_class()}
    assert classes == {"family", "adults_couples", "quiet_peaceful"}


@pytest.mark.integration
def test_evaluate_to_dict_structure(opensearch_client: OpenSearch, lexical_test_index: str) -> None:
    search_fn = _make_search_fn(opensearch_client, lexical_test_index)
    report = evaluate(search_fn, _MINI_DATASET, strategy="bm25", k=6)
    d = report.to_dict()

    assert d["strategy"] == "bm25"
    assert d["k"] == 6
    assert d["n_queries"] == 3
    assert "overall" in d
    assert "by_class" in d
    assert "per_query" in d
    assert len(d["per_query"]) == 3


@pytest.mark.integration
def test_evaluate_latency_recorded(opensearch_client: OpenSearch, lexical_test_index: str) -> None:
    search_fn = _make_search_fn(opensearch_client, lexical_test_index)
    report = evaluate(search_fn, _MINI_DATASET, strategy="bm25", k=6)
    assert report.latency_p50_ms >= 0
    assert report.latency_p95_ms >= report.latency_p50_ms
