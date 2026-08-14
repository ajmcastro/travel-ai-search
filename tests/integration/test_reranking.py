"""Integration tests for cross-encoder reranking.

These tests load the real LocalCrossEncoderReranker model (~86 MB) and run
it against a small curated set of Hit objects to verify that reranking
actually reorders results in semantically meaningful ways.

Requires: the sentence-transformers package and network access on first run
(model is cached locally afterward via Hugging Face hub).
"""

from __future__ import annotations

import pytest

from travel_ai_search.reranking.local import LocalCrossEncoderReranker
from travel_ai_search.retrieval.types import Hit

pytestmark = pytest.mark.integration


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def reranker() -> LocalCrossEncoderReranker:
    """Load the cross-encoder once for all tests in this module."""
    return LocalCrossEncoderReranker()


def _hit(id: str, score: float, source: dict) -> Hit:  # type: ignore[type-arg]
    return Hit(id=id, score=score, source=source)


_ADULTS_SPA = _hit(
    "adults_spa",
    score=0.5,
    source={
        "hotel_name": "La Dolce Vita Adults Spa Resort",
        "destination": "Marbella",
        "country": "Spain",
        "hotel_description": (
            "Exclusive adults-only sanctuary with infinity pools, Michelin-star dining, "
            "and a world-class thalassotherapy spa. Minimum age 16."
        ),
        "activities": ["spa", "yoga", "wine tasting", "golf"],
        "tags": ["luxury", "adults_only", "spa", "romantic"],
    },
)

_FAMILY_BEACH = _hit(
    "family_beach",
    score=0.9,  # higher fused score — reranker should reassign correctly
    source={
        "hotel_name": "Sunny Shores Family Resort",
        "destination": "Benidorm",
        "country": "Spain",
        "hotel_description": (
            "Fun-filled family resort with dedicated kids club, waterslides, "
            "shallow beach pools, and nightly entertainment for all ages."
        ),
        "activities": ["kids club", "waterslides", "beach volleyball", "swimming"],
        "tags": ["family", "beach", "pool", "kids_friendly"],
    },
)

_BUDGET_CITY = _hit(
    "budget_city",
    score=0.3,
    source={
        "hotel_name": "Central City Inn",
        "destination": "Madrid",
        "country": "Spain",
        "hotel_description": (
            "Budget-friendly city centre hotel near public transport. Basic rooms, no pool, no spa."
        ),
        "activities": ["sightseeing", "city tours"],
        "tags": ["budget", "city"],
    },
)

_MOUNTAIN_ADVENTURE = _hit(
    "mountain_adventure",
    score=0.4,
    source={
        "hotel_name": "Summit Trek Lodge",
        "destination": "Sierra Nevada",
        "country": "Spain",
        "hotel_description": (
            "Rustic mountain lodge for hikers and climbers. "
            "No spa. Family-friendly with communal dining."
        ),
        "activities": ["hiking", "climbing", "mountain biking", "kayaking"],
        "tags": ["adventure", "nature", "family"],
    },
)

_KIDS_CLUB_RESORT = _hit(
    "kids_club",
    score=0.6,
    source={
        "hotel_name": "Happy Family Club Med Style",
        "destination": "Lanzarote",
        "country": "Spain",
        "hotel_description": (
            "All-inclusive family resort with extensive kids club, "
            "children's pool, and family activity programme."
        ),
        "activities": ["kids club", "family shows", "snorkelling", "tennis"],
        "tags": ["family", "all_inclusive", "kids_friendly", "beach"],
    },
)


# ── Semantics: adults luxury spa query ────────────────────────────────────────


def test_adults_spa_query_ranks_spa_hotel_first(reranker: LocalCrossEncoderReranker) -> None:
    """Adults luxury spa query should rank the adults-only spa hotel highest."""
    hits = [_FAMILY_BEACH, _ADULTS_SPA, _BUDGET_CITY]
    result = reranker.rerank("adults only luxury spa hotel", hits, top_k=3)
    assert result[0].id == "adults_spa"


def test_adults_spa_query_ranks_spa_above_non_spa(reranker: LocalCrossEncoderReranker) -> None:
    """Adults spa hotel must outscore hotels with no spa or adults-only features."""
    hits = [_FAMILY_BEACH, _ADULTS_SPA, _BUDGET_CITY]
    result = reranker.rerank("adults only luxury spa hotel", hits, top_k=3)
    adults_rank = next(i for i, h in enumerate(result) if h.id == "adults_spa")
    # Any non-spa hotel must rank below the spa hotel
    for i, h in enumerate(result):
        if h.id != "adults_spa":
            assert adults_rank < i


# ── Semantics: family beach kids query ────────────────────────────────────────


def test_family_query_ranks_family_hotels_above_adults_only(
    reranker: LocalCrossEncoderReranker,
) -> None:
    """Family beach holiday query: adults-only spa should rank below family hotels."""
    hits = [_ADULTS_SPA, _FAMILY_BEACH, _KIDS_CLUB_RESORT]
    result = reranker.rerank("family beach holiday with kids", hits, top_k=3)
    adults_rank = next(i for i, h in enumerate(result) if h.id == "adults_spa")
    family_rank = next(i for i, h in enumerate(result) if h.id == "family_beach")
    assert adults_rank > family_rank


def test_family_query_top_result_is_family_hotel(reranker: LocalCrossEncoderReranker) -> None:
    hits = [_ADULTS_SPA, _FAMILY_BEACH, _KIDS_CLUB_RESORT, _BUDGET_CITY]
    result = reranker.rerank("family beach holiday with children waterslides", hits, top_k=4)
    assert result[0].id in {"family_beach", "kids_club"}


# ── Semantics: adventure hiking query ─────────────────────────────────────────


def test_hiking_query_ranks_mountain_lodge_first(reranker: LocalCrossEncoderReranker) -> None:
    hits = [_ADULTS_SPA, _FAMILY_BEACH, _MOUNTAIN_ADVENTURE, _BUDGET_CITY]
    result = reranker.rerank("hiking mountain adventure lodge", hits, top_k=4)
    assert result[0].id == "mountain_adventure"


# ── Output shape and contract ─────────────────────────────────────────────────


def test_reranker_returns_same_count_as_input_when_top_k_exceeds_hits(
    reranker: LocalCrossEncoderReranker,
) -> None:
    hits = [_ADULTS_SPA, _FAMILY_BEACH]
    result = reranker.rerank("spa", hits, top_k=100)
    assert len(result) == 2


def test_reranker_scores_are_floats(reranker: LocalCrossEncoderReranker) -> None:
    hits = [_ADULTS_SPA, _FAMILY_BEACH]
    result = reranker.rerank("luxury spa", hits, top_k=2)
    for h in result:
        assert isinstance(h.score, float)


def test_reranker_does_not_mutate_original_hits(reranker: LocalCrossEncoderReranker) -> None:
    """Reranker must return new Hit objects, not modify the originals in place."""
    original_score = _FAMILY_BEACH.score
    reranker.rerank("spa", [_FAMILY_BEACH, _ADULTS_SPA], top_k=2)
    assert _FAMILY_BEACH.score == original_score


# ── End-to-end: hybrid + reranking ───────────────────────────────────────────


def test_hybrid_search_with_reranking_end_to_end(
    opensearch_client,
    embedding_provider,  # type: ignore[no-untyped-def]
) -> None:
    """End-to-end: hybrid RRF search followed by cross-encoder reranking."""
    from travel_ai_search.retrieval.fusion import FusionMethod
    from travel_ai_search.retrieval.hybrid import HybridSearchParams, hybrid_search

    live_reranker = LocalCrossEncoderReranker()
    params = HybridSearchParams(
        query="adults only luxury spa",
        fusion=FusionMethod.rrf,
        top_k=5,
        rerank_k=20,
        adults_only=True,
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, reranker=live_reranker)
    assert result.total >= 0
    assert result.reranking_took_ms > 0
    for hit in result.hits:
        assert hit.source.get("adults_only") is True


def test_hybrid_search_reranking_took_ms_zero_without_reranker(
    opensearch_client,
    embedding_provider,  # type: ignore[no-untyped-def]
) -> None:
    from travel_ai_search.retrieval.fusion import FusionMethod
    from travel_ai_search.retrieval.hybrid import HybridSearchParams, hybrid_search

    params = HybridSearchParams(query="luxury spa", fusion=FusionMethod.rrf, top_k=5)
    result = hybrid_search(opensearch_client, embedding_provider, params, reranker=None)
    assert result.reranking_took_ms == 0
