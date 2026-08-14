"""Integration tests for hybrid (BM25 + vector) search.

Tests run against the curated 6-hotel index defined in conftest.py, which has
real sentence-transformer embeddings pre-generated.

Index composition (from conftest.py):
  By country:   Spain=3 (001,002,005)  Portugal=1 (004)  Greece=1 (003)  Morocco=1 (006)
  family=True:  001, 004, 006  (3)
  adults=True:  002            (1)
  price ≤ 700:  001 (£699), 005 (£649), 006 (£449)  (3)
  stars = 5:    002            (1)
  month=July:   001, 002, 003, 004, 005  (5)
  airport=LGW:  all 6
  airport=MAN:  001, 003, 004, 005, 006  (5)
"""

from __future__ import annotations

import pytest
from opensearchpy import OpenSearch

from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.retrieval.hybrid import (
    HybridSearchParams,
    HybridSearchResult,
    hybrid_search,
)

# ── Basic retrieval ───────────────────────────────────────────────────────────


@pytest.mark.integration
def test_hybrid_search_returns_result_object(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(query="beach hotel", top_k=6, candidate_k=6)
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert isinstance(result, HybridSearchResult)


@pytest.mark.integration
def test_hybrid_search_returns_hits(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(query="relaxing beach holiday", top_k=6, candidate_k=6)
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert len(result.hits) > 0


@pytest.mark.integration
def test_hybrid_search_top_k_limits_hits(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(query="beach resort", top_k=3, candidate_k=6)
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert len(result.hits) <= 3


@pytest.mark.integration
def test_hybrid_search_hits_are_sorted_descending(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(query="luxury spa adults", top_k=6, candidate_k=6)
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
def test_hybrid_search_timing_fields_present(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(query="family resort", top_k=5, candidate_k=6)
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert result.took_ms >= 0
    assert result.lexical_took_ms >= 0
    assert result.vector_took_ms >= 0


@pytest.mark.integration
def test_hybrid_search_total_is_pool_size(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    # With candidate_k=6 and 6 docs in the index, pool_size ≤ 6.
    params = HybridSearchParams(query="hotel", top_k=3, candidate_k=6)
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert result.total >= len(result.hits)


# ── Filters ───────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_hybrid_filter_by_country_spain(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(query="beach hotel", top_k=6, candidate_k=6, country="Spain")
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    for hit in result.hits:
        assert hit.source.get("country") == "Spain"


@pytest.mark.integration
def test_hybrid_filter_family_friendly(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(
        query="family holiday", top_k=6, candidate_k=6, family_friendly=True
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    # 001, 004, 006 are family_friendly — expect all 3 results
    assert len(result.hits) == 3
    for hit in result.hits:
        assert hit.source.get("family_friendly") is True


@pytest.mark.integration
def test_hybrid_filter_adults_only(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(query="adults retreat", top_k=6, candidate_k=6, adults_only=True)
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    # Only 002 is adults_only
    assert len(result.hits) == 1
    assert result.hits[0].id == "test_hotel_002"


@pytest.mark.integration
def test_hybrid_filter_max_price(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(query="beach holiday", top_k=6, candidate_k=6, max_price=700.0)
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    # 001 (£699), 005 (£649), 006 (£449) — expect 3
    assert len(result.hits) == 3
    for hit in result.hits:
        assert hit.source.get("price_per_person_gbp") <= 700.0


@pytest.mark.integration
def test_hybrid_filter_by_month(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(query="summer holiday", top_k=6, candidate_k=6, month="July")
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    # 001, 002, 003, 004, 005 have July — 5 hotels
    assert len(result.hits) == 5
    for hit in result.hits:
        assert "July" in hit.source.get("available_months", [])


@pytest.mark.integration
def test_hybrid_filter_by_airport(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(query="beach resort", top_k=6, candidate_k=6, airport="MAN")
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    # 001, 003, 004, 005, 006 have MAN — 5 hotels
    assert len(result.hits) == 5
    for hit in result.hits:
        assert "MAN" in hit.source.get("available_departure_airports", [])


# ── Weight sensitivity ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_hybrid_lexical_weight_only_returns_results(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(
        query="luxury spa retreat",
        top_k=6,
        candidate_k=6,
        lexical_weight=1.0,
        vector_weight=0.0,
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert len(result.hits) > 0


@pytest.mark.integration
def test_hybrid_vector_weight_only_returns_results(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = HybridSearchParams(
        query="peaceful countryside retreat",
        top_k=6,
        candidate_k=6,
        lexical_weight=0.0,
        vector_weight=1.0,
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert len(result.hits) > 0


# ── Semantic quality ──────────────────────────────────────────────────────────


@pytest.mark.integration
def test_hybrid_finds_adults_only_luxury_hotel_for_romantic_query(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    # "La Vie Luxury Adults Retreat" (002) should surface for a romantic luxury query.
    params = HybridSearchParams(
        query="exclusive romantic adults-only resort with infinity pool and spa",
        top_k=3,
        candidate_k=6,
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    ids = [h.id for h in result.hits]
    assert "test_hotel_002" in ids


@pytest.mark.integration
def test_hybrid_finds_family_resort_for_family_query(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    # Family beach resorts should include 001 or 004.
    params = HybridSearchParams(
        query="family holiday with kids club and waterslides",
        top_k=3,
        candidate_k=6,
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    ids = {h.id for h in result.hits}
    assert ids & {"test_hotel_001", "test_hotel_004"}


# ── RRF fusion ────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_rrf_search_returns_result_object(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    from travel_ai_search.retrieval.fusion import FusionMethod

    params = HybridSearchParams(
        query="beach hotel", top_k=6, candidate_k=6, fusion=FusionMethod.rrf
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert isinstance(result, HybridSearchResult)


@pytest.mark.integration
def test_rrf_search_returns_hits(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    from travel_ai_search.retrieval.fusion import FusionMethod

    params = HybridSearchParams(
        query="relaxing beach holiday", top_k=6, candidate_k=6, fusion=FusionMethod.rrf
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert len(result.hits) > 0


@pytest.mark.integration
def test_rrf_search_hits_sorted_descending(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    from travel_ai_search.retrieval.fusion import FusionMethod

    params = HybridSearchParams(query="luxury spa", top_k=6, candidate_k=6, fusion=FusionMethod.rrf)
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
def test_rrf_filter_family_friendly(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    from travel_ai_search.retrieval.fusion import FusionMethod

    params = HybridSearchParams(
        query="family holiday",
        top_k=6,
        candidate_k=6,
        fusion=FusionMethod.rrf,
        family_friendly=True,
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert len(result.hits) == 3
    for hit in result.hits:
        assert hit.source.get("family_friendly") is True


@pytest.mark.integration
def test_rrf_filter_adults_only(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    from travel_ai_search.retrieval.fusion import FusionMethod

    params = HybridSearchParams(
        query="adults retreat",
        top_k=6,
        candidate_k=6,
        fusion=FusionMethod.rrf,
        adults_only=True,
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert len(result.hits) == 1
    assert result.hits[0].id == "test_hotel_002"


@pytest.mark.integration
def test_rrf_filter_by_month(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    from travel_ai_search.retrieval.fusion import FusionMethod

    params = HybridSearchParams(
        query="summer holiday",
        top_k=6,
        candidate_k=6,
        fusion=FusionMethod.rrf,
        month="July",
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert len(result.hits) == 5
    for hit in result.hits:
        assert "July" in hit.source.get("available_months", [])


@pytest.mark.integration
def test_rrf_finds_adults_luxury_hotel(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    from travel_ai_search.retrieval.fusion import FusionMethod

    params = HybridSearchParams(
        query="exclusive romantic adults-only resort with infinity pool and spa",
        top_k=3,
        candidate_k=6,
        fusion=FusionMethod.rrf,
    )
    result = hybrid_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    ids = [h.id for h in result.hits]
    assert "test_hotel_002" in ids
