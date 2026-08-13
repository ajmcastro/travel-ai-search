"""Integration tests for vector (ANN) search.

Tests run against the curated 6-hotel index defined in conftest.py, which has
real sentence-transformer embeddings pre-generated.  Semantic tests assert
*directional* correctness (the best-matching hotel appears in the top-N results)
rather than exact score ordering, which is model- and dataset-dependent.

Index composition (from conftest.py):
  By country:   Spain=3 (001,002,005)  Portugal=1 (004)  Greece=1 (003)  Morocco=1 (006)
  family=True:  001, 004, 006
  adults=True:  002
  price ≤ 700:  001 (£699), 005 (£649), 006 (£449)
  stars = 5:    002
  month=July:   001, 002, 003, 004, 005  (5)
  airport=LGW:  001, 002, 003, 004, 005, 006  (all 6)
  airport=MAN:  001, 003, 004, 005, 006  (not 002)
"""

from __future__ import annotations

import pytest
from opensearchpy import OpenSearch

from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.retrieval.vector import (
    VectorSearchParams,
    VectorSearchResult,
    vector_search,
)

# ── Basic retrieval ───────────────────────────────────────────────────────────


@pytest.mark.integration
def test_vector_search_returns_result_object(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="beach hotel", top_k=6)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert isinstance(result, VectorSearchResult)


@pytest.mark.integration
def test_vector_search_returns_hits(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="relaxing beach holiday", top_k=6)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert result.total > 0
    assert len(result.hits) > 0


@pytest.mark.integration
def test_vector_search_top_k_limits_hits(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="beach hotel", top_k=3)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert len(result.hits) <= 3


@pytest.mark.integration
def test_vector_search_hits_have_scores(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="seaside resort", top_k=6)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert all(hit.score >= 0.0 for hit in result.hits)


@pytest.mark.integration
def test_vector_search_hits_have_ids(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="beach hotel", top_k=6)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert all(hit.id for hit in result.hits)


@pytest.mark.integration
def test_vector_search_embedding_vector_excluded_from_source(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="beach hotel", top_k=3)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    for hit in result.hits:
        assert "embedding_vector" not in hit.source


@pytest.mark.integration
def test_vector_search_took_ms_is_non_negative(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="beach hotel", top_k=3)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert result.took_ms >= 0


# ── Semantic relevance ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_luxury_spa_query_ranks_adults_retreat_highly(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    """'luxury spa adults retreat' should surface test_hotel_002 (La Vie Luxury Adults Retreat)."""
    params = VectorSearchParams(query="luxury spa adults only retreat", top_k=3)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    returned_ids = [hit.id for hit in result.hits]
    assert "test_hotel_002" in returned_ids


@pytest.mark.integration
def test_family_beach_query_finds_family_hotels(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    """Family-oriented query should return at least one of the known family hotels."""
    params = VectorSearchParams(query="family beach holiday with kids club", top_k=3)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    returned_ids = {hit.id for hit in result.hits}
    family_hotel_ids = {"test_hotel_001", "test_hotel_004", "test_hotel_006"}
    assert returned_ids & family_hotel_ids, "No family hotel in top-3 results"


@pytest.mark.integration
def test_cultural_heritage_query_ranks_riad_highly(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    """'traditional cultural heritage medina' should surface test_hotel_006 (Marrakech Riad)."""
    params = VectorSearchParams(query="traditional cultural heritage medina riad", top_k=3)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    returned_ids = [hit.id for hit in result.hits]
    assert "test_hotel_006" in returned_ids


# ── Filter correctness ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_country_filter_restricts_to_spain(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="beach hotel", top_k=6, country="Spain")
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    spain_ids = {"test_hotel_001", "test_hotel_002", "test_hotel_005"}
    assert result.total == 3
    assert all(hit.id in spain_ids for hit in result.hits)


@pytest.mark.integration
def test_family_friendly_filter(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="beach holiday", top_k=6, family_friendly=True)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    family_ids = {"test_hotel_001", "test_hotel_004", "test_hotel_006"}
    assert result.total == 3
    assert all(hit.id in family_ids for hit in result.hits)


@pytest.mark.integration
def test_adults_only_filter(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="relaxing retreat", top_k=6, adults_only=True)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert result.total == 1
    assert result.hits[0].id == "test_hotel_002"


@pytest.mark.integration
def test_max_price_filter(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="hotel", top_k=6, max_price=700.0)
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    budget_ids = {"test_hotel_001", "test_hotel_005", "test_hotel_006"}
    assert result.total == 3
    assert all(hit.id in budget_ids for hit in result.hits)


@pytest.mark.integration
def test_month_filter(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="beach hotel", top_k=6, month="July")
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    july_ids = {
        "test_hotel_001",
        "test_hotel_002",
        "test_hotel_003",
        "test_hotel_004",
        "test_hotel_005",
    }
    assert result.total == 5
    assert all(hit.id in july_ids for hit in result.hits)


@pytest.mark.integration
def test_airport_filter(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    params = VectorSearchParams(query="beach hotel", top_k=6, airport="MAN")
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    man_ids = {
        "test_hotel_001",
        "test_hotel_003",
        "test_hotel_004",
        "test_hotel_005",
        "test_hotel_006",
    }
    assert result.total == 5
    assert all(hit.id in man_ids for hit in result.hits)


@pytest.mark.integration
def test_combined_filters(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> None:
    """Spain + family_friendly → hotels 001 only (002 is adults_only, 005 is not family)."""
    params = VectorSearchParams(
        query="beach family resort",
        top_k=6,
        country="Spain",
        family_friendly=True,
    )
    result = vector_search(opensearch_client, embedding_provider, params, index=vector_test_index)
    assert result.total == 1
    assert result.hits[0].id == "test_hotel_001"
