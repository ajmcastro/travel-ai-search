"""Integration tests for BM25 lexical search.

Tests execute against the curated 6-hotel index defined in conftest.py.
Each test is designed around the known dataset properties, not BM25 score
order — exact ranking is left to the evaluation framework (Milestone 4).

Index composition (from conftest.py):
  By country:   Spain=3 (001,002,005)  Portugal=1 (004)  Greece=1 (003)  Morocco=1 (006)
  family=True:  001, 004, 006
  adults=True:  002
  price ≤ 700:  001 (£699), 005 (£649), 006 (£449)
  stars = 5:    002
  month=July:   001, 002, 003, 004, 005
  airport=MAN:  001, 003, 004, 005, 006
"""

from __future__ import annotations

import pytest
from opensearchpy import OpenSearch

from travel_ai_search.retrieval.lexical import (
    LexicalSearchParams,
    LexicalSearchResult,
    lexical_search,
)

# ── Basic retrieval ───────────────────────────────────────────────────────────


@pytest.mark.integration
def test_search_returns_hits(opensearch_client: OpenSearch, lexical_test_index: str) -> None:
    params = LexicalSearchParams(query="beach hotel")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert isinstance(result, LexicalSearchResult)
    assert result.total > 0
    assert len(result.hits) > 0


@pytest.mark.integration
def test_search_total_matches_all_hotels(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="pool", top_k=100)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    # "pool" appears in the amenities of all 6 curated hotels
    assert result.total == 6


@pytest.mark.integration
def test_search_empty_query_returns_all(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert result.total == 6


@pytest.mark.integration
def test_search_top_k_limits_hits(opensearch_client: OpenSearch, lexical_test_index: str) -> None:
    params = LexicalSearchParams(query="hotel", top_k=2)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert len(result.hits) == 2


@pytest.mark.integration
def test_search_hit_has_positive_score(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="beach")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    for hit in result.hits:
        assert hit.score > 0


@pytest.mark.integration
def test_search_took_ms_is_non_negative(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="hotel")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert result.took_ms >= 0


@pytest.mark.integration
def test_search_hit_source_contains_hotel_name(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="beach")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    for hit in result.hits:
        assert "hotel_name" in hit.source


# ── Filter: country ───────────────────────────────────────────────────────────


@pytest.mark.integration
def test_country_filter_returns_correct_count(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, country="Spain")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert result.total == 3


@pytest.mark.integration
def test_country_filter_excludes_other_countries(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, country="Spain")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    for hit in result.hits:
        assert hit.source["country"] == "Spain"


@pytest.mark.integration
def test_country_filter_portugal_returns_one(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, country="Portugal")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert result.total == 1


# ── Filter: family / adults ───────────────────────────────────────────────────


@pytest.mark.integration
def test_family_filter_returns_correct_count(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, family_friendly=True)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert result.total == 3


@pytest.mark.integration
def test_family_filter_excludes_adults_only_hotels(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, family_friendly=True)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    for hit in result.hits:
        assert hit.source["family_friendly"] is True
        assert hit.source["adults_only"] is False


@pytest.mark.integration
def test_adults_only_filter_returns_one(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, adults_only=True)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert result.total == 1
    assert result.hits[0].id == "test_hotel_002"


# ── Filter: price ─────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_max_price_filter_returns_correct_count(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, max_price=700.0)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    # Hotels 001 (£699), 005 (£649), 006 (£449) are ≤ £700
    assert result.total == 3


@pytest.mark.integration
def test_max_price_filter_excludes_expensive_hotels(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, max_price=700.0)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    for hit in result.hits:
        assert hit.source["price_per_person_gbp"] <= 700.0


# ── Filter: star rating ───────────────────────────────────────────────────────


@pytest.mark.integration
def test_min_star_filter_returns_five_star_only(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, min_star_rating=5)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert result.total == 1
    assert result.hits[0].id == "test_hotel_002"


# ── Filter: month ─────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_month_filter_july_returns_five(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, month="July")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert result.total == 5  # all except hotel_006 (Morocco — Mar/Apr/Oct/Nov)


@pytest.mark.integration
def test_month_filter_march_returns_morocco_only(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, month="March")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert result.total == 1
    assert result.hits[0].id == "test_hotel_006"


# ── Filter: airport ───────────────────────────────────────────────────────────


@pytest.mark.integration
def test_airport_filter_lhr_returns_two(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, airport="LHR")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    # Hotels 002 (LGW,LHR) and 006 (LGW,MAN,LHR)
    assert result.total == 2


# ── Combined filters ──────────────────────────────────────────────────────────


@pytest.mark.integration
def test_country_and_family_filter_combined(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100, country="Spain", family_friendly=True)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    # Only hotel_001 (Benidorm, Spain, family)
    assert result.total == 1
    assert result.hits[0].id == "test_hotel_001"


# ── Text ranking (soft assertions) ────────────────────────────────────────────


@pytest.mark.integration
def test_luxury_query_ranks_marbella_in_top_two(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="luxury spa adults", top_k=6)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    top_ids = [h.id for h in result.hits[:2]]
    assert "test_hotel_002" in top_ids  # La Vie Luxury Adults Retreat


@pytest.mark.integration
def test_quiet_nature_query_ranks_menorca_in_top_two(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="quiet peaceful nature retreat", top_k=6)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    top_ids = [h.id for h in result.hits[:2]]
    assert "test_hotel_005" in top_ids  # Menorca Quiet Retreat


@pytest.mark.integration
def test_culture_query_ranks_marrakech_in_top_two(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="cultural heritage authentic medina", top_k=6)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    top_ids = [h.id for h in result.hits[:2]]
    assert "test_hotel_006" in top_ids  # Marrakech Heritage Riad


# ── Aggregations ──────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_aggregations_returned(opensearch_client: OpenSearch, lexical_test_index: str) -> None:
    params = LexicalSearchParams(query="")
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    assert "countries" in result.raw_aggregations
    assert "star_ratings" in result.raw_aggregations


@pytest.mark.integration
def test_country_aggregation_bucket_keys(
    opensearch_client: OpenSearch, lexical_test_index: str
) -> None:
    params = LexicalSearchParams(query="", top_k=100)
    result = lexical_search(opensearch_client, params, index=lexical_test_index)
    country_keys = {b["key"] for b in result.raw_aggregations["countries"]["buckets"]}
    assert "Spain" in country_keys
    assert "Portugal" in country_keys
    assert "Greece" in country_keys
    assert "Morocco" in country_keys
