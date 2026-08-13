"""Unit tests for vector retrieval query construction.

_build_vector_query() is a pure function — no OpenSearch, no embedding model.
Tests verify the exact structure OpenSearch expects for k-NN queries.
"""

from __future__ import annotations

from travel_ai_search.retrieval.vector import VectorSearchParams, _build_vector_query

_DIM = 8
_VECTOR = [0.1] * _DIM


# ── Top-level structure ───────────────────────────────────────────────────────


def test_query_has_size_key() -> None:
    params = VectorSearchParams(query="beach", top_k=10)
    body = _build_vector_query(_VECTOR, params)
    assert "size" in body


def test_query_size_matches_top_k() -> None:
    params = VectorSearchParams(query="beach", top_k=7)
    body = _build_vector_query(_VECTOR, params)
    assert body["size"] == 7


def test_query_has_knn_clause() -> None:
    params = VectorSearchParams(query="beach", top_k=10)
    body = _build_vector_query(_VECTOR, params)
    assert "knn" in body["query"]


def test_knn_targets_embedding_vector_field() -> None:
    params = VectorSearchParams(query="beach", top_k=10)
    body = _build_vector_query(_VECTOR, params)
    knn = body["query"]["knn"]
    assert "embedding_vector" in knn


def test_knn_vector_matches_input() -> None:
    params = VectorSearchParams(query="beach", top_k=10)
    body = _build_vector_query(_VECTOR, params)
    assert body["query"]["knn"]["embedding_vector"]["vector"] == _VECTOR


def test_knn_k_matches_top_k() -> None:
    params = VectorSearchParams(query="beach", top_k=5)
    body = _build_vector_query(_VECTOR, params)
    assert body["query"]["knn"]["embedding_vector"]["k"] == 5


def test_source_excludes_embedding_vector() -> None:
    params = VectorSearchParams(query="beach", top_k=10)
    body = _build_vector_query(_VECTOR, params)
    excludes = body["_source"]["excludes"]
    assert "embedding_vector" in excludes


# ── No filters → no filter clause ────────────────────────────────────────────


def test_no_filters_produces_no_filter_key() -> None:
    params = VectorSearchParams(query="beach", top_k=10)
    body = _build_vector_query(_VECTOR, params)
    knn_clause = body["query"]["knn"]["embedding_vector"]
    assert "filter" not in knn_clause


# ── Individual filter types ───────────────────────────────────────────────────


def test_country_filter_added() -> None:
    params = VectorSearchParams(query="beach", top_k=10, country="Spain")
    body = _build_vector_query(_VECTOR, params)
    filters = body["query"]["knn"]["embedding_vector"]["filter"]["bool"]["filter"]
    assert {"term": {"country": "Spain"}} in filters


def test_destination_filter_added() -> None:
    params = VectorSearchParams(query="beach", top_k=10, destination="Marbella")
    body = _build_vector_query(_VECTOR, params)
    filters = body["query"]["knn"]["embedding_vector"]["filter"]["bool"]["filter"]
    assert {"term": {"destination": "Marbella"}} in filters


def test_family_friendly_filter_added() -> None:
    params = VectorSearchParams(query="beach", top_k=10, family_friendly=True)
    body = _build_vector_query(_VECTOR, params)
    filters = body["query"]["knn"]["embedding_vector"]["filter"]["bool"]["filter"]
    assert {"term": {"family_friendly": True}} in filters


def test_adults_only_filter_added() -> None:
    params = VectorSearchParams(query="retreat", top_k=10, adults_only=True)
    body = _build_vector_query(_VECTOR, params)
    filters = body["query"]["knn"]["embedding_vector"]["filter"]["bool"]["filter"]
    assert {"term": {"adults_only": True}} in filters


def test_min_star_rating_filter_added() -> None:
    params = VectorSearchParams(query="beach", top_k=10, min_star_rating=4)
    body = _build_vector_query(_VECTOR, params)
    filters = body["query"]["knn"]["embedding_vector"]["filter"]["bool"]["filter"]
    assert {"range": {"star_rating": {"gte": 4}}} in filters


def test_max_price_filter_added() -> None:
    params = VectorSearchParams(query="beach", top_k=10, max_price=1000.0)
    body = _build_vector_query(_VECTOR, params)
    filters = body["query"]["knn"]["embedding_vector"]["filter"]["bool"]["filter"]
    assert {"range": {"price_per_person_gbp": {"lte": 1000.0}}} in filters


def test_month_filter_added() -> None:
    params = VectorSearchParams(query="beach", top_k=10, month="July")
    body = _build_vector_query(_VECTOR, params)
    filters = body["query"]["knn"]["embedding_vector"]["filter"]["bool"]["filter"]
    assert {"term": {"available_months": "July"}} in filters


def test_airport_filter_added() -> None:
    params = VectorSearchParams(query="beach", top_k=10, airport="LGW")
    body = _build_vector_query(_VECTOR, params)
    filters = body["query"]["knn"]["embedding_vector"]["filter"]["bool"]["filter"]
    assert {"term": {"available_departure_airports": "LGW"}} in filters


# ── Multiple filters ──────────────────────────────────────────────────────────


def test_multiple_filters_all_present() -> None:
    params = VectorSearchParams(
        query="beach",
        top_k=10,
        country="Spain",
        family_friendly=True,
        max_price=800.0,
    )
    body = _build_vector_query(_VECTOR, params)
    filters = body["query"]["knn"]["embedding_vector"]["filter"]["bool"]["filter"]
    assert len(filters) == 3
    assert {"term": {"country": "Spain"}} in filters
    assert {"term": {"family_friendly": True}} in filters
    assert {"range": {"price_per_person_gbp": {"lte": 800.0}}} in filters
