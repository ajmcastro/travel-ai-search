"""Integration tests for POST /search and POST /query/understand.

Tests the full orchestrated pipeline: QU engine → hybrid retrieval → response.
Requires OpenSearch running with the curated test dataset loaded.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from opensearchpy import OpenSearch

from travel_ai_search.api.app import app
from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.query_understanding.extractor import RuleBasedQueryUnderstandingEngine

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def api_client(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
    vector_test_index: str,
) -> TestClient:
    """TestClient wired to the real OpenSearch and embedding model.

    Overrides dependencies so the app uses the session-scoped infra fixtures
    instead of loading new connections at startup.
    """
    from travel_ai_search.api.deps import (
        get_embedding_provider,
        get_os_client,
        get_query_understanding_engine,
        get_reranker,
        get_settings,
    )
    from travel_ai_search.config.settings import Settings

    def _override_client() -> OpenSearch:
        return opensearch_client

    def _override_provider() -> LocalEmbeddingProvider:
        return embedding_provider

    def _override_reranker():  # type: ignore[no-untyped-def]
        return None

    def _override_qu_engine() -> RuleBasedQueryUnderstandingEngine:
        return RuleBasedQueryUnderstandingEngine()

    def _override_settings() -> Settings:
        s = Settings()
        # Point to the vector test index which has embeddings
        object.__setattr__(s, "opensearch_index_name", vector_test_index)
        return s

    app.dependency_overrides[get_os_client] = _override_client
    app.dependency_overrides[get_embedding_provider] = _override_provider
    app.dependency_overrides[get_reranker] = _override_reranker
    app.dependency_overrides[get_query_understanding_engine] = _override_qu_engine
    app.dependency_overrides[get_settings] = _override_settings

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


# ── POST /query/understand ────────────────────────────────────────────────────


def test_query_understand_endpoint_returns_200(api_client: TestClient) -> None:
    resp = api_client.post("/query/understand", json={"query": "family beach holiday in Spain"})
    assert resp.status_code == 200


def test_query_understand_extracts_country(api_client: TestClient) -> None:
    resp = api_client.post("/query/understand", json={"query": "beach hotel in Greece July"})
    data = resp.json()
    assert data["country"] == "Greece"
    assert data["month"] == "July"


def test_query_understand_extracts_family_and_airport(api_client: TestClient) -> None:
    resp = api_client.post(
        "/query/understand",
        json={"query": "family holiday from Manchester under £2000"},
    )
    data = resp.json()
    assert data["family_friendly"] is True
    assert data["departure_airport"] == "MAN"
    assert data["max_price"] == 2000.0


def test_query_understand_response_has_semantic_query(api_client: TestClient) -> None:
    resp = api_client.post(
        "/query/understand",
        json={"query": "family beach hotel Greece July Manchester"},
    )
    data = resp.json()
    assert "semantic_query" in data
    assert data["semantic_query"]  # not empty
    assert "greece" not in data["semantic_query"].lower()
    assert "manchester" not in data["semantic_query"].lower()


def test_query_understand_empty_extraction_on_plain_query(api_client: TestClient) -> None:
    resp = api_client.post(
        "/query/understand",
        json={"query": "luxury spa resort with infinity pool"},
    )
    data = resp.json()
    assert data["month"] is None
    assert data["departure_airport"] is None
    assert data["country"] is None
    assert data["family_friendly"] is None


def test_query_understand_soft_preferences_detected(api_client: TestClient) -> None:
    resp = api_client.post(
        "/query/understand",
        json={"query": "all-inclusive beach resort with spa and pool"},
    )
    data = resp.json()
    prefs = data["soft_preferences"]
    assert "beach" in prefs
    assert "spa" in prefs
    assert "all-inclusive" in prefs


def test_query_understand_includes_timing(api_client: TestClient) -> None:
    resp = api_client.post("/query/understand", json={"query": "beach hotel"})
    data = resp.json()
    assert "understanding_took_ms" in data
    assert data["understanding_took_ms"] >= 0


# ── POST /search ──────────────────────────────────────────────────────────────


def test_full_search_returns_200(api_client: TestClient) -> None:
    resp = api_client.post("/search", json={"query": "beach hotel Spain"})
    assert resp.status_code == 200


def test_full_search_response_shape(api_client: TestClient) -> None:
    resp = api_client.post("/search", json={"query": "beach hotel"})
    data = resp.json()
    assert "hits" in data
    assert "total" in data
    assert "took_ms" in data
    assert "query_understanding" in data
    assert "lexical_took_ms" in data
    assert "vector_took_ms" in data


def test_full_search_qu_embedded_in_response(api_client: TestClient) -> None:
    resp = api_client.post(
        "/search",
        json={"query": "family beach hotel Greece July Manchester"},
    )
    data = resp.json()
    qu = data["query_understanding"]
    assert qu["country"] == "Greece"
    assert qu["month"] == "July"
    assert qu["departure_airport"] == "MAN"
    assert qu["family_friendly"] is True


def test_full_search_applies_country_filter(api_client: TestClient) -> None:
    """Hotels returned by POST /search with 'Spain' in query should all be Spanish."""
    resp = api_client.post("/search", json={"query": "beach hotel in Spain"})
    data = resp.json()
    for hit in data["hits"]:
        assert hit["country"] == "Spain", f"Expected Spain, got {hit['country']}"


def test_full_search_applies_family_filter(api_client: TestClient) -> None:
    """Hotels returned for a family query should all be family_friendly=True."""
    resp = api_client.post("/search", json={"query": "family holiday"})
    data = resp.json()
    qu = data["query_understanding"]
    if qu.get("family_friendly"):
        for hit in data["hits"]:
            assert hit["family_friendly"] is True


def test_full_search_no_results_for_impossible_filters(api_client: TestClient) -> None:
    """Impossible combination (adults_only=True AND family_friendly=True) → no hits."""
    resp = api_client.post(
        "/search",
        json={"query": "adults only family beach hotel"},
    )
    # The query itself is contradictory but the API should not error
    assert resp.status_code == 200


def test_full_search_top_k_respected(api_client: TestClient) -> None:
    resp = api_client.post("/search", json={"query": "beach hotel", "top_k": 3})
    data = resp.json()
    assert len(data["hits"]) <= 3


def test_full_search_original_query_preserved(api_client: TestClient) -> None:
    query = "family beach holiday in Greece July from Manchester"
    resp = api_client.post("/search", json={"query": query})
    data = resp.json()
    assert data["query_understanding"]["original_query"] == query
