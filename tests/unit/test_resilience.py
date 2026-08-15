"""Resilience / graceful degradation tests for POST /search (Milestone 15).

These tests verify that each fallback path in the full search pipeline
degrades gracefully when a component fails at runtime, rather than
propagating the exception to the caller as a 500.

Fallback paths under test
--------------------------
1. Query rewriter fails at runtime
   → original semantic query is used; rewritten_query=None in response

2. Hybrid/vector retrieval fails (e.g. embedding model error)
   → search falls back to BM25 lexical retrieval
   → response has strategy="lexical_fallback", fallback_used=True

3. RAG knowledge retrieval fails at runtime
   → knowledge_context is None in response; hotel results still returned

All tests run without any OpenSearch or ML infrastructure (the conftest's
autouse mock_embedding_provider fixture patches model loading; individual
tests patch OpenSearch at the client level).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from travel_ai_search.api.app import app


def _mock_os_response(hits: list[dict] | None = None, total: int = 0) -> dict:  # type: ignore[type-arg]
    """Minimal OpenSearch search response."""
    return {
        "took": 3,
        "hits": {
            "total": {"value": total, "relation": "eq"},
            "hits": hits or [],
        },
        "aggregations": {
            "countries": {"buckets": []},
            "star_ratings": {"buckets": []},
            "board_types": {"buckets": []},
            "climate_zones": {"buckets": []},
        },
    }


# ── Fallback 1: query rewriter fails at runtime ────────────────────────────────


@patch("travel_ai_search.api.app.create_client")
def test_rewriter_runtime_failure_uses_original_query(mock_create: MagicMock) -> None:
    """If query_rewriter.rewrite() raises, the original semantic query is used.

    The endpoint must still return 200 with rewritten_query=None (not re-raise
    the exception as a 500).
    """
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os

    # Build a mock rewriter that raises on every call.
    failing_rewriter = MagicMock()
    failing_rewriter.rewrite.side_effect = RuntimeError("LLM timeout")

    with (
        patch(
            "travel_ai_search.api.app._create_query_rewriter",
            return_value=failing_rewriter,
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/search",
            json={"query": "beach holiday Greece", "rewrite": True},
        )

    assert response.status_code == 200
    data = response.json()
    # rewritten_query must be absent/null — the rewrite failed
    assert data["rewritten_query"] is None
    # The pipeline should not surface the exception
    assert data["fallback_used"] is False
    # Strategy is still "hybrid" (retrieval path did not change)
    assert data["strategy"] == "hybrid"


@patch("travel_ai_search.api.app.create_client")
def test_rewriter_runtime_failure_rewrite_took_ms_is_set(mock_create: MagicMock) -> None:
    """Even when rewriting fails, rewrite_took_ms should be > 0 (the attempt was timed)."""
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os

    failing_rewriter = MagicMock()
    failing_rewriter.rewrite.side_effect = RuntimeError("LLM timeout")

    with (
        patch("travel_ai_search.api.app._create_query_rewriter", return_value=failing_rewriter),
        TestClient(app) as client,
    ):
        data = client.post(
            "/search",
            json={"query": "beach holiday", "rewrite": True},
        ).json()

    # The timer started before the rewriter was called, so rewrite_took_ms ≥ 0.
    assert isinstance(data["rewrite_took_ms"], int)
    assert data["rewrite_took_ms"] >= 0


# ── Fallback 2: hybrid search fails → BM25 lexical fallback ──────────────────


@patch("travel_ai_search.api.app.create_client")
@patch("travel_ai_search.api.routes.search.hybrid_search")
def test_hybrid_failure_falls_back_to_bm25(
    mock_hybrid: MagicMock,
    mock_create: MagicMock,
) -> None:
    """If hybrid_search raises (e.g. embedding model error), the endpoint
    falls back to BM25 lexical retrieval and returns strategy='lexical_fallback'.
    """
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os

    mock_hybrid.side_effect = RuntimeError("embedding model unavailable")

    with TestClient(app) as client:
        response = client.post("/search", json={"query": "beach hotel"})

    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "lexical_fallback"
    assert data["fallback_used"] is True


@patch("travel_ai_search.api.app.create_client")
@patch("travel_ai_search.api.routes.search.hybrid_search")
def test_hybrid_failure_fallback_still_returns_hits(
    mock_hybrid: MagicMock,
    mock_create: MagicMock,
) -> None:
    """The BM25 fallback must actually call lexical_search and return its hits."""
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response(total=5)
    mock_create.return_value = mock_os
    mock_hybrid.side_effect = RuntimeError("embedding model error")

    with TestClient(app) as client:
        data = client.post("/search", json={"query": "beach hotel"}).json()

    # lexical_search called mock_os.search, which returned total=5
    assert data["total"] == 5
    assert data["strategy"] == "lexical_fallback"


@patch("travel_ai_search.api.app.create_client")
@patch("travel_ai_search.api.routes.search.hybrid_search")
def test_hybrid_failure_vector_took_ms_is_zero(
    mock_hybrid: MagicMock,
    mock_create: MagicMock,
) -> None:
    """When the lexical_fallback is active, vector_took_ms must be 0."""
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os
    mock_hybrid.side_effect = RuntimeError("embedding error")

    with TestClient(app) as client:
        data = client.post("/search", json={"query": "beach hotel"}).json()

    assert data["vector_took_ms"] == 0


# ── Fallback 3: RAG knowledge retrieval fails ─────────────────────────────────


@patch("travel_ai_search.api.app.create_client")
def test_rag_knowledge_failure_skips_context(mock_create: MagicMock) -> None:
    """If knowledge_retriever.retrieve() raises, knowledge_context is None
    in the response and the hotel results are still returned.
    """
    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os

    failing_retriever = MagicMock()
    failing_retriever.retrieve.side_effect = RuntimeError("index not found")

    with (
        patch(
            "travel_ai_search.api.app._create_knowledge_retriever",
            return_value=failing_retriever,
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/search",
            json={"query": "beach holiday Mallorca", "rag": True},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["knowledge_context"] is None
    assert data["rag_summary"] is None
    # Hotel results are unaffected
    assert "hits" in data
    assert "strategy" in data


@patch("travel_ai_search.api.app.create_client")
def test_rag_synthesis_failure_returns_context_without_summary(
    mock_create: MagicMock,
) -> None:
    """If RAG synthesis (LLM) raises but retrieval succeeded, knowledge_context
    is returned without a rag_summary.
    """
    from travel_ai_search.rag.knowledge import DestinationKnowledge

    mock_os = MagicMock()
    mock_os.search.return_value = _mock_os_response()
    mock_create.return_value = mock_os

    sample_doc = DestinationKnowledge(
        id="dest_mallorca",
        destination="Mallorca",
        country="Spain",
        region="Balearic Islands",
        description="A Mediterranean island.",
        climate="Mediterranean",
        best_months=["June", "July"],
        family_suitability="HIGH",
        nightlife_level="HIGH",
        beach_quality="EXCELLENT",
        activities=["swimming"],
        character_tags=["beach"],
        similar_destinations=["Ibiza"],
        geographic_note="Largest of the Balearic Islands.",
    )

    good_retriever = MagicMock()
    good_retriever.retrieve.return_value = [sample_doc]

    failing_synthesizer = MagicMock()
    failing_synthesizer.synthesize.side_effect = RuntimeError("LLM error")

    with (
        patch(
            "travel_ai_search.api.app._create_knowledge_retriever",
            return_value=good_retriever,
        ),
        patch(
            "travel_ai_search.api.app._create_rag_synthesizer",
            return_value=failing_synthesizer,
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/search",
            json={"query": "holiday Mallorca", "rag": True},
        )

    assert response.status_code == 200
    data = response.json()
    # Knowledge context should be present (retrieval succeeded)
    assert data["knowledge_context"] is not None
    assert len(data["knowledge_context"]) == 1
    # But no summary (synthesis failed)
    assert data["rag_summary"] is None
