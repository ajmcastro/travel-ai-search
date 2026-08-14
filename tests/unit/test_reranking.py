"""Unit tests for the reranking module — no infrastructure or model loading required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from travel_ai_search.reranking.base import Reranker
from travel_ai_search.reranking.local import LocalCrossEncoderReranker, _build_reranking_text
from travel_ai_search.retrieval.types import Hit

# ── Helpers ───────────────────────────────────────────────────────────────────

_ADULTS_SOURCE: dict = {
    "hotel_name": "La Vie Luxury Adults Retreat",
    "destination": "Marbella",
    "country": "Spain",
    "hotel_description": "Exclusive adults-only sanctuary with infinity pool and spa.",
    "activities": ["spa", "golf", "wine tasting"],
    "tags": ["luxury", "adults_only", "spa"],
}

_FAMILY_SOURCE: dict = {
    "hotel_name": "Sunny Beach Family Resort",
    "destination": "Benidorm",
    "country": "Spain",
    "hotel_description": "Family-friendly resort with waterslides and kids club.",
    "activities": ["swimming", "beach volleyball", "kids club"],
    "tags": ["family", "beach", "pool"],
}

_EMPTY_SOURCE: dict = {}


def _make_hit(id: str, score: float, source: dict) -> Hit:  # type: ignore[type-arg]
    return Hit(id=id, score=score, source=source)


def _make_mock_reranker(scores: list[float]) -> LocalCrossEncoderReranker:
    """Return a LocalCrossEncoderReranker whose model predict() returns `scores`."""
    with patch("travel_ai_search.reranking.local.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        import numpy as np

        mock_model.predict.return_value = np.array(scores)
        mock_cls.return_value = mock_model
        reranker = LocalCrossEncoderReranker()
    reranker._model = mock_model  # keep the mock alive after the patch context exits
    return reranker


# ── _build_reranking_text ─────────────────────────────────────────────────────


def test_build_text_contains_hotel_name() -> None:
    text = _build_reranking_text(_ADULTS_SOURCE)
    assert "La Vie Luxury Adults Retreat" in text


def test_build_text_contains_destination_and_country() -> None:
    text = _build_reranking_text(_ADULTS_SOURCE)
    assert "Marbella" in text
    assert "Spain" in text


def test_build_text_contains_description() -> None:
    text = _build_reranking_text(_ADULTS_SOURCE)
    assert "infinity pool" in text


def test_build_text_contains_activities() -> None:
    text = _build_reranking_text(_ADULTS_SOURCE)
    assert "spa" in text
    assert "golf" in text


def test_build_text_contains_tags() -> None:
    text = _build_reranking_text(_ADULTS_SOURCE)
    assert "luxury" in text
    assert "adults_only" in text


def test_build_text_handles_missing_fields() -> None:
    text = _build_reranking_text(_EMPTY_SOURCE)
    assert isinstance(text, str)


def test_build_text_skips_empty_activities() -> None:
    source = {**_FAMILY_SOURCE, "activities": []}
    text = _build_reranking_text(source)
    assert "Activities:" not in text


def test_build_text_skips_empty_tags() -> None:
    source = {**_FAMILY_SOURCE, "tags": []}
    text = _build_reranking_text(source)
    assert "Tags:" not in text


# ── Reranker protocol ─────────────────────────────────────────────────────────


def test_structural_typing_any_object_with_rerank_satisfies_protocol() -> None:
    """Structural typing: any object with a rerank() method satisfies the Protocol."""

    class _Stub:
        def rerank(self, query: str, hits: list[Hit], *, top_k: int) -> list[Hit]:
            return hits[:top_k]

    assert isinstance(_Stub(), Reranker)


def test_local_reranker_satisfies_protocol() -> None:
    reranker = _make_mock_reranker([0.5])
    assert isinstance(reranker, Reranker)


# ── LocalCrossEncoderReranker.rerank ─────────────────────────────────────────


def test_rerank_empty_hits_returns_empty() -> None:
    reranker = _make_mock_reranker([])
    result = reranker.rerank("luxury spa", [], top_k=5)
    assert result == []


def test_rerank_sorts_by_cross_encoder_score_descending() -> None:
    hits = [
        _make_hit("a", 0.9, _FAMILY_SOURCE),
        _make_hit("b", 0.5, _ADULTS_SOURCE),
        _make_hit("c", 0.7, _EMPTY_SOURCE),
    ]
    # Cross-encoder gives: a→0.1, b→0.9, c→0.5
    reranker = _make_mock_reranker([0.1, 0.9, 0.5])
    result = reranker.rerank("adults luxury spa", hits, top_k=3)
    assert [h.id for h in result] == ["b", "c", "a"]


def test_rerank_replaces_original_scores() -> None:
    hits = [_make_hit("x", 100.0, _ADULTS_SOURCE)]
    reranker = _make_mock_reranker([7.3])
    result = reranker.rerank("spa", hits, top_k=1)
    assert len(result) == 1
    assert result[0].score == pytest.approx(7.3)


def test_rerank_truncates_to_top_k() -> None:
    hits = [_make_hit(str(i), float(i), _FAMILY_SOURCE) for i in range(10)]
    reranker = _make_mock_reranker(list(range(10)))
    result = reranker.rerank("beach", hits, top_k=3)
    assert len(result) == 3


def test_rerank_fewer_hits_than_top_k_returns_all() -> None:
    hits = [_make_hit("a", 1.0, _ADULTS_SOURCE), _make_hit("b", 0.5, _FAMILY_SOURCE)]
    reranker = _make_mock_reranker([0.8, 0.3])
    result = reranker.rerank("spa", hits, top_k=10)
    assert len(result) == 2


def test_rerank_preserves_source() -> None:
    hits = [_make_hit("a", 0.5, _ADULTS_SOURCE)]
    reranker = _make_mock_reranker([1.0])
    result = reranker.rerank("luxury", hits, top_k=1)
    assert result[0].source == _ADULTS_SOURCE


def test_rerank_preserves_hit_id() -> None:
    hits = [_make_hit("hotel_999", 0.5, _ADULTS_SOURCE)]
    reranker = _make_mock_reranker([2.5])
    result = reranker.rerank("spa", hits, top_k=1)
    assert result[0].id == "hotel_999"


def test_rerank_single_hit_returned_at_top() -> None:
    hits = [_make_hit("only", 0.1, _FAMILY_SOURCE)]
    reranker = _make_mock_reranker([5.0])
    result = reranker.rerank("family", hits, top_k=1)
    assert result[0].id == "only"
    assert result[0].score == pytest.approx(5.0)


def test_rerank_all_equal_scores_returns_top_k() -> None:
    hits = [_make_hit(str(i), 1.0, _FAMILY_SOURCE) for i in range(5)]
    reranker = _make_mock_reranker([0.5, 0.5, 0.5, 0.5, 0.5])
    result = reranker.rerank("beach", hits, top_k=3)
    assert len(result) == 3


# ── hybrid_search graceful degradation ───────────────────────────────────────


def test_hybrid_search_fallback_when_reranker_raises() -> None:
    """hybrid_search returns fused results (not an exception) when reranker fails."""
    from unittest.mock import MagicMock

    from travel_ai_search.retrieval.fusion import FusionMethod
    from travel_ai_search.retrieval.hybrid import HybridSearchParams, hybrid_search

    mock_client = MagicMock()
    mock_provider = MagicMock()
    mock_provider.embed.return_value = [0.0] * 384

    # Minimal OpenSearch response
    mock_response = {
        "took": 1,
        "hits": {
            "total": {"value": 0, "relation": "eq"},
            "hits": [],
        },
        "aggregations": {
            "countries": {"buckets": []},
            "star_ratings": {"buckets": []},
            "board_types": {"buckets": []},
            "climate_zones": {"buckets": []},
        },
    }
    mock_client.search.return_value = mock_response

    bad_reranker = MagicMock()
    bad_reranker.rerank.side_effect = RuntimeError("model exploded")

    params = HybridSearchParams(query="luxury spa", fusion=FusionMethod.rrf, top_k=5, rerank_k=10)
    # Should not raise — graceful fallback
    result = hybrid_search(mock_client, mock_provider, params, reranker=bad_reranker)
    assert result.hits == []
    assert result.reranking_took_ms >= 0
