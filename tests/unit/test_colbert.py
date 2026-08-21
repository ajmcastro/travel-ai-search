"""Unit tests for ColBERT late-interaction reranker — Milestone 18.

All tests use mocked/injected encoders so no real model is loaded.
ColBERTReranker accepts a ``_encoder`` argument for dependency injection.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from travel_ai_search.reranking.colbert import (
    ColBERTReranker,
    build_colbert_document_text,
    maxsim,
)
from travel_ai_search.retrieval.types import Hit

# ── Helpers ────────────────────────────────────────────────────────────────────


def _hit(hotel_id: str, score: float = 1.0, source: dict[str, Any] | None = None) -> Hit:
    return Hit(
        id=hotel_id,
        score=score,
        source=source
        or {
            "hotel_name": f"Hotel {hotel_id}",
            "destination": "Paris",
            "hotel_description": f"A lovely hotel in Paris, {hotel_id}.",
            "activities": ["sightseeing"],
            "tags": ["romantic"],
        },
    )


def _rand_embs(seq_len: int, dim: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((seq_len, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (embs / norms).astype(np.float32)


class _FakeEncoder:
    """Deterministic encoder that returns small fixed arrays without loading a model."""

    def __init__(self, dim: int = 8, q_len: int = 4, d_len: int = 6) -> None:
        self.dim = dim
        self.model_id = "fake"
        self._q_len = q_len
        self._d_len = d_len

    def encode_query(self, text: str) -> np.ndarray:
        seed = sum(ord(c) for c in text) % 100
        return _rand_embs(self._q_len, self.dim, seed=seed)

    def encode_document(self, text: str) -> np.ndarray:
        seed = (sum(ord(c) for c in text) + 1) % 100
        return _rand_embs(self._d_len, self.dim, seed=seed)


# ── maxsim ─────────────────────────────────────────────────────────────────────


class TestMaxSim:
    def test_returns_float(self) -> None:
        q = _rand_embs(3, 8)
        d = _rand_embs(5, 8)
        result = maxsim(q, d)
        assert isinstance(result, float)

    def test_identical_embeddings_high_score(self) -> None:
        embs = _rand_embs(4, 8)
        score = maxsim(embs, embs)
        assert score > 3.5

    def test_score_positive_with_l2_normalised(self) -> None:
        q = _rand_embs(4, 8)
        d = _rand_embs(6, 8)
        assert maxsim(q, d) >= 0

    def test_antisymmetry_in_general(self) -> None:
        q = _rand_embs(3, 8, seed=1)
        d = _rand_embs(5, 8, seed=2)
        assert maxsim(q, d) != pytest.approx(maxsim(d, q), abs=0.1)

    def test_single_token_query_equals_max_dot_product(self) -> None:
        q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        d = np.array(
            [[0.5, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        score = maxsim(q, d)
        assert score == pytest.approx(1.0)

    def test_two_token_query_sums_individual_maxes(self) -> None:
        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        d = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        score = maxsim(q, d)
        assert score == pytest.approx(2.0)

    def test_orthogonal_embeddings_near_zero(self) -> None:
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        d = np.array([[0.0, 1.0]], dtype=np.float32)
        score = maxsim(q, d)
        assert abs(score) < 1e-5

    def test_monotonic_with_relevance(self) -> None:
        base = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        q = np.repeat(base, 4, axis=0)
        d_relevant = np.repeat(base, 6, axis=0)
        d_irrelevant = np.array([[0.0, 1.0, 0.0]] * 6, dtype=np.float32)
        assert maxsim(q, d_relevant) > maxsim(q, d_irrelevant)

    def test_shape_invariance_dim(self) -> None:
        for dim in (8, 32, 128):
            q = _rand_embs(4, dim)
            d = _rand_embs(8, dim)
            result = maxsim(q, d)
            assert math.isfinite(result)

    def test_max_score_is_bounded_by_q_len(self) -> None:
        dim = 8
        q = _rand_embs(4, dim)
        d = _rand_embs(6, dim)
        assert maxsim(q, d) <= 4.0 + 1e-5


# ── ColBERTReranker ────────────────────────────────────────────────────────────


class TestColBERTReranker:
    def _make_reranker(self, tmp_path: Path) -> ColBERTReranker:
        enc = _FakeEncoder(dim=8, q_len=3, d_len=5)
        return ColBERTReranker(_encoder=enc, embeddings_dir=tmp_path)

    def _write_emb(self, hotel_id: str, tmp_path: Path, seed: int = 0) -> None:
        embs = _rand_embs(5, 8, seed=seed)
        np.save(str(tmp_path / f"{hotel_id}.npy"), embs)

    def test_empty_hits_returns_empty(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        result = r.rerank("beaches", [], top_k=5)
        assert result == []

    def test_returns_at_most_top_k(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        for i in range(6):
            self._write_emb(f"h{i}", tmp_path, seed=i)
        hits = [_hit(f"h{i}") for i in range(6)]
        result = r.rerank("beach holiday", hits, top_k=3)
        assert len(result) == 3

    def test_hits_with_no_npy_keep_original_score(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        self._write_emb("known", tmp_path, seed=0)
        hits = [_hit("known", score=0.9), _hit("unknown", score=0.1)]
        result = r.rerank("query", hits, top_k=2)
        ids = [h.id for h in result]
        assert set(ids) == {"known", "unknown"}
        unknown_score = next(h.score for h in result if h.id == "unknown")
        assert unknown_score == pytest.approx(0.1)

    def test_result_is_sorted_descending(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        for i in range(5):
            self._write_emb(f"h{i}", tmp_path, seed=i)
        hits = [_hit(f"h{i}") for i in range(5)]
        result = r.rerank("beach", hits, top_k=5)
        scores = [h.score for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_hit_objects_have_correct_ids(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        for i in range(3):
            self._write_emb(f"hotel_{i}", tmp_path, seed=i)
        hits = [_hit(f"hotel_{i}") for i in range(3)]
        result = r.rerank("query", hits, top_k=3)
        assert len(result) == 3
        assert all(h.id.startswith("hotel_") for h in result)

    def test_source_preserved_in_result(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        self._write_emb("h1", tmp_path)
        src = {"hotel_name": "Test Hotel"}
        hits = [_hit("h1", source=src)]
        result = r.rerank("query", hits, top_k=1)
        assert result[0].source == src

    def test_scores_are_floats(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        self._write_emb("h1", tmp_path)
        hits = [_hit("h1")]
        result = r.rerank("query", hits, top_k=1)
        assert isinstance(result[0].score, float)

    def test_top_k_larger_than_hits_returns_all(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        self._write_emb("h1", tmp_path)
        hits = [_hit("h1")]
        result = r.rerank("query", hits, top_k=100)
        assert len(result) == 1

    def test_all_missing_embeddings_returns_original_scores(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        hits = [_hit("x1", score=0.8), _hit("x2", score=0.3)]
        result = r.rerank("query", hits, top_k=2)
        scores = {h.id: h.score for h in result}
        assert scores["x1"] == pytest.approx(0.8)
        assert scores["x2"] == pytest.approx(0.3)

    def test_colbert_score_different_from_bm25_score(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        self._write_emb("h1", tmp_path, seed=42)
        original_score = 0.5
        hits = [_hit("h1", score=original_score)]
        result = r.rerank("beach vacation", hits, top_k=1)
        assert result[0].score != pytest.approx(original_score)

    def test_consistent_results_same_query(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        for i in range(4):
            self._write_emb(f"h{i}", tmp_path, seed=i)
        hits = [_hit(f"h{i}") for i in range(4)]
        result1 = r.rerank("relaxing beach resort", hits, top_k=4)
        result2 = r.rerank("relaxing beach resort", hits, top_k=4)
        assert [h.id for h in result1] == [h.id for h in result2]

    def test_different_queries_can_produce_different_orderings(self, tmp_path: Path) -> None:
        enc = _FakeEncoder(dim=8, q_len=3, d_len=5)
        r = ColBERTReranker(_encoder=enc, embeddings_dir=tmp_path)
        for i in range(4):
            embs = _rand_embs(5, 8, seed=i * 10)
            np.save(str(tmp_path / f"h{i}.npy"), embs)
        hits = [_hit(f"h{i}") for i in range(4)]
        result_q1 = [h.id for h in r.rerank("beach", hits, top_k=4)]
        result_q2 = [h.id for h in r.rerank("mountain ski resort", hits, top_k=4)]
        # Not guaranteed to differ for random embeddings, but this at least verifies
        # the reranker runs two separate encodes.
        assert len(result_q1) == 4
        assert len(result_q2) == 4

    def test_missing_file_triggers_warning(self, tmp_path: Path, caplog: Any) -> None:
        import logging

        r = self._make_reranker(tmp_path)
        hits = [_hit("missing_hotel")]
        with caplog.at_level(logging.WARNING, logger="travel_ai_search.reranking.colbert"):
            r.rerank("query", hits, top_k=1)
        assert any("lacked embeddings" in record.message for record in caplog.records)

    def test_npy_shape_is_loaded_correctly(self, tmp_path: Path) -> None:
        r = self._make_reranker(tmp_path)
        embs = np.eye(5, 8, dtype=np.float32)
        np.save(str(tmp_path / "h_eye.npy"), embs)
        hits = [_hit("h_eye")]
        result = r.rerank("query", hits, top_k=1)
        assert len(result) == 1


# ── build_colbert_document_text ────────────────────────────────────────────────


class TestBuildColbertDocumentText:
    def _src(self) -> dict[str, Any]:
        return {
            "hotel_name": "Sunny Resort",
            "destination": "Mallorca",
            "country": "Spain",
            "hotel_description": "A lovely beachside hotel.",
            "activities": ["surfing", "yoga"],
            "tags": ["family", "beach"],
        }

    def test_returns_non_empty_string(self) -> None:
        text = build_colbert_document_text(self._src())
        assert isinstance(text, str) and len(text) > 10

    def test_includes_hotel_name(self) -> None:
        text = build_colbert_document_text(self._src())
        assert "Sunny Resort" in text

    def test_includes_description(self) -> None:
        text = build_colbert_document_text(self._src())
        assert "beachside" in text

    def test_activities_included(self) -> None:
        text = build_colbert_document_text(self._src())
        assert "surfing" in text

    def test_tags_included(self) -> None:
        text = build_colbert_document_text(self._src())
        assert "family" in text

    def test_empty_source_does_not_raise(self) -> None:
        text = build_colbert_document_text({})
        assert isinstance(text, str)

    def test_missing_optional_fields_does_not_raise(self) -> None:
        text = build_colbert_document_text({"hotel_name": "X"})
        assert "X" in text


# ── Protocol compliance ────────────────────────────────────────────────────────


class TestRerankerProtocol:
    def test_colbert_reranker_satisfies_reranker_protocol(self, tmp_path: Path) -> None:
        from travel_ai_search.reranking.base import Reranker

        enc = _FakeEncoder()
        r = ColBERTReranker(_encoder=enc, embeddings_dir=tmp_path)
        assert isinstance(r, Reranker)

    def test_rerank_method_signature(self, tmp_path: Path) -> None:
        import inspect

        enc = _FakeEncoder()
        r = ColBERTReranker(_encoder=enc, embeddings_dir=tmp_path)
        sig = inspect.signature(r.rerank)
        params = list(sig.parameters)
        assert "query" in params
        assert "hits" in params
        assert "top_k" in params
