"""Unit tests for the embeddings module.

SentenceTransformer is mocked so these tests run without downloading the model
or loading PyTorch.  They verify the adapter logic — argument forwarding,
output shaping, L2-normalisation — not the model's correctness.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np

from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.embeddings.local import LocalEmbeddingProvider

_DIM = 384


def _make_unit_vector(dim: int) -> np.ndarray:
    """Return an L2-normalised numpy vector of length `dim`."""
    v = np.ones(dim, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_provider() -> tuple[LocalEmbeddingProvider, MagicMock]:
    """Create a LocalEmbeddingProvider with a mocked SentenceTransformer."""
    with patch("travel_ai_search.embeddings.local.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = _DIM
        mock_model.encode.return_value = _make_unit_vector(_DIM)
        mock_cls.return_value = mock_model

        provider = LocalEmbeddingProvider()
        return provider, mock_model


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_local_provider_satisfies_protocol() -> None:
    with patch("travel_ai_search.embeddings.local.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = _DIM
        mock_cls.return_value = mock_model

        provider = LocalEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)


# ── Dimension property ────────────────────────────────────────────────────────


def test_dimension_returns_int() -> None:
    provider, _ = _make_provider()
    assert isinstance(provider.dimension, int)


def test_dimension_matches_model() -> None:
    provider, _ = _make_provider()
    assert provider.dimension == _DIM


def test_dimension_fallback_when_model_returns_none() -> None:
    with patch("travel_ai_search.embeddings.local.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = None
        mock_cls.return_value = mock_model

        provider = LocalEmbeddingProvider()

    assert provider.dimension == 384


# ── embed() ──────────────────────────────────────────────────────────────────


def test_embed_returns_list_of_floats() -> None:
    provider, mock_model = _make_provider()
    mock_model.encode.return_value = _make_unit_vector(_DIM)

    result = provider.embed("beach hotel")

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


def test_embed_returns_correct_dimension() -> None:
    provider, mock_model = _make_provider()
    mock_model.encode.return_value = _make_unit_vector(_DIM)

    result = provider.embed("beach hotel")

    assert len(result) == _DIM


def test_embed_passes_normalize_flag() -> None:
    provider, mock_model = _make_provider()
    mock_model.encode.return_value = _make_unit_vector(_DIM)

    provider.embed("beach hotel")

    call_kwargs = mock_model.encode.call_args[1]
    assert call_kwargs.get("normalize_embeddings") is True


def test_embed_output_is_unit_norm() -> None:
    """embed() must return an L2-normalised vector (norm ≈ 1.0)."""
    provider, mock_model = _make_provider()
    mock_model.encode.return_value = _make_unit_vector(_DIM)

    result = provider.embed("any text")

    norm = math.sqrt(sum(x * x for x in result))
    assert abs(norm - 1.0) < 1e-5


# ── embed_batch() ─────────────────────────────────────────────────────────────


def test_embed_batch_empty_returns_empty() -> None:
    provider, _ = _make_provider()
    result = provider.embed_batch([])
    assert result == []


def test_embed_batch_returns_list_of_lists() -> None:
    provider, mock_model = _make_provider()
    texts = ["beach hotel", "mountain retreat"]
    mock_model.encode.return_value = np.stack([_make_unit_vector(_DIM), _make_unit_vector(_DIM)])

    result = provider.embed_batch(texts)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(v, list) for v in result)


def test_embed_batch_each_vector_correct_dimension() -> None:
    provider, mock_model = _make_provider()
    texts = ["a", "b", "c"]
    mock_model.encode.return_value = np.stack([_make_unit_vector(_DIM)] * 3)

    result = provider.embed_batch(texts)

    assert all(len(v) == _DIM for v in result)


def test_embed_batch_passes_batch_size() -> None:
    provider, mock_model = _make_provider()
    texts = ["a", "b"]
    mock_model.encode.return_value = np.stack([_make_unit_vector(_DIM)] * 2)

    provider.embed_batch(texts)

    call_kwargs = mock_model.encode.call_args[1]
    assert "batch_size" in call_kwargs


def test_embed_batch_skips_model_for_empty_input() -> None:
    provider, mock_model = _make_provider()
    provider.embed_batch([])
    mock_model.encode.assert_not_called()
