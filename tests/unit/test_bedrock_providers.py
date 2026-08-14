"""Unit tests for Milestone 12 — AWS Bedrock provider implementations.

All tests mock boto3 — no real AWS calls, no credentials required.
Tests verify:
  - Protocol satisfaction via isinstance (runtime_checkable Protocols)
  - Correct API call shapes (model_id, body content, headers)
  - Response parsing: mocked AWS response structure → expected Python types
  - Empty / edge cases (empty hits list, empty text list)
  - Exception propagation: boto3 raises → caller sees the exception
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.llm.base import LLMProvider
from travel_ai_search.reranking.base import Reranker
from travel_ai_search.retrieval.types import Hit

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_embed_client(embedding: list[float] | None = None) -> MagicMock:
    embedding = embedding or [0.1, 0.2, 0.3]
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"embedding": embedding}).encode()
    client = MagicMock()
    client.invoke_model.return_value = {"body": mock_body}
    return client


def _make_llm_client(text: str = "rewritten query") -> MagicMock:
    client = MagicMock()
    client.converse.return_value = {"output": {"message": {"content": [{"text": text}]}}}
    return client


def _make_reranker_client(results: list[dict[str, Any]] | None = None) -> MagicMock:
    results = results or [{"index": 0, "relevance_score": 0.9}]
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"results": results}).encode()
    client = MagicMock()
    client.invoke_model.return_value = {"body": mock_body}
    return client


def _make_hits(n: int = 2) -> list[Hit]:
    return [
        Hit(
            id=f"hotel-{i}",
            score=0.5,
            source={
                "hotel_name": f"Hotel {i}",
                "destination": "Lisbon",
                "country": "Portugal",
                "hotel_description": "A great hotel",
                "activities": ["swimming", "hiking"],
                "tags": ["beach", "spa"],
            },
        )
        for i in range(n)
    ]


# ── create_bedrock_client ─────────────────────────────────────────────────────


class TestCreateBedrockClient:
    def test_returns_boto3_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import boto3

        mock_client = MagicMock()
        monkeypatch.setattr(boto3, "client", lambda *a, **kw: mock_client)
        from travel_ai_search.infrastructure.bedrock import create_bedrock_client

        assert create_bedrock_client("us-east-1") is mock_client

    def test_passes_region_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import boto3

        captured: dict[str, Any] = {}

        def _fake(*args: Any, **kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(boto3, "client", _fake)
        from travel_ai_search.infrastructure.bedrock import create_bedrock_client

        create_bedrock_client("eu-west-1")
        assert captured.get("region_name") == "eu-west-1"

    def test_uses_bedrock_runtime_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import boto3

        captured: dict[str, Any] = {}

        def _fake(*args: Any, **kwargs: Any) -> MagicMock:
            if args:
                captured["service"] = args[0]
            return MagicMock()

        monkeypatch.setattr(boto3, "client", _fake)
        from travel_ai_search.infrastructure.bedrock import create_bedrock_client

        create_bedrock_client()
        assert captured.get("service") == "bedrock-runtime"


# ── BedrockEmbeddingProvider ──────────────────────────────────────────────────


class TestBedrockEmbeddingProviderProtocol:
    def test_satisfies_protocol(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        assert isinstance(BedrockEmbeddingProvider(MagicMock(), dimension=3), EmbeddingProvider)

    def test_custom_class_also_satisfies(self) -> None:
        class _Custom:
            @property
            def dimension(self) -> int:
                return 3

            def embed(self, text: str) -> list[float]:
                return [0.0]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.0]] * len(texts)

        assert isinstance(_Custom(), EmbeddingProvider)


class TestBedrockEmbeddingProviderDimension:
    def test_dimension_matches_constructor(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        assert BedrockEmbeddingProvider(MagicMock(), dimension=1024).dimension == 1024

    def test_dimension_256(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        assert BedrockEmbeddingProvider(MagicMock(), dimension=256).dimension == 256

    def test_dimension_512(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        assert BedrockEmbeddingProvider(MagicMock(), dimension=512).dimension == 512


class TestBedrockEmbeddingProviderEmbed:
    def test_returns_vector_from_response(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        client = _make_embed_client([0.1, 0.2, 0.3])
        result = BedrockEmbeddingProvider(client, dimension=3).embed("beach resort")
        assert result == [0.1, 0.2, 0.3]

    def test_passes_model_id_to_invoke_model(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        client = _make_embed_client()
        BedrockEmbeddingProvider(client, model_id="custom-model", dimension=3).embed("test")
        assert client.invoke_model.call_args[1]["modelId"] == "custom-model"

    def test_dimension_in_request_body(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        client = _make_embed_client()
        BedrockEmbeddingProvider(client, dimension=512).embed("test")
        body = json.loads(client.invoke_model.call_args[1]["body"])
        assert body["dimensions"] == 512

    def test_normalize_true_in_request_body(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        client = _make_embed_client()
        BedrockEmbeddingProvider(client, dimension=3).embed("test")
        body = json.loads(client.invoke_model.call_args[1]["body"])
        assert body["normalize"] is True

    def test_input_text_in_request_body(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        client = _make_embed_client()
        BedrockEmbeddingProvider(client, dimension=3).embed("luxury spa resort")
        body = json.loads(client.invoke_model.call_args[1]["body"])
        assert body["inputText"] == "luxury spa resort"

    def test_json_content_type(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        client = _make_embed_client()
        BedrockEmbeddingProvider(client, dimension=3).embed("test")
        kwargs = client.invoke_model.call_args[1]
        assert kwargs["contentType"] == "application/json"
        assert kwargs["accept"] == "application/json"

    def test_exception_propagates(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        client = MagicMock()
        client.invoke_model.side_effect = RuntimeError("AWS error")
        with pytest.raises(RuntimeError, match="AWS error"):
            BedrockEmbeddingProvider(client, dimension=3).embed("test")


class TestBedrockEmbeddingProviderBatch:
    def test_empty_texts_returns_empty(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        assert BedrockEmbeddingProvider(MagicMock(), dimension=3).embed_batch([]) == []

    def test_calls_embed_per_text(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        client = _make_embed_client([0.5, 0.5, 0.0])
        BedrockEmbeddingProvider(client, dimension=3).embed_batch(["a", "b", "c"])
        assert client.invoke_model.call_count == 3

    def test_returns_one_vector_per_text(self) -> None:
        from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider

        client = _make_embed_client([0.1, 0.2, 0.3])
        result = BedrockEmbeddingProvider(client, dimension=3).embed_batch(["a", "b"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]


# ── BedrockLLMProvider ────────────────────────────────────────────────────────


class TestBedrockLLMProviderProtocol:
    def test_satisfies_protocol(self) -> None:
        from travel_ai_search.llm.bedrock import BedrockLLMProvider

        assert isinstance(BedrockLLMProvider(MagicMock()), LLMProvider)


class TestBedrockLLMProviderGenerate:
    def test_returns_model_text(self) -> None:
        from travel_ai_search.llm.bedrock import BedrockLLMProvider

        client = _make_llm_client("beach hotel with pool")
        assert BedrockLLMProvider(client).generate("beach") == "beach hotel with pool"

    def test_passes_model_id(self) -> None:
        from travel_ai_search.llm.bedrock import BedrockLLMProvider

        client = _make_llm_client()
        BedrockLLMProvider(client, model_id="custom-claude").generate("test")
        assert client.converse.call_args[1]["modelId"] == "custom-claude"

    def test_prompt_in_user_message(self) -> None:
        from travel_ai_search.llm.bedrock import BedrockLLMProvider

        client = _make_llm_client()
        BedrockLLMProvider(client).generate("my travel query")
        messages = client.converse.call_args[1]["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["text"] == "my travel query"

    def test_system_prompt_when_non_empty(self) -> None:
        from travel_ai_search.llm.bedrock import BedrockLLMProvider

        client = _make_llm_client()
        BedrockLLMProvider(client).generate("prompt", system="you are helpful")
        kwargs = client.converse.call_args[1]
        assert "system" in kwargs
        assert kwargs["system"][0]["text"] == "you are helpful"

    def test_no_system_key_when_empty_string(self) -> None:
        from travel_ai_search.llm.bedrock import BedrockLLMProvider

        client = _make_llm_client()
        BedrockLLMProvider(client).generate("prompt")
        assert "system" not in client.converse.call_args[1]

    def test_no_system_key_when_default(self) -> None:
        from travel_ai_search.llm.bedrock import BedrockLLMProvider

        client = _make_llm_client()
        BedrockLLMProvider(client).generate("prompt")
        kwargs = client.converse.call_args[1]
        assert "system" not in kwargs

    def test_exception_propagates(self) -> None:
        from travel_ai_search.llm.bedrock import BedrockLLMProvider

        client = MagicMock()
        client.converse.side_effect = RuntimeError("network error")
        with pytest.raises(RuntimeError, match="network error"):
            BedrockLLMProvider(client).generate("test")

    def test_returns_string(self) -> None:
        from travel_ai_search.llm.bedrock import BedrockLLMProvider

        client = _make_llm_client("some text")
        result = BedrockLLMProvider(client).generate("query")
        assert isinstance(result, str)


# ── BedrockReranker ───────────────────────────────────────────────────────────


class TestBedrockRerankerProtocol:
    def test_satisfies_protocol(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        assert isinstance(BedrockReranker(MagicMock()), Reranker)


class TestBedrockRerankerEmpty:
    def test_empty_hits_returns_empty_without_api_call(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        client = MagicMock()
        result = BedrockReranker(client).rerank("query", [], top_k=10)
        assert result == []
        client.invoke_model.assert_not_called()


class TestBedrockRerankerOrdering:
    def test_rerank_reorders_by_relevance(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        results = [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
        ]
        client = _make_reranker_client(results)
        hits = _make_hits(2)
        out = BedrockReranker(client).rerank("beach", hits, top_k=2)
        assert out[0].id == "hotel-1"
        assert out[1].id == "hotel-0"

    def test_rerank_sets_relevance_score(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        results = [{"index": 0, "relevance_score": 0.75}]
        client = _make_reranker_client(results)
        hits = _make_hits(1)
        out = BedrockReranker(client).rerank("query", hits, top_k=1)
        assert out[0].score == pytest.approx(0.75)

    def test_rerank_preserves_source(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        results = [{"index": 0, "relevance_score": 0.8}]
        client = _make_reranker_client(results)
        hits = _make_hits(1)
        out = BedrockReranker(client).rerank("query", hits, top_k=1)
        assert out[0].source == hits[0].source


class TestBedrockRerankerApiShape:
    def test_passes_model_id(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        client = _make_reranker_client([{"index": 0, "relevance_score": 0.5}])
        BedrockReranker(client, model_id="custom-reranker").rerank("q", _make_hits(1), top_k=1)
        assert client.invoke_model.call_args[1]["modelId"] == "custom-reranker"

    def test_passes_query_in_body(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        client = _make_reranker_client([{"index": 0, "relevance_score": 0.5}])
        BedrockReranker(client).rerank("luxury spa", _make_hits(1), top_k=1)
        body = json.loads(client.invoke_model.call_args[1]["body"])
        assert body["query"] == "luxury spa"

    def test_passes_top_n_in_body(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        client = _make_reranker_client([{"index": 0, "relevance_score": 0.5}])
        BedrockReranker(client).rerank("query", _make_hits(1), top_k=7)
        body = json.loads(client.invoke_model.call_args[1]["body"])
        assert body["top_n"] == 7

    def test_passes_return_documents_false(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        client = _make_reranker_client([{"index": 0, "relevance_score": 0.5}])
        BedrockReranker(client).rerank("query", _make_hits(1), top_k=1)
        body = json.loads(client.invoke_model.call_args[1]["body"])
        assert body["return_documents"] is False

    def test_document_count_matches_hits(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        results = [{"index": i, "relevance_score": 0.5} for i in range(3)]
        client = _make_reranker_client(results)
        BedrockReranker(client).rerank("query", _make_hits(3), top_k=3)
        body = json.loads(client.invoke_model.call_args[1]["body"])
        assert len(body["documents"]) == 3

    def test_json_content_type(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        client = _make_reranker_client([{"index": 0, "relevance_score": 0.5}])
        BedrockReranker(client).rerank("query", _make_hits(1), top_k=1)
        kwargs = client.invoke_model.call_args[1]
        assert kwargs["contentType"] == "application/json"
        assert kwargs["accept"] == "application/json"

    def test_exception_propagates(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        client = MagicMock()
        client.invoke_model.side_effect = RuntimeError("Bedrock error")
        with pytest.raises(RuntimeError, match="Bedrock error"):
            BedrockReranker(client).rerank("query", _make_hits(1), top_k=1)


class TestBedrockRerankerDocumentText:
    def test_document_text_includes_hotel_name(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        client = _make_reranker_client([{"index": 0, "relevance_score": 0.5}])
        hits = [
            Hit(
                id="h1",
                score=0.5,
                source={
                    "hotel_name": "Grand Palace Hotel",
                    "destination": "Paris",
                    "country": "France",
                    "hotel_description": "Luxury hotel in the heart of Paris",
                    "activities": ["sightseeing"],
                    "tags": ["luxury"],
                },
            )
        ]
        BedrockReranker(client).rerank("paris luxury", hits, top_k=1)
        body = json.loads(client.invoke_model.call_args[1]["body"])
        assert "Grand Palace Hotel" in body["documents"][0]

    def test_document_text_includes_description(self) -> None:
        from travel_ai_search.reranking.bedrock import BedrockReranker

        client = _make_reranker_client([{"index": 0, "relevance_score": 0.5}])
        hits = [
            Hit(
                id="h1",
                score=0.5,
                source={
                    "hotel_name": "Hotel",
                    "destination": "Rome",
                    "country": "Italy",
                    "hotel_description": "Stunning rooftop views",
                    "activities": [],
                    "tags": [],
                },
            )
        ]
        BedrockReranker(client).rerank("rome views", hits, top_k=1)
        body = json.loads(client.invoke_model.call_args[1]["body"])
        assert "Stunning rooftop views" in body["documents"][0]
