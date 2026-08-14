"""FastAPI application entry point.

Resources created once at startup and stored on app.state:
  os_client                  — shared OpenSearch connection pool
  embedding_provider         — sentence-transformers model (loaded once; ~1-2 s on first run)
  reranker                   — cross-encoder model (None when reranking_enabled=False)
  query_understanding_engine — rule-based QU engine (always created; pure Python, no I/O)
  query_rewriter             — LLM-based query rewriter (None when query_rewriting_enabled=False)
  query_expander             — query expander for multi-query retrieval (None when disabled)

All route handlers receive these via dependency functions in deps.py.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from travel_ai_search.api.routes import query as query_router
from travel_ai_search.api.routes import search as search_router
from travel_ai_search.config.settings import Settings, get_settings
from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.llm.base import LLMProvider
from travel_ai_search.query_understanding.base import QueryUnderstandingEngine
from travel_ai_search.query_understanding.extractor import RuleBasedQueryUnderstandingEngine
from travel_ai_search.query_understanding.rewriter import QueryRewriter
from travel_ai_search.reranking.base import Reranker

logger = logging.getLogger(__name__)


def _create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory — separated from lifespan so tests can patch it cheaply.

    Graceful degradation: if the Bedrock provider fails to initialise (missing
    credentials, unavailable region, etc.) log a warning and fall back to local.
    """
    if settings.embedding_provider == "bedrock":
        try:
            from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider
            from travel_ai_search.infrastructure.bedrock import create_bedrock_client

            client = create_bedrock_client(settings.aws_region)
            logger.info(
                "Bedrock embedding provider: %s (dim=%d)",
                settings.bedrock_embedding_model_id,
                settings.bedrock_embedding_dimension,
            )
            return BedrockEmbeddingProvider(
                client,
                model_id=settings.bedrock_embedding_model_id,
                dimension=settings.bedrock_embedding_dimension,
            )
        except Exception as exc:
            logger.warning(
                "Failed to create Bedrock embedding provider: %s — falling back to local.",
                exc,
            )
    return LocalEmbeddingProvider(settings.embedding_model_name)


def _create_query_understanding_engine(settings: Settings) -> QueryUnderstandingEngine:
    """Factory — separated from lifespan so tests can patch it cheaply."""
    return RuleBasedQueryUnderstandingEngine()


def _create_query_rewriter(settings: Settings) -> QueryRewriter | None:
    """Factory — returns None when query rewriting is disabled.

    Graceful degradation: a failed provider load logs a warning and the system
    continues without rewriting.  The llm_provider setting selects the backend:
    'local' (keyword expansion, no deps), 'echo' (identity stub, for testing),
    'bedrock' (Claude via Converse API, requires AWS credentials).
    """
    if not settings.query_rewriting_enabled:
        return None
    try:
        llm: LLMProvider
        if settings.llm_provider == "bedrock":
            from travel_ai_search.infrastructure.bedrock import create_bedrock_client
            from travel_ai_search.llm.bedrock import BedrockLLMProvider

            bedrock_client = create_bedrock_client(settings.aws_region)
            llm = BedrockLLMProvider(bedrock_client, model_id=settings.bedrock_llm_model_id)
            logger.info("Bedrock LLM provider: %s", settings.bedrock_llm_model_id)
        elif settings.llm_provider == "echo":
            from travel_ai_search.llm.local import EchoLLMProvider

            llm = EchoLLMProvider()
        else:
            from travel_ai_search.llm.local import LocalLLMProvider

            llm = LocalLLMProvider()
        return QueryRewriter(llm)
    except Exception as exc:
        logger.warning(
            "Failed to create query rewriter ('%s'): %s — rewriting disabled.",
            settings.llm_provider,
            exc,
        )
        return None


def _create_query_expander(settings: Settings) -> object | None:
    """Factory — returns None when query expansion is disabled.

    Graceful degradation: a failed expander load logs a warning and the system
    continues without expansion.  LocalQueryExpander is pure Python with no
    external dependencies, so failure is unlikely in practice.
    """
    if not settings.query_expansion_enabled:
        return None
    try:
        from travel_ai_search.query_understanding.expander import LocalQueryExpander

        return LocalQueryExpander()
    except Exception as exc:
        logger.warning(
            "Failed to create query expander: %s — expansion disabled.",
            exc,
        )
        return None


def _create_reranker(settings: Settings) -> Reranker | None:
    """Factory — returns None when reranking is disabled or the model fails to load.

    Graceful degradation: a failed reranker load logs a warning and the system
    continues without reranking rather than refusing to start.  If the Bedrock
    reranker fails, falls back to the local cross-encoder.
    """
    if not settings.reranking_enabled:
        return None
    if settings.reranker_provider == "bedrock":
        try:
            from travel_ai_search.infrastructure.bedrock import create_bedrock_client
            from travel_ai_search.reranking.bedrock import BedrockReranker

            client = create_bedrock_client(settings.aws_region)
            logger.info("Bedrock reranker: %s", settings.bedrock_reranker_model_id)
            return BedrockReranker(client, model_id=settings.bedrock_reranker_model_id)
        except Exception as exc:
            logger.warning(
                "Failed to create Bedrock reranker: %s — falling back to local.",
                exc,
            )
    try:
        from travel_ai_search.reranking.local import LocalCrossEncoderReranker

        return LocalCrossEncoderReranker(settings.reranker_model_name)
    except Exception as exc:
        logger.warning(
            "Failed to load reranker '%s': %s — reranking disabled.",
            settings.reranker_model_name,
            exc,
        )
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.os_client = create_client(settings)
    app.state.embedding_provider = _create_embedding_provider(settings)
    app.state.reranker = _create_reranker(settings)
    app.state.query_understanding_engine = _create_query_understanding_engine(settings)
    app.state.query_rewriter = _create_query_rewriter(settings)
    app.state.query_expander = _create_query_expander(settings)
    yield
    app.state.os_client.close()


app = FastAPI(
    title="Travel AI Search",
    description="Educational AI search system for travel products",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(search_router.router, prefix="/search")
app.include_router(query_router.router, prefix="/query")


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}
