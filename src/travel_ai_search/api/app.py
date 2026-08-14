"""FastAPI application entry point.

Resources created once at startup and stored on app.state:
  os_client                  — shared OpenSearch connection pool
  embedding_provider         — sentence-transformers model (loaded once; ~1-2 s on first run)
  reranker                   — cross-encoder model (None when reranking_enabled=False)
  query_understanding_engine — rule-based QU engine (always created; pure Python, no I/O)
  query_rewriter             — LLM-based query rewriter (None when query_rewriting_enabled=False)

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
from travel_ai_search.query_understanding.base import QueryUnderstandingEngine
from travel_ai_search.query_understanding.extractor import RuleBasedQueryUnderstandingEngine
from travel_ai_search.query_understanding.rewriter import QueryRewriter
from travel_ai_search.reranking.base import Reranker

logger = logging.getLogger(__name__)


def _create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory — separated from lifespan so tests can patch it cheaply."""
    return LocalEmbeddingProvider(settings.embedding_model_name)


def _create_query_understanding_engine(settings: Settings) -> QueryUnderstandingEngine:
    """Factory — separated from lifespan so tests can patch it cheaply."""
    return RuleBasedQueryUnderstandingEngine()


def _create_query_rewriter(settings: Settings) -> QueryRewriter | None:
    """Factory — returns None when query rewriting is disabled.

    Graceful degradation: a failed provider load logs a warning and the system
    continues without rewriting.  The llm_provider setting selects the backend:
    'local' (keyword expansion, no deps), 'echo' (identity stub, for testing),
    'bedrock' (Milestone 12).
    """
    if not settings.query_rewriting_enabled:
        return None
    try:
        from travel_ai_search.llm.local import EchoLLMProvider, LocalLLMProvider

        llm = EchoLLMProvider() if settings.llm_provider == "echo" else LocalLLMProvider()
        return QueryRewriter(llm)
    except Exception as exc:
        logger.warning(
            "Failed to create query rewriter ('%s'): %s — rewriting disabled.",
            settings.llm_provider,
            exc,
        )
        return None


def _create_reranker(settings: Settings) -> Reranker | None:
    """Factory — returns None when reranking is disabled or the model fails to load.

    Graceful degradation: a failed reranker load logs a warning and the system
    continues without reranking rather than refusing to start.
    """
    if not settings.reranking_enabled:
        return None
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
