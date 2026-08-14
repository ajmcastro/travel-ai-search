"""FastAPI application entry point.

Resources created once at startup and stored on app.state:
  os_client          — shared OpenSearch connection pool
  embedding_provider — sentence-transformers model (loaded once; ~1-2 s on first run)
  reranker           — cross-encoder model (None when reranking_enabled=False)

All route handlers receive these via dependency functions in deps.py.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from travel_ai_search.api.routes import search as search_router
from travel_ai_search.config.settings import Settings, get_settings
from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.reranking.base import Reranker

logger = logging.getLogger(__name__)


def _create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory — separated from lifespan so tests can patch it cheaply."""
    return LocalEmbeddingProvider(settings.embedding_model_name)


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
    yield
    app.state.os_client.close()


app = FastAPI(
    title="Travel AI Search",
    description="Educational AI search system for travel products",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(search_router.router, prefix="/search")


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}
