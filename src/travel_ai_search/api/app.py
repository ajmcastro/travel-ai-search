"""FastAPI application entry point.

Resources created once at startup and stored on app.state:
  os_client          — shared OpenSearch connection pool
  embedding_provider — sentence-transformers model (loaded once; ~1-2 s on first run)

All route handlers receive these via dependency functions in deps.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from travel_ai_search.api.routes import search as search_router
from travel_ai_search.config.settings import Settings, get_settings
from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.infrastructure.opensearch import create_client


def _create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory — separated from lifespan so tests can patch it cheaply."""
    return LocalEmbeddingProvider(settings.embedding_model_name)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.os_client = create_client(settings)
    app.state.embedding_provider = _create_embedding_provider(settings)
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
