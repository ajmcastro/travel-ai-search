"""FastAPI dependency functions.

Each function is intended to be used with Depends() in route handlers.
Using Depends rather than direct imports keeps route handlers testable —
tests can override any dependency via app.dependency_overrides.
"""

from __future__ import annotations

from fastapi import Request
from opensearchpy import OpenSearch

from travel_ai_search.config.settings import Settings
from travel_ai_search.config.settings import get_settings as _get_settings
from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.query_understanding.base import QueryUnderstandingEngine
from travel_ai_search.query_understanding.rewriter import QueryRewriter
from travel_ai_search.reranking.base import Reranker


def get_os_client(request: Request) -> OpenSearch:
    """Return the shared OpenSearch client stored on app state at startup."""
    client: OpenSearch = request.app.state.os_client
    return client


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    """Return the shared embedding provider stored on app state at startup."""
    provider: EmbeddingProvider = request.app.state.embedding_provider
    return provider


def get_reranker(request: Request) -> Reranker | None:
    """Return the shared reranker stored on app state, or None if not loaded.

    The reranker is None when reranking_enabled=False in settings or when
    the model failed to load at startup.  Route handlers that accept an
    optional reranker must pass None-safely to downstream functions.
    """
    return getattr(request.app.state, "reranker", None)


def get_query_understanding_engine(request: Request) -> QueryUnderstandingEngine:
    """Return the shared query understanding engine stored on app state at startup."""
    engine: QueryUnderstandingEngine = request.app.state.query_understanding_engine
    return engine


def get_query_rewriter(request: Request) -> QueryRewriter | None:
    """Return the shared query rewriter stored on app state, or None if not created.

    The rewriter is None when query_rewriting_enabled=False in settings.
    Route handlers that accept an optional rewriter must check for None before use.
    """
    return getattr(request.app.state, "query_rewriter", None)


def get_settings() -> Settings:
    """Return the application settings singleton."""
    return _get_settings()
