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


def get_os_client(request: Request) -> OpenSearch:
    """Return the shared OpenSearch client stored on app state at startup."""
    client: OpenSearch = request.app.state.os_client
    return client


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    """Return the shared embedding provider stored on app state at startup."""
    provider: EmbeddingProvider = request.app.state.embedding_provider
    return provider


def get_settings() -> Settings:
    """Return the application settings singleton."""
    return _get_settings()
