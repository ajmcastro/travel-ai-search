"""Shared fixtures for unit and integration tests."""

import pytest
from opensearchpy import OpenSearch

from travel_ai_search.config.settings import Settings
from travel_ai_search.infrastructure.opensearch import create_client


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def opensearch_client(settings: Settings) -> OpenSearch:
    """Return a connected OpenSearch client, or skip the test if unavailable."""
    client = create_client(settings)
    try:
        client.info()
    except Exception:
        pytest.skip("OpenSearch is not reachable — start it with: docker compose up -d")
    return client
