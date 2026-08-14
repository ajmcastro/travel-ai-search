"""Unit test fixtures.

Patches infrastructure-level dependencies (OpenSearch, embedding model,
reranker) so unit tests run without any external services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_embedding_provider() -> MagicMock:
    """Patch startup factories so TestClient doesn't load real models.

    The sentence-transformers embedding model (~80 MB) and the cross-encoder
    reranker (~86 MB) each take 1-2 s to load — that cost belongs in
    integration tests, not unit tests.  Any test that needs a real model
    should be marked @pytest.mark.integration.
    """
    mock = MagicMock()
    mock.dimension = 384
    mock.embed.return_value = [0.1] * 384
    mock.embed_batch.return_value = [[0.1] * 384]

    with (
        patch(
            "travel_ai_search.api.app._create_embedding_provider",
            return_value=mock,
        ),
        patch(
            "travel_ai_search.api.app._create_reranker",
            return_value=None,
        ),
    ):
        yield mock
