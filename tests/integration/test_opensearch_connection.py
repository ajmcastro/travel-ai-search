"""Integration tests for OpenSearch connectivity.

Requires OpenSearch to be running:
    docker compose up -d

Run with:
    uv run pytest tests/integration
"""

import pytest
from opensearchpy import OpenSearch


@pytest.mark.integration
def test_cluster_is_reachable(opensearch_client: OpenSearch) -> None:
    info = opensearch_client.info()
    assert "version" in info


@pytest.mark.integration
def test_opensearch_version_is_2x(opensearch_client: OpenSearch) -> None:
    info = opensearch_client.info()
    version: str = info["version"]["number"]
    assert version.startswith("2."), f"Expected OpenSearch 2.x, got {version}"


@pytest.mark.integration
def test_cluster_health_is_acceptable(opensearch_client: OpenSearch) -> None:
    health = opensearch_client.cluster.health()
    assert health["status"] in ("green", "yellow"), (
        f"Cluster status is '{health['status']}' — expected green or yellow"
    )


@pytest.mark.integration
def test_cluster_has_at_least_one_node(opensearch_client: OpenSearch) -> None:
    health = opensearch_client.cluster.health()
    assert health["number_of_nodes"] >= 1
