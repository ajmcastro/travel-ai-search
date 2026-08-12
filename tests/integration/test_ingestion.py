"""Integration tests for index creation and data ingestion.

Requires OpenSearch to be running:
    docker compose up -d

Run with:
    make test-integration
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opensearchpy import OpenSearch

from travel_ai_search.domain.models import TravelProduct
from travel_ai_search.ingestion.index import INDEX_BODY, create_index, delete_index, index_exists
from travel_ai_search.ingestion.ingestor import IngestResult, ingest

# A dedicated test index keeps integration tests isolated from production data.
_TEST_INDEX = "travel_hotels_test"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_test_index(opensearch_client: OpenSearch) -> Iterator[str]:
    """Ensure the test index does NOT exist before the test; delete it after."""
    if opensearch_client.indices.exists(index=_TEST_INDEX):
        opensearch_client.indices.delete(index=_TEST_INDEX)
    yield _TEST_INDEX
    if opensearch_client.indices.exists(index=_TEST_INDEX):
        opensearch_client.indices.delete(index=_TEST_INDEX)


@pytest.fixture
def test_index(opensearch_client: OpenSearch) -> Iterator[str]:
    """Create a fresh test index with the production mapping; delete it after."""
    if opensearch_client.indices.exists(index=_TEST_INDEX):
        opensearch_client.indices.delete(index=_TEST_INDEX)
    opensearch_client.indices.create(index=_TEST_INDEX, body=INDEX_BODY)
    yield _TEST_INDEX
    if opensearch_client.indices.exists(index=_TEST_INDEX):
        opensearch_client.indices.delete(index=_TEST_INDEX)


@pytest.fixture
def sample_products() -> list[TravelProduct]:
    """Three minimal valid TravelProducts for ingestion tests."""
    return [
        TravelProduct(
            id=f"hotel_{i:06d}",
            hotel_name=f"Test Hotel {i}",
            hotel_description=f"A test hotel number {i} for integration tests.",
            destination="Albufeira",
            region="Algarve",
            country="Portugal",
            latitude=37.09,
            longitude=-8.25,
            star_rating=3,
            customer_rating=7.5,
            family_friendly=(i % 2 == 0),
            adults_only=(i % 2 != 0),
            amenities=["pool"],
            board_types=["bed_and_breakfast"],
            beach_distance_km=0.5,
            airport_distance_km=20.0,
            activities=["swimming"],
            tags=["beach"],
            available_departure_airports=["LGW"],
            price_per_person_gbp=500.0,
            available_months=["July"],
            climate_zone="Mediterranean",
            peak_season_months=["July"],
        )
        for i in range(1, 4)
    ]


# ── Index lifecycle ───────────────────────────────────────────────────────────


@pytest.mark.integration
def test_create_index_succeeds(opensearch_client: OpenSearch, clean_test_index: str) -> None:
    create_index(opensearch_client, index=clean_test_index)
    assert opensearch_client.indices.exists(index=clean_test_index)


@pytest.mark.integration
def test_create_index_is_idempotent(opensearch_client: OpenSearch, test_index: str) -> None:
    # Index already exists via fixture — calling again with recreate=False is a no-op
    create_index(opensearch_client, index=test_index, recreate=False)
    assert index_exists(opensearch_client, index=test_index)


@pytest.mark.integration
def test_delete_index_removes_index(opensearch_client: OpenSearch, test_index: str) -> None:
    delete_index(opensearch_client, index=test_index)
    assert not opensearch_client.indices.exists(index=test_index)


@pytest.mark.integration
def test_delete_index_on_missing_is_silent(opensearch_client: OpenSearch) -> None:
    delete_index(opensearch_client, index="nonexistent_index_xyz_abc")


@pytest.mark.integration
def test_index_exists_returns_false_when_missing(opensearch_client: OpenSearch) -> None:
    assert not index_exists(opensearch_client, index="nonexistent_index_xyz_abc")


@pytest.mark.integration
def test_index_exists_returns_true_after_creation(
    opensearch_client: OpenSearch, test_index: str
) -> None:
    assert index_exists(opensearch_client, index=test_index)


# ── Mapping verification ──────────────────────────────────────────────────────


@pytest.mark.integration
def test_mapping_hotel_name_is_text(opensearch_client: OpenSearch, test_index: str) -> None:
    mapping = opensearch_client.indices.get_mapping(index=test_index)
    props = mapping[test_index]["mappings"]["properties"]
    assert props["hotel_name"]["type"] == "text"


@pytest.mark.integration
def test_mapping_destination_is_keyword(opensearch_client: OpenSearch, test_index: str) -> None:
    mapping = opensearch_client.indices.get_mapping(index=test_index)
    props = mapping[test_index]["mappings"]["properties"]
    assert props["destination"]["type"] == "keyword"


@pytest.mark.integration
def test_mapping_location_is_geo_point(opensearch_client: OpenSearch, test_index: str) -> None:
    mapping = opensearch_client.indices.get_mapping(index=test_index)
    props = mapping[test_index]["mappings"]["properties"]
    assert props["location"]["type"] == "geo_point"


@pytest.mark.integration
def test_mapping_family_friendly_is_boolean(opensearch_client: OpenSearch, test_index: str) -> None:
    mapping = opensearch_client.indices.get_mapping(index=test_index)
    props = mapping[test_index]["mappings"]["properties"]
    assert props["family_friendly"]["type"] == "boolean"


# ── Ingestion ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_ingest_returns_correct_count(
    opensearch_client: OpenSearch,
    test_index: str,
    sample_products: list[TravelProduct],
) -> None:
    result: IngestResult = ingest(opensearch_client, sample_products, index=test_index)
    assert result.indexed == len(sample_products)
    assert result.errors == 0


@pytest.mark.integration
def test_ingest_document_count_matches(
    opensearch_client: OpenSearch,
    test_index: str,
    sample_products: list[TravelProduct],
) -> None:
    ingest(opensearch_client, sample_products, index=test_index)
    count = opensearch_client.count(index=test_index)["count"]
    assert count == len(sample_products)


@pytest.mark.integration
def test_ingest_document_retrievable_by_id(
    opensearch_client: OpenSearch,
    test_index: str,
    sample_products: list[TravelProduct],
) -> None:
    ingest(opensearch_client, sample_products, index=test_index)
    first = sample_products[0]
    doc = opensearch_client.get(index=test_index, id=first.id)
    assert doc["_source"]["hotel_name"] == first.hotel_name


@pytest.mark.integration
def test_ingest_document_has_geo_point(
    opensearch_client: OpenSearch,
    test_index: str,
    sample_products: list[TravelProduct],
) -> None:
    ingest(opensearch_client, sample_products, index=test_index)
    doc = opensearch_client.get(index=test_index, id=sample_products[0].id)
    location = doc["_source"]["location"]
    assert "lat" in location
    assert "lon" in location
    assert location["lat"] == pytest.approx(37.09)


@pytest.mark.integration
def test_ingest_is_idempotent(
    opensearch_client: OpenSearch,
    test_index: str,
    sample_products: list[TravelProduct],
) -> None:
    ingest(opensearch_client, sample_products, index=test_index)
    ingest(opensearch_client, sample_products, index=test_index)
    count = opensearch_client.count(index=test_index)["count"]
    assert count == len(sample_products)
