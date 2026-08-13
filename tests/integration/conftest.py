"""Shared fixtures for integration tests that require indexed search data.

The curated hotel set is small and deliberate: each property has known
characteristics so filter-correctness tests can make precise assertions
without guessing BM25 rankings.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk  # type: ignore[import-untyped]

from travel_ai_search.domain.models import TravelProduct, build_embedding_text
from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.ingestion.index import INDEX_BODY
from travel_ai_search.ingestion.ingestor import ingest

_LEXICAL_TEST_INDEX = "travel_hotels_lexical_test"
_VECTOR_TEST_INDEX = "travel_hotels_vector_test"

# ── Curated test dataset ──────────────────────────────────────────────────────
# Designed so that filter tests can assert exact document counts without
# relying on BM25 score ordering.
#
# By country:   Spain=3 (001,002,005)  Portugal=1 (004)  Greece=1 (003)  Morocco=1 (006)
# family=True:  001, 004, 006  (3)
# adults=True:  002           (1)
# price ≤ 700:  001 (£699), 005 (£649), 006 (£449)  (3)
# stars = 5:    002           (1)
# month=July:   001, 002, 003, 004, 005  (5)
# airport=MAN:  001, 003, 004, 006  (4)

_CURATED_HOTELS: list[TravelProduct] = [
    TravelProduct(
        id="test_hotel_001",
        hotel_name="Sunny Beach Family Resort",
        hotel_description=(
            "A vibrant family-friendly beach resort with kids clubs, "
            "waterslides, and direct access to a wide sandy beach."
        ),
        destination="Benidorm",
        region="Costa Blanca",
        country="Spain",
        latitude=38.54,
        longitude=-0.13,
        star_rating=3,
        customer_rating=7.2,
        family_friendly=True,
        adults_only=False,
        amenities=["outdoor pool", "kids' club", "waterslide", "restaurant"],
        board_types=["half_board"],
        beach_distance_km=0.2,
        airport_distance_km=55.0,
        activities=["swimming", "beach volleyball", "kids' club"],
        tags=["family", "beach", "pool"],
        available_departure_airports=["MAN", "LGW"],
        price_per_person_gbp=699.0,
        available_months=["June", "July", "August", "September"],
        climate_zone="Mediterranean",
        peak_season_months=["July", "August"],
    ),
    TravelProduct(
        id="test_hotel_002",
        hotel_name="La Vie Luxury Adults Retreat",
        hotel_description=(
            "An exclusive adults-only sanctuary with Michelin-starred dining, "
            "an infinity pool overlooking the sea, and a world-class thalassotherapy spa."
        ),
        destination="Marbella",
        region="Costa del Sol",
        country="Spain",
        latitude=36.52,
        longitude=-4.91,
        star_rating=5,
        customer_rating=9.2,
        family_friendly=False,
        adults_only=True,
        amenities=["infinity pool", "spa", "fine dining", "butler service"],
        board_types=["all_inclusive"],
        beach_distance_km=0.5,
        airport_distance_km=80.0,
        activities=["spa", "golf", "wine tasting"],
        tags=["luxury", "adults_only", "spa", "fine dining"],
        available_departure_airports=["LGW", "LHR"],
        price_per_person_gbp=2499.0,
        available_months=["April", "May", "June", "July", "August", "September", "October"],
        climate_zone="Mediterranean",
        peak_season_months=["June", "July", "August"],
    ),
    TravelProduct(
        id="test_hotel_003",
        hotel_name="Mykonos Party and Beach Hotel",
        hotel_description=(
            "A lively beachfront hotel in the heart of Mykonos nightlife scene, "
            "with a rooftop bar, beach club, and direct access to a Blue Flag beach."
        ),
        destination="Mykonos",
        region="Cyclades",
        country="Greece",
        latitude=37.44,
        longitude=25.33,
        star_rating=4,
        customer_rating=8.1,
        family_friendly=False,
        adults_only=False,
        amenities=["rooftop bar", "infinity pool", "beach club"],
        board_types=["bed_and_breakfast"],
        beach_distance_km=0.1,
        airport_distance_km=4.0,
        activities=["nightlife", "beach", "water sports"],
        tags=["nightlife", "beach", "party"],
        available_departure_airports=["LGW", "MAN"],
        price_per_person_gbp=1199.0,
        available_months=["May", "June", "July", "August", "September"],
        climate_zone="Mediterranean",
        peak_season_months=["July", "August"],
    ),
    TravelProduct(
        id="test_hotel_004",
        hotel_name="Algarve Ocean Family Resort",
        hotel_description=(
            "A family resort with direct beach access, a dedicated children's "
            "programme, and half-board and all-inclusive options."
        ),
        destination="Albufeira",
        region="Algarve",
        country="Portugal",
        latitude=37.09,
        longitude=-8.25,
        star_rating=4,
        customer_rating=8.5,
        family_friendly=True,
        adults_only=False,
        amenities=["outdoor pool", "kids' club", "restaurant"],
        board_types=["half_board", "all_inclusive"],
        beach_distance_km=0.05,
        airport_distance_km=40.0,
        activities=["swimming", "snorkelling", "kids' club"],
        tags=["family", "beach", "ocean"],
        available_departure_airports=["LGW", "MAN", "BHX"],
        price_per_person_gbp=849.0,
        available_months=["June", "July", "August", "September"],
        climate_zone="Mediterranean",
        peak_season_months=["July", "August"],
    ),
    TravelProduct(
        id="test_hotel_005",
        hotel_name="Menorca Quiet Retreat",
        hotel_description=(
            "A peaceful boutique hotel set in unspoiled countryside, "
            "ideal for guests seeking tranquility, nature walks, and total relaxation."
        ),
        destination="Mahón",
        region="Menorca",
        country="Spain",
        latitude=39.88,
        longitude=4.29,
        star_rating=4,
        customer_rating=8.8,
        family_friendly=False,
        adults_only=False,
        amenities=["outdoor pool", "garden", "restaurant"],
        board_types=["bed_and_breakfast", "half_board"],
        beach_distance_km=1.5,
        airport_distance_km=5.0,
        activities=["hiking", "cycling", "yoga"],
        tags=["quiet", "nature", "peaceful", "boutique"],
        available_departure_airports=["LGW", "MAN"],
        price_per_person_gbp=649.0,
        available_months=["May", "June", "July", "August", "September"],
        climate_zone="Mediterranean",
        peak_season_months=["July", "August"],
    ),
    TravelProduct(
        id="test_hotel_006",
        hotel_name="Marrakech Heritage Riad",
        hotel_description=(
            "An authentic riad within the historic medina, offering traditional "
            "Moroccan architecture, a rooftop terrace, and guided cultural tours."
        ),
        destination="Marrakech",
        region="Marrakech-Safi",
        country="Morocco",
        latitude=31.63,
        longitude=-7.99,
        star_rating=3,
        customer_rating=8.0,
        family_friendly=True,
        adults_only=False,
        amenities=["pool", "rooftop terrace", "restaurant"],
        board_types=["bed_and_breakfast"],
        beach_distance_km=250.0,
        airport_distance_km=6.0,
        activities=["cultural tours", "cooking class", "souk exploration"],
        tags=["culture", "heritage", "authentic", "medina"],
        available_departure_airports=["LGW", "MAN", "LHR"],
        price_per_person_gbp=449.0,
        available_months=["March", "April", "October", "November"],
        climate_zone="Subtropical",
        peak_season_months=["March", "April"],
    ),
]


@pytest.fixture(scope="session")
def lexical_test_index(opensearch_client: OpenSearch) -> Iterator[str]:
    """Session-scoped test index pre-loaded with curated hotels.

    Uses a unique name to avoid conflicting with other integration test fixtures.
    Shared across all tests in the session — tests must only READ, never write.
    """
    if opensearch_client.indices.exists(index=_LEXICAL_TEST_INDEX):
        opensearch_client.indices.delete(index=_LEXICAL_TEST_INDEX)
    opensearch_client.indices.create(index=_LEXICAL_TEST_INDEX, body=INDEX_BODY)
    ingest(opensearch_client, _CURATED_HOTELS, index=_LEXICAL_TEST_INDEX)
    yield _LEXICAL_TEST_INDEX
    if opensearch_client.indices.exists(index=_LEXICAL_TEST_INDEX):
        opensearch_client.indices.delete(index=_LEXICAL_TEST_INDEX)


@pytest.fixture(scope="session")
def embedding_provider() -> LocalEmbeddingProvider:
    """Session-scoped LocalEmbeddingProvider.  Model is loaded once per session."""
    return LocalEmbeddingProvider()


@pytest.fixture(scope="session")
def vector_test_index(
    opensearch_client: OpenSearch,
    embedding_provider: LocalEmbeddingProvider,
) -> Iterator[str]:
    """Session-scoped knn index pre-loaded with curated hotels AND their embeddings.

    Build steps:
      1. Create the index (knn=True mapping, identical to production).
      2. Ingest the curated hotels without embeddings (plain document fields).
      3. Generate real embeddings for each hotel and update via bulk update.
      4. Refresh so the HNSW graph is built and queries return results.

    Tests must only READ; they must not write to this index.
    """
    if opensearch_client.indices.exists(index=_VECTOR_TEST_INDEX):
        opensearch_client.indices.delete(index=_VECTOR_TEST_INDEX)
    opensearch_client.indices.create(index=_VECTOR_TEST_INDEX, body=INDEX_BODY)
    ingest(opensearch_client, _CURATED_HOTELS, index=_VECTOR_TEST_INDEX)

    texts = [build_embedding_text(h) for h in _CURATED_HOTELS]
    vectors = embedding_provider.embed_batch(texts)
    actions = [
        {
            "_op_type": "update",
            "_index": _VECTOR_TEST_INDEX,
            "_id": hotel.id,
            "doc": {"embedding_vector": vector},
        }
        for hotel, vector in zip(_CURATED_HOTELS, vectors)
    ]
    bulk(opensearch_client, actions, raise_on_error=True)
    opensearch_client.indices.refresh(index=_VECTOR_TEST_INDEX)

    yield _VECTOR_TEST_INDEX

    if opensearch_client.indices.exists(index=_VECTOR_TEST_INDEX):
        opensearch_client.indices.delete(index=_VECTOR_TEST_INDEX)
