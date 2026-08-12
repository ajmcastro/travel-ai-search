"""Unit tests for the ingestion module.

Tests here do NOT require a running OpenSearch — they verify:
  - INDEX_BODY mapping structure (pure dict inspection)
  - to_document() serialisation logic
  - load_products() JSONL round-trip
"""

from pathlib import Path

import pytest

from travel_ai_search.domain.models import TravelProduct
from travel_ai_search.ingestion.index import INDEX_BODY
from travel_ai_search.ingestion.ingestor import load_products, to_document

# ── Helpers ───────────────────────────────────────────────────────────────────


def _valid_product(**overrides: object) -> TravelProduct:
    defaults: dict[str, object] = {
        "id": "hotel_000001",
        "hotel_name": "Azure Sands Resort",
        "hotel_description": "A bright family resort on a wide sandy beach.",
        "destination": "Albufeira",
        "region": "Algarve",
        "country": "Portugal",
        "latitude": 37.09,
        "longitude": -8.25,
        "star_rating": 4,
        "customer_rating": 8.3,
        "family_friendly": True,
        "adults_only": False,
        "amenities": ["outdoor pool", "kids' club"],
        "board_types": ["half_board"],
        "beach_distance_km": 0.1,
        "airport_distance_km": 22.5,
        "activities": ["swimming", "snorkelling"],
        "tags": ["beach", "family"],
        "available_departure_airports": ["LGW", "MAN"],
        "price_per_person_gbp": 749.0,
        "available_months": ["June", "July", "August"],
        "climate_zone": "Mediterranean",
        "peak_season_months": ["July", "August"],
    }
    defaults.update(overrides)
    return TravelProduct(**defaults)  # type: ignore[arg-type]


# ── INDEX_BODY mapping structure ──────────────────────────────────────────────


def _props() -> dict:  # type: ignore[type-arg]
    return INDEX_BODY["mappings"]["properties"]  # type: ignore[index]


def test_index_has_mappings_key() -> None:
    assert "mappings" in INDEX_BODY
    assert "properties" in INDEX_BODY["mappings"]


def test_hotel_name_is_text_with_english_analyser() -> None:
    field = _props()["hotel_name"]
    assert field["type"] == "text"
    assert field["analyzer"] == "english"


def test_hotel_name_has_keyword_subfield() -> None:
    assert _props()["hotel_name"]["fields"]["keyword"]["type"] == "keyword"


def test_hotel_description_is_text() -> None:
    assert _props()["hotel_description"]["type"] == "text"


def test_destination_is_keyword_with_text_subfield() -> None:
    field = _props()["destination"]
    assert field["type"] == "keyword"
    assert field["fields"]["text"]["type"] == "text"


def test_country_is_keyword() -> None:
    assert _props()["country"]["type"] == "keyword"


def test_star_rating_is_integer() -> None:
    assert _props()["star_rating"]["type"] == "integer"


def test_price_is_float() -> None:
    assert _props()["price_per_person_gbp"]["type"] == "float"


def test_family_friendly_is_boolean() -> None:
    assert _props()["family_friendly"]["type"] == "boolean"


def test_adults_only_is_boolean() -> None:
    assert _props()["adults_only"]["type"] == "boolean"


def test_location_is_geo_point() -> None:
    assert _props()["location"]["type"] == "geo_point"


def test_board_types_is_keyword() -> None:
    assert _props()["board_types"]["type"] == "keyword"


def test_available_months_is_keyword() -> None:
    assert _props()["available_months"]["type"] == "keyword"


def test_activities_is_text_with_keyword_subfield() -> None:
    field = _props()["activities"]
    assert field["type"] == "text"
    assert field["fields"]["keyword"]["type"] == "keyword"


def test_settings_have_single_shard() -> None:
    assert INDEX_BODY["settings"]["number_of_shards"] == 1  # type: ignore[index]


# ── to_document ───────────────────────────────────────────────────────────────


def test_to_document_adds_location_geo_point() -> None:
    p = _valid_product()
    doc = to_document(p)
    assert doc["location"] == {"lat": 37.09, "lon": -8.25}


def test_to_document_location_lat_equals_latitude() -> None:
    p = _valid_product(latitude=28.04, longitude=-16.74)
    doc = to_document(p)
    assert doc["location"]["lat"] == pytest.approx(28.04)
    assert doc["location"]["lon"] == pytest.approx(-16.74)


def test_to_document_preserves_id() -> None:
    doc = to_document(_valid_product())
    assert doc["id"] == "hotel_000001"


def test_to_document_preserves_hotel_name() -> None:
    doc = to_document(_valid_product())
    assert doc["hotel_name"] == "Azure Sands Resort"


def test_to_document_preserves_list_fields() -> None:
    doc = to_document(_valid_product())
    assert "outdoor pool" in doc["amenities"]
    assert "half_board" in doc["board_types"]


def test_to_document_preserves_boolean_fields() -> None:
    doc = to_document(_valid_product(family_friendly=True, adults_only=False))
    assert doc["family_friendly"] is True
    assert doc["adults_only"] is False


# ── load_products ─────────────────────────────────────────────────────────────


def test_load_products_reads_single_product(tmp_path: Path) -> None:
    p = _valid_product()
    jsonl = tmp_path / "hotels.jsonl"
    jsonl.write_text(p.model_dump_json() + "\n")
    products = load_products(jsonl)
    assert len(products) == 1
    assert products[0].id == p.id
    assert products[0].hotel_name == p.hotel_name


def test_load_products_reads_multiple_products(tmp_path: Path) -> None:
    products_written = [_valid_product(id=f"hotel_{i:06d}") for i in range(1, 6)]
    jsonl = tmp_path / "hotels.jsonl"
    jsonl.write_text("\n".join(p.model_dump_json() for p in products_written) + "\n")
    loaded = load_products(jsonl)
    assert len(loaded) == 5
    assert [p.id for p in loaded] == [p.id for p in products_written]


def test_load_products_skips_blank_lines(tmp_path: Path) -> None:
    p = _valid_product()
    jsonl = tmp_path / "hotels.jsonl"
    jsonl.write_text(f"\n{p.model_dump_json()}\n\n")
    products = load_products(jsonl)
    assert len(products) == 1


def test_load_products_empty_file(tmp_path: Path) -> None:
    jsonl = tmp_path / "hotels.jsonl"
    jsonl.write_text("")
    products = load_products(jsonl)
    assert products == []
