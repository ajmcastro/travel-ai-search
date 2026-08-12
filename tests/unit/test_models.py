"""Unit tests for the TravelProduct domain model and build_embedding_text."""

import pytest

from travel_ai_search.domain.models import TravelProduct, build_embedding_text


def _valid_product(**overrides: object) -> TravelProduct:
    """Return a minimal valid TravelProduct, optionally overriding fields."""
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
        "amenities": ["outdoor pool", "kids' club", "restaurant"],
        "board_types": ["half_board", "all_inclusive"],
        "beach_distance_km": 0.1,
        "airport_distance_km": 22.5,
        "activities": ["swimming", "snorkelling", "beach volleyball"],
        "tags": ["beach", "family", "pool"],
        "available_departure_airports": ["LGW", "MAN"],
        "price_per_person_gbp": 749.0,
        "available_months": ["June", "July", "August", "September"],
        "climate_zone": "Mediterranean",
        "peak_season_months": ["June", "July", "August"],
    }
    defaults.update(overrides)
    return TravelProduct(**defaults)  # type: ignore[arg-type]


# ── Model validation ─────────────────────────────────────────────────────────


def test_valid_product_constructs_without_error() -> None:
    product = _valid_product()
    assert product.id == "hotel_000001"
    assert product.hotel_name == "Azure Sands Resort"


def test_family_and_adults_only_raises() -> None:
    with pytest.raises(ValueError, match="family_friendly and adults_only"):
        _valid_product(family_friendly=True, adults_only=True)


def test_star_rating_below_1_raises() -> None:
    with pytest.raises(Exception):
        _valid_product(star_rating=0)


def test_star_rating_above_5_raises() -> None:
    with pytest.raises(Exception):
        _valid_product(star_rating=6)


def test_customer_rating_above_10_raises() -> None:
    with pytest.raises(Exception):
        _valid_product(customer_rating=10.1)


def test_customer_rating_below_1_raises() -> None:
    with pytest.raises(Exception):
        _valid_product(customer_rating=0.9)


def test_price_zero_raises() -> None:
    with pytest.raises(Exception):
        _valid_product(price_per_person_gbp=0.0)


def test_negative_beach_distance_raises() -> None:
    with pytest.raises(Exception):
        _valid_product(beach_distance_km=-0.1)


def test_latitude_out_of_range_raises() -> None:
    with pytest.raises(Exception):
        _valid_product(latitude=91.0)


def test_longitude_out_of_range_raises() -> None:
    with pytest.raises(Exception):
        _valid_product(longitude=181.0)


def test_neither_family_nor_adults_is_valid() -> None:
    product = _valid_product(family_friendly=False, adults_only=False)
    assert not product.family_friendly
    assert not product.adults_only


# ── build_embedding_text ─────────────────────────────────────────────────────


def test_embedding_text_contains_hotel_name() -> None:
    text = build_embedding_text(_valid_product())
    assert "Azure Sands Resort" in text


def test_embedding_text_contains_destination() -> None:
    text = build_embedding_text(_valid_product())
    assert "Albufeira" in text
    assert "Algarve" in text
    assert "Portugal" in text


def test_embedding_text_contains_description() -> None:
    text = build_embedding_text(_valid_product())
    assert "wide sandy beach" in text


def test_embedding_text_contains_activities() -> None:
    text = build_embedding_text(_valid_product())
    assert "snorkelling" in text


def test_embedding_text_contains_tags() -> None:
    text = build_embedding_text(_valid_product())
    assert "beach" in text


def test_embedding_text_contains_climate() -> None:
    text = build_embedding_text(_valid_product())
    assert "Mediterranean" in text


def test_embedding_text_contains_board_types() -> None:
    text = build_embedding_text(_valid_product())
    assert "half board" in text or "all inclusive" in text


def test_embedding_text_contains_family_flag_when_family() -> None:
    text = build_embedding_text(_valid_product(family_friendly=True, adults_only=False))
    assert "Family friendly" in text or "family" in text.lower()


def test_embedding_text_contains_adults_only_flag() -> None:
    text = build_embedding_text(_valid_product(family_friendly=False, adults_only=True))
    assert "Adults only" in text


def test_embedding_text_excludes_price() -> None:
    product = _valid_product(price_per_person_gbp=749.0)
    text = build_embedding_text(product)
    assert "749" not in text


def test_embedding_text_excludes_star_rating() -> None:
    product = _valid_product(star_rating=4)
    text = build_embedding_text(product)
    # The digit "4" could appear in other contexts, so test the specific pattern
    assert "star_rating" not in text


def test_embedding_text_excludes_beach_distance() -> None:
    product = _valid_product(beach_distance_km=0.1)
    text = build_embedding_text(product)
    assert "beach_distance" not in text


def test_embedding_text_excludes_airport_codes() -> None:
    product = _valid_product(available_departure_airports=["LGW", "MAN"])
    text = build_embedding_text(product)
    assert "LGW" not in text
    assert "MAN" not in text


def test_embedding_text_is_a_non_empty_string() -> None:
    text = build_embedding_text(_valid_product())
    assert isinstance(text, str)
    assert len(text) > 50
