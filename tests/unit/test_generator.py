"""Unit tests for the synthetic dataset generator."""

# generate_dataset is imported from scripts/ which adds sys.path internally.
# We use importlib to load it without installing it as a package.
import importlib.util
import sys
from collections import Counter
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "generate_dataset.py"
_spec = importlib.util.spec_from_file_location("generate_dataset", _SCRIPT)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["generate_dataset"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

generate_dataset = _mod.generate_dataset


# ── Determinism ───────────────────────────────────────────────────────────────


def test_same_seed_produces_identical_datasets() -> None:
    a = generate_dataset(seed=42)
    b = generate_dataset(seed=42)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.model_dump() == y.model_dump()


def test_different_seeds_produce_different_datasets() -> None:
    a = generate_dataset(seed=42)
    b = generate_dataset(seed=99)
    # At least some hotels should differ
    diffs = sum(1 for x, y in zip(a, b) if x.hotel_name != y.hotel_name)
    assert diffs > 100


# ── Size ──────────────────────────────────────────────────────────────────────


def test_dataset_size_within_spec_bounds() -> None:
    products = generate_dataset()
    assert 5_000 <= len(products) <= 10_000


def test_ids_are_unique() -> None:
    products = generate_dataset()
    ids = [p.id for p in products]
    assert len(ids) == len(set(ids))


def test_ids_are_sequential() -> None:
    products = generate_dataset()
    for i, p in enumerate(products, start=1):
        assert p.id == f"hotel_{i:06d}"


# ── Geographic diversity ──────────────────────────────────────────────────────


def test_multiple_countries_represented() -> None:
    products = generate_dataset()
    countries = {p.country for p in products}
    assert len(countries) >= 10


def test_all_spec_countries_present() -> None:
    products = generate_dataset()
    countries = {p.country for p in products}
    expected = {
        "Spain",
        "Portugal",
        "Greece",
        "Cyprus",
        "Turkey",
        "Italy",
        "Croatia",
        "Morocco",
        "Maldives",
        "Thailand",
    }
    assert expected <= countries


def test_multiple_destinations_per_country() -> None:
    products = generate_dataset()
    spain = [p.destination for p in products if p.country == "Spain"]
    assert len(set(spain)) >= 5


# ── Data quality ──────────────────────────────────────────────────────────────


def test_no_product_is_both_family_and_adults_only() -> None:
    products = generate_dataset()
    violations = [p for p in products if p.family_friendly and p.adults_only]
    assert violations == []


def test_star_ratings_in_valid_range() -> None:
    products = generate_dataset()
    assert all(1 <= p.star_rating <= 5 for p in products)


def test_customer_ratings_in_valid_range() -> None:
    products = generate_dataset()
    assert all(1.0 <= p.customer_rating <= 10.0 for p in products)


def test_prices_are_positive() -> None:
    products = generate_dataset()
    assert all(p.price_per_person_gbp > 0 for p in products)


def test_beach_distances_are_non_negative() -> None:
    products = generate_dataset()
    assert all(p.beach_distance_km >= 0 for p in products)


def test_all_products_have_amenities() -> None:
    products = generate_dataset()
    assert all(len(p.amenities) > 0 for p in products)


def test_all_products_have_activities() -> None:
    products = generate_dataset()
    assert all(len(p.activities) > 0 for p in products)


def test_all_products_have_available_months() -> None:
    products = generate_dataset()
    assert all(len(p.available_months) > 0 for p in products)


def test_all_products_have_airports() -> None:
    products = generate_dataset()
    assert all(len(p.available_departure_airports) > 0 for p in products)


# ── Distribution ──────────────────────────────────────────────────────────────


def test_star_rating_distribution_is_reasonable() -> None:
    products = generate_dataset()
    stars = Counter(p.star_rating for p in products)
    total = len(products)
    # No single star rating should dominate (> 50 %) or be absent
    for star in range(1, 6):
        assert stars[star] > 0, f"{star}-star hotels are missing"
        pct = stars[star] / total
        assert pct < 0.50, f"{star}-star hotels are {pct:.0%} — too many"


def test_meaningful_family_and_adults_only_proportions() -> None:
    products = generate_dataset()
    n = len(products)
    n_family = sum(1 for p in products if p.family_friendly)
    n_adults = sum(1 for p in products if p.adults_only)
    # Both groups should be a meaningful proportion of the dataset
    assert 0.10 < n_family / n < 0.70
    assert 0.05 < n_adults / n < 0.50


def test_price_range_reflects_destination_variety() -> None:
    products = generate_dataset()
    prices = [p.price_per_person_gbp for p in products]
    assert min(prices) < 400
    assert max(prices) > 1500
    # Wide spread
    assert max(prices) / min(prices) > 3


def test_all_climate_zones_represented() -> None:
    products = generate_dataset()
    zones = {p.climate_zone for p in products}
    assert "Mediterranean" in zones
    assert "Tropical" in zones
    assert "Subtropical" in zones
