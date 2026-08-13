"""Build the golden relevance dataset from the processed hotel corpus.

This script is a one-time builder. It scans data/processed/hotels.jsonl,
applies hand-crafted relevance criteria for each query, and writes
data/evaluation/golden_queries.jsonl.

Run:
    uv run python scripts/build_golden_dataset.py

Commit the output alongside this script so the criteria are auditable.

Relevance grades
----------------
3 = highly relevant (matches primary intent and all key constraints)
2 = relevant        (matches intent, one secondary constraint missing)
1 = marginal        (loosely related; may satisfy the user)
0 = irrelevant      (omitted from output; unjudged-is-irrelevant assumption)

Limits per grade
----------------
To keep the file size manageable and Recall@K meaningful, grade-1
judgments are capped at MAX_GRADE1 hotels (sampled deterministically).
Grade-2 and grade-3 judgments are included in full.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HOTELS_PATH = Path("data/processed/hotels.jsonl")
OUTPUT_PATH = Path("data/evaluation/golden_queries.jsonl")
MAX_GRADE1 = 40  # cap on marginal judgments per query
SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────


def has_tag(h: dict[str, Any], *tags: str) -> bool:
    hotel_tags = set(h.get("tags", []))
    return all(t in hotel_tags for t in tags)


def has_any_tag(h: dict[str, Any], *tags: str) -> bool:
    hotel_tags = set(h.get("tags", []))
    return any(t in hotel_tags for t in tags)


def has_board(h: dict[str, Any], *boards: str) -> bool:
    hotel_boards = set(h.get("board_types", []))
    return all(b in hotel_boards for b in boards)


def has_any_board(h: dict[str, Any], *boards: str) -> bool:
    hotel_boards = set(h.get("board_types", []))
    return any(b in hotel_boards for b in boards)


def has_month(h: dict[str, Any], month: str) -> bool:
    return month in h.get("available_months", [])


def has_airport(h: dict[str, Any], airport: str) -> bool:
    return airport in h.get("available_departure_airports", [])


def has_activity(h: dict[str, Any], *activities: str) -> bool:
    hotel_acts = set(h.get("activities", []))
    return all(a in hotel_acts for a in activities)


def has_amenity(h: dict[str, Any], *amenities: str) -> bool:
    hotel_ams = set(h.get("amenities", []))
    return all(a in hotel_ams for a in amenities)


# ── Query specification ────────────────────────────────────────────────────────


@dataclass
class QuerySpec:
    query_id: str
    query_text: str
    query_class: str
    grade3: Callable[[dict[str, Any]], bool]
    grade2: Callable[[dict[str, Any]], bool]
    grade1: Callable[[dict[str, Any]], bool]
    filters: dict[str, Any] = field(default_factory=dict)


_FALSE: Callable[[dict[str, Any]], bool] = lambda h: False  # noqa: E731

QUERY_SPECS: list[QuerySpec] = [
    # ── Exact destination ─────────────────────────────────────────────────────
    QuerySpec(
        "q001",
        "hotels in Tenerife",
        "exact_destination",
        grade3=lambda h: h["region"] == "Tenerife",
        grade2=lambda h: (
            h["country"] == "Spain"
            and h["region"] in {"Gran Canaria", "Lanzarote", "Fuerteventura"}
        ),
        grade1=lambda h: h["country"] == "Spain",
    ),
    QuerySpec(
        "q002",
        "Mallorca beach resort",
        "exact_destination",
        grade3=lambda h: h["region"] == "Mallorca",
        grade2=lambda h: h["country"] == "Spain" and h["region"] in {"Menorca", "Ibiza"},
        grade1=lambda h: h["country"] == "Spain" and has_tag(h, "beach"),
    ),
    QuerySpec(
        "q003",
        "Crete holiday",
        "exact_destination",
        grade3=lambda h: h["region"] == "Crete",
        grade2=lambda h: h["country"] == "Greece",
        grade1=lambda h: h.get("climate_zone") == "Mediterranean",
    ),
    QuerySpec(
        "q004",
        "Algarve Portugal beach",
        "exact_destination",
        grade3=lambda h: h["region"] == "Algarve",
        grade2=lambda h: h["country"] == "Portugal",
        grade1=_FALSE,
    ),
    QuerySpec(
        "q005",
        "Turkey beach hotel",
        "exact_destination",
        grade3=lambda h: h["country"] == "Turkey" and has_tag(h, "beach"),
        grade2=lambda h: h["country"] == "Turkey",
        grade1=lambda h: h.get("climate_zone") == "Mediterranean" and has_tag(h, "beach"),
    ),
    QuerySpec(
        "q006",
        "Ibiza party holiday",
        "exact_destination",
        grade3=lambda h: h["region"] == "Ibiza",
        grade2=lambda h: h["country"] == "Spain" and has_tag(h, "nightlife"),
        grade1=lambda h: has_tag(h, "nightlife") and has_tag(h, "beach"),
    ),
    QuerySpec(
        "q007",
        "Gran Canaria resort",
        "exact_destination",
        grade3=lambda h: h["region"] == "Gran Canaria",
        grade2=lambda h: (
            h["country"] == "Spain" and h["region"] in {"Tenerife", "Lanzarote", "Fuerteventura"}
        ),
        grade1=lambda h: h["country"] == "Spain",
    ),
    QuerySpec(
        "q008",
        "Cyprus family holiday",
        "exact_destination",
        grade3=lambda h: h["country"] == "Cyprus" and h["family_friendly"],
        grade2=lambda h: h["country"] == "Cyprus",
        grade1=lambda h: h.get("climate_zone") == "Mediterranean" and h["family_friendly"],
    ),
    QuerySpec(
        "q009",
        "Morocco authentic cultural experience",
        "exact_destination",
        grade3=lambda h: h["country"] == "Morocco" and has_tag(h, "culture"),
        grade2=lambda h: h["country"] == "Morocco",
        grade1=lambda h: has_tag(h, "culture") and has_tag(h, "history"),
    ),
    QuerySpec(
        "q010",
        "Thailand luxury beach resort",
        "exact_destination",
        grade3=lambda h: (
            h["country"] == "Thailand" and has_tag(h, "luxury") and has_tag(h, "beach")
        ),
        grade2=lambda h: h["country"] == "Thailand",
        grade1=lambda h: h.get("climate_zone") == "Tropical" and has_tag(h, "luxury"),
    ),
    # ── Family ────────────────────────────────────────────────────────────────
    QuerySpec(
        "q011",
        "family beach holiday",
        "family",
        grade3=lambda h: (
            h["family_friendly"] and has_tag(h, "beach") and h["beach_distance_km"] <= 0.5
        ),
        grade2=lambda h: h["family_friendly"] and has_tag(h, "beach"),
        grade1=lambda h: h["family_friendly"],
    ),
    QuerySpec(
        "q012",
        "family hotel kids club entertainment",
        "family",
        grade3=lambda h: h["family_friendly"] and has_tag(h, "kids-club"),
        grade2=lambda h: h["family_friendly"],
        grade1=lambda h: has_tag(h, "kids-club"),
    ),
    QuerySpec(
        "q013",
        "all inclusive family resort",
        "family",
        grade3=lambda h: h["family_friendly"] and has_any_board(h, "all_inclusive"),
        grade2=lambda h: h["family_friendly"],
        grade1=lambda h: has_any_board(h, "all_inclusive"),
    ),
    QuerySpec(
        "q014",
        "family holiday Tenerife",
        "family",
        grade3=lambda h: h["family_friendly"] and h["region"] == "Tenerife",
        grade2=lambda h: h["region"] == "Tenerife",
        grade1=lambda h: h["family_friendly"] and h["country"] == "Spain",
    ),
    QuerySpec(
        "q015",
        "cheap family holiday under 800 pounds",
        "family",
        grade3=lambda h: h["family_friendly"] and h["price_per_person_gbp"] <= 800,
        grade2=lambda h: h["family_friendly"] and h["price_per_person_gbp"] <= 1000,
        grade1=lambda h: h["price_per_person_gbp"] <= 800,
        filters={"family_friendly": True, "max_price": 800},
    ),
    QuerySpec(
        "q016",
        "family beach half board Spain",
        "family",
        grade3=lambda h: (
            h["family_friendly"]
            and has_any_board(h, "half_board")
            and h["country"] == "Spain"
            and h["beach_distance_km"] <= 0.5
        ),
        grade2=lambda h: (
            h["family_friendly"] and has_any_board(h, "half_board") and h["country"] == "Spain"
        ),
        grade1=lambda h: h["family_friendly"] and h["country"] == "Spain",
        filters={"family_friendly": True, "country": "Spain"},
    ),
    QuerySpec(
        "q017",
        "family friendly Greece beach",
        "family",
        grade3=lambda h: h["family_friendly"] and h["country"] == "Greece" and has_tag(h, "beach"),
        grade2=lambda h: h["family_friendly"] and h["country"] == "Greece",
        grade1=lambda h: h["country"] == "Greece" and has_tag(h, "beach"),
        filters={"family_friendly": True, "country": "Greece"},
    ),
    QuerySpec(
        "q018",
        "water sports beach resort lively activities",
        "activities",
        grade3=lambda h: has_tag(h, "watersports") and h["beach_distance_km"] <= 0.3,
        grade2=lambda h: has_tag(h, "watersports") and h["beach_distance_km"] <= 0.5,
        grade1=lambda h: (
            has_tag(h, "lively") and has_tag(h, "beach") and not has_tag(h, "watersports")
        ),
    ),
    QuerySpec(
        "q019",
        "kids club hotel Mallorca children",
        "family",
        grade3=lambda h: (
            h["family_friendly"] and h["region"] == "Mallorca" and has_tag(h, "kids-club")
        ),
        grade2=lambda h: h["family_friendly"] and h["region"] == "Mallorca",
        grade1=lambda h: h["region"] == "Mallorca",
    ),
    QuerySpec(
        "q020",
        "family holiday October departing Manchester",
        "family",
        grade3=lambda h: h["family_friendly"] and has_month(h, "October") and has_airport(h, "MAN"),
        grade2=lambda h: h["family_friendly"] and has_month(h, "October"),
        grade1=lambda h: has_month(h, "October") and has_airport(h, "MAN"),
        filters={"family_friendly": True, "month": "October", "airport": "MAN"},
    ),
    # ── Adults / couples ──────────────────────────────────────────────────────
    QuerySpec(
        "q021",
        "adults only hotel",
        "adults_couples",
        grade3=lambda h: h["adults_only"],
        grade2=_FALSE,
        grade1=_FALSE,
        filters={"adults_only": True},
    ),
    QuerySpec(
        "q022",
        "romantic couples retreat spa",
        "adults_couples",
        grade3=lambda h: h["adults_only"] and has_tag(h, "romantic") and has_tag(h, "spa"),
        grade2=lambda h: h["adults_only"] and has_tag(h, "romantic"),
        grade1=lambda h: has_tag(h, "romantic") and has_tag(h, "spa"),
    ),
    QuerySpec(
        "q023",
        "adults only all inclusive resort",
        "adults_couples",
        grade3=lambda h: h["adults_only"] and has_any_board(h, "all_inclusive"),
        grade2=lambda h: h["adults_only"],
        grade1=lambda h: has_any_board(h, "all_inclusive") and has_tag(h, "adults-only"),
        filters={"adults_only": True},
    ),
    QuerySpec(
        "q024",
        "luxury adults only spa 5 star hotel",
        "adults_couples",
        grade3=lambda h: h["adults_only"] and has_tag(h, "spa") and h["star_rating"] == 5,
        grade2=lambda h: h["adults_only"] and has_tag(h, "spa") and h["star_rating"] >= 4,
        grade1=lambda h: h["adults_only"] and h["star_rating"] >= 4,
        filters={"adults_only": True, "min_star_rating": 4},
    ),
    QuerySpec(
        "q025",
        "honeymoon resort romantic luxury",
        "adults_couples",
        grade3=lambda h: (
            h["adults_only"]
            and has_tag(h, "romantic")
            and h["star_rating"] >= 4
            and has_tag(h, "luxury")
        ),
        grade2=lambda h: h["adults_only"] and has_tag(h, "romantic") and h["star_rating"] >= 4,
        grade1=lambda h: h["adults_only"] and has_tag(h, "romantic"),
    ),
    QuerySpec(
        "q026",
        "adults only Maldives resort",
        "adults_couples",
        grade3=lambda h: h["adults_only"] and h["country"] == "Maldives",
        grade2=lambda h: h["country"] == "Maldives",
        grade1=lambda h: h["adults_only"] and h.get("climate_zone") == "Tropical",
        filters={"adults_only": True, "country": "Maldives"},
    ),
    # ── Luxury ────────────────────────────────────────────────────────────────
    QuerySpec(
        "q027",
        "5 star luxury hotel",
        "luxury",
        grade3=lambda h: h["star_rating"] == 5,
        grade2=lambda h: h["star_rating"] == 4,
        grade1=lambda h: h["star_rating"] == 3 and has_tag(h, "luxury"),
        filters={"min_star_rating": 5},
    ),
    QuerySpec(
        "q028",
        "luxury spa resort",
        "luxury",
        grade3=lambda h: has_tag(h, "luxury") and has_tag(h, "spa") and h["star_rating"] >= 4,
        grade2=lambda h: has_tag(h, "luxury") and has_tag(h, "spa"),
        grade1=lambda h: has_tag(h, "luxury") and h["star_rating"] >= 4,
    ),
    QuerySpec(
        "q029",
        "boutique luxury hotel peaceful",
        "luxury",
        grade3=lambda h: has_tag(h, "boutique") and has_tag(h, "luxury"),
        grade2=lambda h: has_tag(h, "boutique") and has_tag(h, "peaceful"),
        grade1=lambda h: has_tag(h, "boutique"),
    ),
    QuerySpec(
        "q030",
        "premium all inclusive luxury resort",
        "luxury",
        grade3=lambda h: (
            h["star_rating"] >= 4 and has_any_board(h, "all_inclusive") and has_tag(h, "luxury")
        ),
        grade2=lambda h: has_any_board(h, "all_inclusive") and has_tag(h, "luxury"),
        grade1=lambda h: has_any_board(h, "all_inclusive") and h["star_rating"] >= 4,
    ),
    QuerySpec(
        "q031",
        "high end beach resort luxury",
        "luxury",
        grade3=lambda h: (
            h["star_rating"] >= 4 and has_tag(h, "luxury") and h["beach_distance_km"] <= 0.3
        ),
        grade2=lambda h: has_tag(h, "luxury") and h["beach_distance_km"] <= 0.5,
        grade1=lambda h: h["star_rating"] >= 4 and h["beach_distance_km"] <= 0.5,
    ),
    QuerySpec(
        "q032",
        "fine dining luxury hotel high rating",
        "luxury",
        grade3=lambda h: (
            has_tag(h, "luxury") and h["star_rating"] >= 4 and h["customer_rating"] >= 8.5
        ),
        grade2=lambda h: has_tag(h, "luxury") and h["star_rating"] >= 4,
        grade1=lambda h: h["star_rating"] >= 4 and h["customer_rating"] >= 8.5,
    ),
    # ── Budget ────────────────────────────────────────────────────────────────
    QuerySpec(
        "q033",
        "budget beach holiday affordable",
        "budget",
        grade3=lambda h: h["price_per_person_gbp"] <= 600 and has_tag(h, "beach"),
        grade2=lambda h: h["price_per_person_gbp"] <= 800 and has_tag(h, "beach"),
        grade1=lambda h: h["price_per_person_gbp"] <= 600,
    ),
    QuerySpec(
        "q034",
        "cheap hotel Spain budget",
        "budget",
        grade3=lambda h: h["price_per_person_gbp"] <= 600 and h["country"] == "Spain",
        grade2=lambda h: h["price_per_person_gbp"] <= 800 and h["country"] == "Spain",
        grade1=lambda h: h["country"] == "Spain" and has_tag(h, "budget"),
        filters={"country": "Spain", "max_price": 800},
    ),
    QuerySpec(
        "q035",
        "affordable family holiday good value",
        "budget",
        grade3=lambda h: h["family_friendly"] and h["price_per_person_gbp"] <= 700,
        grade2=lambda h: h["family_friendly"] and h["price_per_person_gbp"] <= 900,
        grade1=lambda h: h["price_per_person_gbp"] <= 700,
        filters={"family_friendly": True, "max_price": 900},
    ),
    QuerySpec(
        "q036",
        "value for money 4 star highly rated",
        "budget",
        grade3=lambda h: (
            h["star_rating"] >= 4
            and h["price_per_person_gbp"] <= 900
            and h["customer_rating"] >= 8.0
        ),
        grade2=lambda h: h["star_rating"] >= 4 and h["price_per_person_gbp"] <= 1100,
        grade1=lambda h: h["star_rating"] >= 3 and h["price_per_person_gbp"] <= 700,
    ),
    QuerySpec(
        "q037",
        "good value all inclusive holiday",
        "budget",
        grade3=lambda h: (
            has_any_board(h, "all_inclusive")
            and h["price_per_person_gbp"] <= 800
            and h["customer_rating"] >= 7.5
        ),
        grade2=lambda h: has_any_board(h, "all_inclusive") and h["price_per_person_gbp"] <= 800,
        grade1=lambda h: has_any_board(h, "all_inclusive") and h["price_per_person_gbp"] <= 1000,
    ),
    # ── Nightlife ─────────────────────────────────────────────────────────────
    QuerySpec(
        "q038",
        "nightlife party resort bars clubs",
        "nightlife",
        grade3=lambda h: has_tag(h, "nightlife") and has_tag(h, "party"),
        grade2=lambda h: has_tag(h, "nightlife") and has_tag(h, "bars"),
        grade1=lambda h: has_tag(h, "nightlife"),
    ),
    QuerySpec(
        "q039",
        "Ibiza party beach nightlife",
        "nightlife",
        grade3=lambda h: h["region"] == "Ibiza",
        grade2=lambda h: h["country"] == "Spain" and has_tag(h, "nightlife"),
        grade1=lambda h: has_tag(h, "nightlife") and has_tag(h, "beach"),
    ),
    QuerySpec(
        "q040",
        "beach bars clubs nightlife holiday",
        "nightlife",
        grade3=lambda h: has_tag(h, "nightlife") and has_tag(h, "bars") and has_tag(h, "beach"),
        grade2=lambda h: has_tag(h, "nightlife") and has_tag(h, "beach"),
        grade1=lambda h: has_tag(h, "lively") and has_tag(h, "beach"),
    ),
    QuerySpec(
        "q041",
        "lively resort entertainment animation",
        "nightlife",
        grade3=lambda h: has_tag(h, "lively") and has_tag(h, "nightlife"),
        grade2=lambda h: has_tag(h, "lively"),
        grade1=lambda h: has_tag(h, "pool-bar") and has_tag(h, "beach"),
    ),
    QuerySpec(
        "q042",
        "young party holiday summer beach",
        "nightlife",
        grade3=lambda h: has_tag(h, "party") and has_tag(h, "nightlife") and has_tag(h, "beach"),
        grade2=lambda h: has_tag(h, "party") and has_tag(h, "beach"),
        grade1=lambda h: has_tag(h, "lively") and has_tag(h, "beach"),
    ),
    # ── Quiet / peaceful ──────────────────────────────────────────────────────
    QuerySpec(
        "q043",
        "quiet peaceful relaxing beach retreat",
        "quiet_peaceful",
        grade3=lambda h: (
            has_tag(h, "quiet") and has_tag(h, "peaceful") and h["beach_distance_km"] <= 1.0
        ),
        grade2=lambda h: has_tag(h, "quiet") and has_tag(h, "peaceful"),
        grade1=lambda h: has_tag(h, "quiet") and has_tag(h, "beach"),
    ),
    QuerySpec(
        "q044",
        "tranquil nature escape scenic",
        "quiet_peaceful",
        grade3=lambda h: has_tag(h, "peaceful") and has_tag(h, "nature") and has_tag(h, "scenic"),
        grade2=lambda h: has_tag(h, "peaceful") and has_tag(h, "nature"),
        grade1=lambda h: has_tag(h, "nature") and has_tag(h, "scenic"),
    ),
    QuerySpec(
        "q045",
        "peaceful boutique adults only retreat",
        "quiet_peaceful",
        grade3=lambda h: h["adults_only"] and has_tag(h, "peaceful") and has_tag(h, "boutique"),
        grade2=lambda h: h["adults_only"] and has_tag(h, "quiet"),
        grade1=lambda h: has_tag(h, "peaceful") and has_tag(h, "boutique"),
        filters={"adults_only": True},
    ),
    QuerySpec(
        "q046",
        "boutique peaceful countryside hotel",
        "quiet_peaceful",
        grade3=lambda h: (
            has_tag(h, "boutique") and has_tag(h, "peaceful") and h["beach_distance_km"] >= 2.0
        ),
        grade2=lambda h: has_tag(h, "boutique") and has_tag(h, "peaceful"),
        grade1=lambda h: has_tag(h, "boutique") and has_tag(h, "quiet"),
    ),
    QuerySpec(
        "q047",
        "peaceful yoga wellness nature retreat",
        "quiet_peaceful",
        grade3=lambda h: has_tag(h, "peaceful") and has_activity(h, "yoga"),
        grade2=lambda h: has_tag(h, "nature") and has_activity(h, "yoga"),
        grade1=lambda h: has_activity(h, "yoga") and has_tag(h, "quiet"),
    ),
    # ── Activities ────────────────────────────────────────────────────────────
    QuerySpec(
        "q048",
        "snorkelling diving watersports resort",
        "activities",
        grade3=lambda h: (
            has_tag(h, "watersports")
            and has_activity(h, "snorkelling")
            and has_activity(h, "kayaking")
        ),
        grade2=lambda h: has_tag(h, "watersports") and has_activity(h, "snorkelling"),
        grade1=lambda h: has_tag(h, "watersports"),
    ),
    QuerySpec(
        "q049",
        "hiking walking nature holiday",
        "activities",
        grade3=lambda h: (
            has_tag(h, "nature") and has_activity(h, "hiking") and has_tag(h, "scenic")
        ),
        grade2=lambda h: has_tag(h, "nature") and has_activity(h, "hiking"),
        grade1=lambda h: has_activity(h, "hiking"),
    ),
    QuerySpec(
        "q050",
        "yoga wellness spa luxury retreat",
        "activities",
        grade3=lambda h: has_activity(h, "yoga") and has_tag(h, "spa"),
        grade2=lambda h: has_activity(h, "yoga") and has_tag(h, "luxury"),
        grade1=lambda h: has_activity(h, "yoga"),
    ),
    QuerySpec(
        "q051",
        "watersports jet skiing beach holiday",
        "activities",
        grade3=lambda h: (
            has_tag(h, "watersports")
            and has_activity(h, "jet skiing")
            and h["beach_distance_km"] <= 0.5
        ),
        grade2=lambda h: has_tag(h, "watersports") and h["beach_distance_km"] <= 0.5,
        grade1=lambda h: has_tag(h, "watersports"),
    ),
    QuerySpec(
        "q052",
        "cultural heritage history sightseeing",
        "activities",
        grade3=lambda h: (
            has_tag(h, "culture") and has_tag(h, "history") and has_tag(h, "authentic")
        ),
        grade2=lambda h: has_tag(h, "culture") and has_tag(h, "history"),
        grade1=lambda h: has_tag(h, "culture"),
    ),
    # ── Multi-constraint ──────────────────────────────────────────────────────
    QuerySpec(
        "q053",
        "family beach hotel Greece July Manchester",
        "multi_constraint",
        grade3=lambda h: (
            h["family_friendly"]
            and h["country"] == "Greece"
            and has_month(h, "July")
            and has_airport(h, "MAN")
            and h["beach_distance_km"] <= 1.0
        ),
        grade2=lambda h: h["family_friendly"] and h["country"] == "Greece" and has_month(h, "July"),
        grade1=lambda h: (
            h["country"] == "Greece" and has_month(h, "July") and has_airport(h, "MAN")
        ),
        filters={"family_friendly": True, "country": "Greece", "month": "July", "airport": "MAN"},
    ),
    QuerySpec(
        "q054",
        "luxury adults only Spain 5 star",
        "multi_constraint",
        grade3=lambda h: h["adults_only"] and h["country"] == "Spain" and h["star_rating"] == 5,
        grade2=lambda h: h["adults_only"] and h["country"] == "Spain" and h["star_rating"] >= 4,
        grade1=lambda h: h["adults_only"] and h["country"] == "Spain",
        filters={"adults_only": True, "country": "Spain", "min_star_rating": 4},
    ),
    QuerySpec(
        "q055",
        "all inclusive family Portugal departing London",
        "multi_constraint",
        grade3=lambda h: (
            h["family_friendly"]
            and h["country"] == "Portugal"
            and has_any_board(h, "all_inclusive")
            and (has_airport(h, "LGW") or has_airport(h, "LHR"))
        ),
        grade2=lambda h: (
            h["family_friendly"]
            and h["country"] == "Portugal"
            and has_any_board(h, "all_inclusive")
        ),
        grade1=lambda h: h["family_friendly"] and h["country"] == "Portugal",
        filters={"family_friendly": True, "country": "Portugal"},
    ),
    QuerySpec(
        "q056",
        "budget family all inclusive September under 800",
        "multi_constraint",
        grade3=lambda h: (
            h["family_friendly"]
            and has_any_board(h, "all_inclusive")
            and h["price_per_person_gbp"] <= 800
            and has_month(h, "September")
        ),
        grade2=lambda h: (
            h["family_friendly"]
            and has_any_board(h, "all_inclusive")
            and h["price_per_person_gbp"] <= 800
        ),
        grade1=lambda h: h["family_friendly"] and has_any_board(h, "all_inclusive"),
        filters={"family_friendly": True, "max_price": 800, "month": "September"},
    ),
    QuerySpec(
        "q057",
        "4 star adults only Turkey all inclusive",
        "multi_constraint",
        grade3=lambda h: (
            h["adults_only"]
            and h["country"] == "Turkey"
            and h["star_rating"] >= 4
            and has_any_board(h, "all_inclusive")
        ),
        grade2=lambda h: h["adults_only"] and h["country"] == "Turkey" and h["star_rating"] >= 4,
        grade1=lambda h: h["country"] == "Turkey" and has_any_board(h, "all_inclusive"),
        filters={"adults_only": True, "country": "Turkey", "min_star_rating": 4},
    ),
    QuerySpec(
        "q058",
        "family hotel Crete July half board",
        "multi_constraint",
        grade3=lambda h: (
            h["family_friendly"]
            and h["region"] == "Crete"
            and has_month(h, "July")
            and has_any_board(h, "half_board")
        ),
        grade2=lambda h: h["family_friendly"] and h["region"] == "Crete" and has_month(h, "July"),
        grade1=lambda h: h["region"] == "Crete" and has_month(h, "July"),
        filters={"family_friendly": True, "country": "Greece", "month": "July"},
    ),
    # ── Natural language ──────────────────────────────────────────────────────
    QuerySpec(
        "q059",
        "Find me a family holiday somewhere warm in October departing from Manchester under 2000 pounds",
        "natural_language",
        grade3=lambda h: (
            h["family_friendly"]
            and has_month(h, "October")
            and has_airport(h, "MAN")
            and h["price_per_person_gbp"] <= 2000
        ),
        grade2=lambda h: (
            h["family_friendly"] and has_month(h, "October") and h["price_per_person_gbp"] <= 2000
        ),
        grade1=lambda h: h["family_friendly"] and has_month(h, "October"),
        filters={"family_friendly": True, "month": "October", "airport": "MAN", "max_price": 2000},
    ),
    QuerySpec(
        "q060",
        "Somewhere like Mallorca but quieter and more peaceful",
        "natural_language",
        grade3=lambda h: h["region"] == "Menorca" and has_tag(h, "quiet"),
        grade2=lambda h: (
            h["region"] in {"Menorca", "Mallorca"} and has_any_tag(h, "quiet", "peaceful")
        ),
        grade1=lambda h: has_tag(h, "quiet") and h["country"] == "Spain" and has_tag(h, "beach"),
    ),
    QuerySpec(
        "q061",
        "Romantic getaway with a spa and excellent food for two adults",
        "natural_language",
        grade3=lambda h: (
            h["adults_only"]
            and has_tag(h, "romantic")
            and has_tag(h, "spa")
            and h["star_rating"] >= 4
        ),
        grade2=lambda h: h["adults_only"] and has_tag(h, "romantic") and has_tag(h, "spa"),
        grade1=lambda h: has_tag(h, "romantic") and has_tag(h, "spa"),
    ),
    QuerySpec(
        "q062",
        "Beach holiday for families with young children lots of entertainment",
        "natural_language",
        grade3=lambda h: (
            h["family_friendly"] and has_tag(h, "kids-club") and h["beach_distance_km"] <= 0.5
        ),
        grade2=lambda h: h["family_friendly"] and has_tag(h, "kids-club"),
        grade1=lambda h: h["family_friendly"] and h["beach_distance_km"] <= 0.5,
    ),
]


# ── Builder ────────────────────────────────────────────────────────────────────


def build(hotels: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for spec in QUERY_SPECS:
        g3, g2, g1 = [], [], []
        seen: set[str] = set()

        for h in hotels:
            hid = h["id"]
            if spec.grade3(h):
                g3.append(hid)
                seen.add(hid)
            elif spec.grade2(h):
                g2.append(hid)
                seen.add(hid)
            elif spec.grade1(h) and hid not in seen:
                g1.append(hid)
                seen.add(hid)

        # Cap and shuffle grade-1 to get a deterministic sample
        if len(g1) > MAX_GRADE1:
            g1_shuffled = g1[:]
            rng.shuffle(g1_shuffled)
            g1 = sorted(g1_shuffled[:MAX_GRADE1])

        judgments: list[dict[str, Any]] = []
        for hid in sorted(g3):
            judgments.append({"doc_id": hid, "grade": 3})
        for hid in sorted(g2):
            judgments.append({"doc_id": hid, "grade": 2})
        for hid in sorted(g1):
            judgments.append({"doc_id": hid, "grade": 1})

        output.append(
            {
                "query_id": spec.query_id,
                "query_text": spec.query_text,
                "query_class": spec.query_class,
                "filters": spec.filters,
                "judgments": judgments,
            }
        )

    return output


def main() -> None:
    print(f"Loading hotels from {HOTELS_PATH} …")
    hotels = [
        json.loads(line)
        for line in HOTELS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"  {len(hotels)} hotels loaded")

    rng = random.Random(SEED)
    queries = build(hotels, rng)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    total_judgments = sum(len(q["judgments"]) for q in queries)
    classes = {q["query_class"] for q in queries}
    print(f"\nWrote {len(queries)} queries, {total_judgments} judgments, {len(classes)} classes")
    print(f"Output: {OUTPUT_PATH}")

    for qc in sorted(classes):
        class_queries = [q for q in queries if q["query_class"] == qc]
        total_g3 = sum(sum(1 for j in q["judgments"] if j["grade"] == 3) for q in class_queries)
        total_g2 = sum(sum(1 for j in q["judgments"] if j["grade"] == 2) for q in class_queries)
        total_g1 = sum(sum(1 for j in q["judgments"] if j["grade"] == 1) for q in class_queries)
        print(f"  {qc}: {len(class_queries)} queries  g3={total_g3} g2={total_g2} g1={total_g1}")


if __name__ == "__main__":
    main()
