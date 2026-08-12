#!/usr/bin/env python
"""Generate the synthetic travel hotel dataset.

Usage:
    make generate-data
    uv run python scripts/generate_dataset.py [--output PATH] [--seed N]

Outputs a JSON Lines file (one TravelProduct per line) to data/processed/hotels.jsonl.
The generator is fully deterministic given the same seed (default: 42).
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_ai_search.domain.models import TravelProduct

# ---------------------------------------------------------------------------
# Destination clusters
# ---------------------------------------------------------------------------

_ALL_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


@dataclass(frozen=True)
class DestinationCluster:
    destination: str
    region: str
    country: str
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    climate_zone: str
    peak_months: list[str]
    count: int
    # Character weights 0.0–1.0
    luxury: float
    family: float
    nightlife: float
    beach: float
    nature: float
    culture: float
    # Typical UK departure airports (IATA)
    airports: list[str]
    # Base GBP price per person for a 3-star, half-board stay
    base_price_gbp: tuple[float, float]


_DESTINATIONS: list[DestinationCluster] = [
    # ── Canary Islands ─────────────────────────────────────────────────────
    DestinationCluster(
        "Playa de las Américas",
        "Tenerife",
        "Spain",
        (28.04, 28.10),
        (-16.74, -16.68),
        "Subtropical",
        ["May", "June", "July", "August", "September", "October"],
        300,
        luxury=0.30,
        family=0.70,
        nightlife=0.50,
        beach=0.90,
        nature=0.10,
        culture=0.10,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA", "NCL", "LBA"],
        base_price_gbp=(549, 949),
    ),
    DestinationCluster(
        "Puerto de la Cruz",
        "Tenerife",
        "Spain",
        (28.40, 28.44),
        (-16.56, -16.50),
        "Subtropical",
        ["April", "May", "September", "October", "November"],
        120,
        luxury=0.35,
        family=0.50,
        nightlife=0.20,
        beach=0.40,
        nature=0.65,
        culture=0.50,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI"],
        base_price_gbp=(449, 799),
    ),
    DestinationCluster(
        "Las Palmas",
        "Gran Canaria",
        "Spain",
        (27.92, 27.98),
        (-15.44, -15.38),
        "Subtropical",
        ["May", "June", "July", "August", "September", "October"],
        200,
        luxury=0.25,
        family=0.65,
        nightlife=0.40,
        beach=0.85,
        nature=0.15,
        culture=0.20,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA"],
        base_price_gbp=(499, 849),
    ),
    DestinationCluster(
        "Costa Calma",
        "Fuerteventura",
        "Spain",
        (28.15, 28.21),
        (-14.25, -14.18),
        "Arid Subtropical",
        ["April", "May", "June", "July", "August", "September", "October"],
        160,
        luxury=0.30,
        family=0.55,
        nightlife=0.20,
        beach=0.95,
        nature=0.30,
        culture=0.10,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA"],
        base_price_gbp=(529, 899),
    ),
    DestinationCluster(
        "Puerto del Carmen",
        "Lanzarote",
        "Spain",
        (28.92, 28.96),
        (-13.67, -13.61),
        "Arid Subtropical",
        ["April", "May", "June", "July", "August", "September", "October"],
        170,
        luxury=0.30,
        family=0.55,
        nightlife=0.45,
        beach=0.85,
        nature=0.40,
        culture=0.20,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI"],
        base_price_gbp=(499, 849),
    ),
    # ── Balearic Islands ────────────────────────────────────────────────────
    DestinationCluster(
        "Palma Nova",
        "Mallorca",
        "Spain",
        (39.52, 39.56),
        (2.52, 2.57),
        "Mediterranean",
        ["May", "June", "July", "August", "September"],
        160,
        luxury=0.30,
        family=0.70,
        nightlife=0.40,
        beach=0.85,
        nature=0.15,
        culture=0.15,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA", "NCL", "LBA", "LPL"],
        base_price_gbp=(549, 949),
    ),
    DestinationCluster(
        "Puerto Pollensa",
        "Mallorca",
        "Spain",
        (39.89, 39.93),
        (3.07, 3.13),
        "Mediterranean",
        ["May", "June", "July", "August", "September"],
        140,
        luxury=0.55,
        family=0.50,
        nightlife=0.15,
        beach=0.70,
        nature=0.45,
        culture=0.35,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI"],
        base_price_gbp=(649, 1099),
    ),
    DestinationCluster(
        "Ibiza Town",
        "Ibiza",
        "Spain",
        (38.90, 38.94),
        (1.42, 1.47),
        "Mediterranean",
        ["May", "June", "July", "August", "September"],
        160,
        luxury=0.50,
        family=0.20,
        nightlife=0.90,
        beach=0.80,
        nature=0.10,
        culture=0.30,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA"],
        base_price_gbp=(699, 1299),
    ),
    DestinationCluster(
        "Mahón",
        "Menorca",
        "Spain",
        (39.85, 39.90),
        (4.25, 4.33),
        "Mediterranean",
        ["May", "June", "July", "August", "September"],
        130,
        luxury=0.40,
        family=0.60,
        nightlife=0.15,
        beach=0.70,
        nature=0.55,
        culture=0.40,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI"],
        base_price_gbp=(599, 999),
    ),
    # ── Spanish mainland ────────────────────────────────────────────────────
    DestinationCluster(
        "Benidorm",
        "Costa Blanca",
        "Spain",
        (38.52, 38.56),
        (-0.15, -0.10),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        180,
        luxury=0.15,
        family=0.75,
        nightlife=0.65,
        beach=0.90,
        nature=0.05,
        culture=0.10,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA", "NCL", "LBA", "LPL"],
        base_price_gbp=(399, 699),
    ),
    DestinationCluster(
        "Marbella",
        "Costa del Sol",
        "Spain",
        (36.50, 36.54),
        (-4.95, -4.88),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        170,
        luxury=0.65,
        family=0.40,
        nightlife=0.55,
        beach=0.85,
        nature=0.15,
        culture=0.25,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI"],
        base_price_gbp=(699, 1249),
    ),
    # ── Portugal ────────────────────────────────────────────────────────────
    DestinationCluster(
        "Albufeira",
        "Algarve",
        "Portugal",
        (37.07, 37.11),
        (-8.28, -8.22),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        240,
        luxury=0.35,
        family=0.65,
        nightlife=0.45,
        beach=0.90,
        nature=0.20,
        culture=0.20,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA", "NCL", "LBA"],
        base_price_gbp=(549, 949),
    ),
    DestinationCluster(
        "Funchal",
        "Madeira",
        "Portugal",
        (32.63, 32.68),
        (-16.95, -16.88),
        "Subtropical",
        ["March", "April", "May", "September", "October", "November"],
        130,
        luxury=0.45,
        family=0.45,
        nightlife=0.20,
        beach=0.30,
        nature=0.80,
        culture=0.60,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS"],
        base_price_gbp=(549, 949),
    ),
    # ── Greece ──────────────────────────────────────────────────────────────
    DestinationCluster(
        "Heraklion",
        "Crete",
        "Greece",
        (35.30, 35.36),
        (25.10, 25.17),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        250,
        luxury=0.35,
        family=0.60,
        nightlife=0.35,
        beach=0.80,
        nature=0.30,
        culture=0.55,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA", "NCL", "LBA"],
        base_price_gbp=(499, 899),
    ),
    DestinationCluster(
        "Rhodes Town",
        "Rhodes",
        "Greece",
        (36.41, 36.46),
        (28.20, 28.27),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        190,
        luxury=0.35,
        family=0.60,
        nightlife=0.40,
        beach=0.85,
        nature=0.20,
        culture=0.50,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA"],
        base_price_gbp=(499, 899),
    ),
    DestinationCluster(
        "Corfu Town",
        "Corfu",
        "Greece",
        (39.62, 39.67),
        (19.92, 19.99),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        170,
        luxury=0.40,
        family=0.60,
        nightlife=0.30,
        beach=0.70,
        nature=0.55,
        culture=0.50,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA"],
        base_price_gbp=(499, 899),
    ),
    DestinationCluster(
        "Fira",
        "Santorini",
        "Greece",
        (36.41, 36.45),
        (25.42, 25.48),
        "Mediterranean",
        ["May", "June", "July", "August", "September"],
        130,
        luxury=0.75,
        family=0.25,
        nightlife=0.45,
        beach=0.60,
        nature=0.40,
        culture=0.65,
        airports=["LGW", "LHR", "MAN"],
        base_price_gbp=(899, 1699),
    ),
    DestinationCluster(
        "Mykonos Town",
        "Mykonos",
        "Greece",
        (37.44, 37.48),
        (25.33, 25.39),
        "Mediterranean",
        ["June", "July", "August", "September"],
        120,
        luxury=0.65,
        family=0.15,
        nightlife=0.85,
        beach=0.75,
        nature=0.10,
        culture=0.35,
        airports=["LGW", "LHR", "MAN"],
        base_price_gbp=(799, 1499),
    ),
    # ── Cyprus ──────────────────────────────────────────────────────────────
    DestinationCluster(
        "Ayia Napa",
        "Famagusta",
        "Cyprus",
        (34.98, 35.03),
        (33.98, 34.06),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        160,
        luxury=0.25,
        family=0.55,
        nightlife=0.75,
        beach=0.90,
        nature=0.10,
        culture=0.10,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA"],
        base_price_gbp=(499, 849),
    ),
    DestinationCluster(
        "Paphos",
        "Paphos",
        "Cyprus",
        (34.76, 34.81),
        (32.40, 32.47),
        "Mediterranean",
        ["April", "May", "June", "July", "August", "September", "October", "November"],
        140,
        luxury=0.45,
        family=0.55,
        nightlife=0.25,
        beach=0.70,
        nature=0.30,
        culture=0.55,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI"],
        base_price_gbp=(499, 899),
    ),
    # ── Turkey ──────────────────────────────────────────────────────────────
    DestinationCluster(
        "Bodrum",
        "Muğla",
        "Turkey",
        (37.02, 37.07),
        (27.42, 27.49),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        170,
        luxury=0.50,
        family=0.40,
        nightlife=0.60,
        beach=0.85,
        nature=0.15,
        culture=0.30,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI"],
        base_price_gbp=(549, 999),
    ),
    DestinationCluster(
        "Belek",
        "Antalya",
        "Turkey",
        (36.85, 36.90),
        (31.03, 31.11),
        "Mediterranean",
        ["April", "May", "June", "July", "August", "September", "October", "November"],
        250,
        luxury=0.55,
        family=0.70,
        nightlife=0.25,
        beach=0.85,
        nature=0.20,
        culture=0.25,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA", "NCL", "LBA"],
        base_price_gbp=(599, 1099),
    ),
    DestinationCluster(
        "Marmaris",
        "Muğla",
        "Turkey",
        (36.85, 36.89),
        (28.25, 28.32),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        170,
        luxury=0.35,
        family=0.60,
        nightlife=0.45,
        beach=0.85,
        nature=0.25,
        culture=0.25,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI", "GLA"],
        base_price_gbp=(499, 899),
    ),
    # ── Italy ───────────────────────────────────────────────────────────────
    DestinationCluster(
        "Positano",
        "Campania",
        "Italy",
        (40.62, 40.64),
        (14.47, 14.50),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        110,
        luxury=0.70,
        family=0.30,
        nightlife=0.30,
        beach=0.60,
        nature=0.60,
        culture=0.75,
        airports=["LGW", "LHR", "MAN"],
        base_price_gbp=(799, 1499),
    ),
    DestinationCluster(
        "Porto Cervo",
        "Sardinia",
        "Italy",
        (41.13, 41.17),
        (9.52, 9.58),
        "Mediterranean",
        ["June", "July", "August", "September"],
        120,
        luxury=0.70,
        family=0.30,
        nightlife=0.40,
        beach=0.85,
        nature=0.35,
        culture=0.30,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS"],
        base_price_gbp=(749, 1349),
    ),
    # ── Croatia ─────────────────────────────────────────────────────────────
    DestinationCluster(
        "Dubrovnik",
        "Dalmatia",
        "Croatia",
        (42.65, 42.69),
        (18.08, 18.14),
        "Mediterranean",
        ["May", "June", "July", "August", "September", "October"],
        130,
        luxury=0.60,
        family=0.40,
        nightlife=0.35,
        beach=0.65,
        nature=0.45,
        culture=0.80,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI"],
        base_price_gbp=(649, 1149),
    ),
    DestinationCluster(
        "Hvar Town",
        "Split-Dalmatia",
        "Croatia",
        (43.17, 43.21),
        (16.43, 16.49),
        "Mediterranean",
        ["June", "July", "August", "September"],
        120,
        luxury=0.50,
        family=0.35,
        nightlife=0.65,
        beach=0.75,
        nature=0.45,
        culture=0.55,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS"],
        base_price_gbp=(649, 1149),
    ),
    # ── Morocco ─────────────────────────────────────────────────────────────
    DestinationCluster(
        "Marrakech",
        "Marrakesh-Safi",
        "Morocco",
        (31.62, 31.66),
        (-8.02, -7.96),
        "Semi-Arid",
        ["March", "April", "October", "November"],
        120,
        luxury=0.50,
        family=0.40,
        nightlife=0.30,
        beach=0.00,
        nature=0.35,
        culture=0.90,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS", "EDI"],
        base_price_gbp=(549, 999),
    ),
    DestinationCluster(
        "Agadir",
        "Souss-Massa",
        "Morocco",
        (30.40, 30.44),
        (-9.63, -9.57),
        "Semi-Arid",
        ["October", "November", "December", "January", "February", "March", "April", "May"],
        120,
        luxury=0.35,
        family=0.60,
        nightlife=0.25,
        beach=0.85,
        nature=0.25,
        culture=0.30,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS"],
        base_price_gbp=(499, 849),
    ),
    # ── Caribbean ───────────────────────────────────────────────────────────
    DestinationCluster(
        "Bridgetown",
        "Saint Michael",
        "Barbados",
        (13.08, 13.15),
        (-59.64, -59.57),
        "Tropical",
        ["December", "January", "February", "March", "April", "May"],
        120,
        luxury=0.60,
        family=0.50,
        nightlife=0.40,
        beach=0.90,
        nature=0.30,
        culture=0.40,
        airports=["LGW", "LHR", "MAN"],
        base_price_gbp=(999, 1799),
    ),
    DestinationCluster(
        "Cancún",
        "Quintana Roo",
        "Mexico",
        (21.14, 21.18),
        (-86.87, -86.82),
        "Tropical",
        ["November", "December", "January", "February", "March", "April", "May"],
        160,
        luxury=0.45,
        family=0.65,
        nightlife=0.55,
        beach=0.90,
        nature=0.25,
        culture=0.25,
        airports=["LGW", "LHR", "MAN", "BHX", "BRS"],
        base_price_gbp=(1099, 1899),
    ),
    # ── Maldives ────────────────────────────────────────────────────────────
    DestinationCluster(
        "North Malé Atoll",
        "Kaafu Atoll",
        "Maldives",
        (4.17, 4.25),
        (73.47, 73.56),
        "Tropical",
        ["November", "December", "January", "February", "March", "April"],
        160,
        luxury=0.85,
        family=0.35,
        nightlife=0.10,
        beach=0.99,
        nature=0.55,
        culture=0.20,
        airports=["LGW", "LHR"],
        base_price_gbp=(1499, 2999),
    ),
    # ── Thailand ────────────────────────────────────────────────────────────
    DestinationCluster(
        "Patong Beach",
        "Phuket",
        "Thailand",
        (7.88, 7.92),
        (98.28, 98.33),
        "Tropical",
        ["November", "December", "January", "February", "March", "April"],
        170,
        luxury=0.45,
        family=0.50,
        nightlife=0.70,
        beach=0.90,
        nature=0.30,
        culture=0.35,
        airports=["LGW", "LHR", "MAN"],
        base_price_gbp=(799, 1399),
    ),
    DestinationCluster(
        "Chaweng",
        "Koh Samui",
        "Thailand",
        (9.52, 9.56),
        (100.05, 100.10),
        "Tropical",
        ["January", "February", "March", "April", "December"],
        130,
        luxury=0.55,
        family=0.45,
        nightlife=0.50,
        beach=0.85,
        nature=0.50,
        culture=0.35,
        airports=["LGW", "LHR", "MAN"],
        base_price_gbp=(849, 1499),
    ),
]

# ---------------------------------------------------------------------------
# Description templates  (character → sentence components)
# ---------------------------------------------------------------------------
# Each hotel is assigned a character type and its description assembled from
# randomly chosen sentences from that type's pool. This ensures:
#   - hotels with the same character cluster together semantically
#   - hotels with different characters are semantically distant
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "luxury_adults": {
        "opening": [
            "An exquisite adults-only sanctuary where understated luxury meets the warmth of Mediterranean hospitality.",
            "A refined retreat for discerning travellers seeking seclusion and world-class service in equal measure.",
            "Indulge in the ultimate escape at this award-winning adults-only resort, where every detail is curated for perfection.",
            "A haven of sophistication and calm, this exclusive property redefines luxury for the most discerning of guests.",
            "Where quiet elegance and impeccable personal service combine to create an adults-only experience without compromise.",
        ],
        "setting": [
            "Perched above a secluded cove with sweeping sea views from every terrace.",
            "Set within manicured gardens leading directly to a pristine private beach.",
            "Commanding breathtaking panoramic views from its clifftop position above an azure bay.",
            "Nestled in lush tropical grounds, steps from a sheltered white-sand beach.",
            "Occupying a prime beachfront position with uninterrupted ocean views from the infinity pool.",
        ],
        "experience": [
            "The spa offers a full menu of holistic treatments, alongside an infinity pool that appears to merge with the horizon. Guests enjoy fine-dining across three distinct restaurants, each led by an executive chef.",
            "Multiple heated pools, a champagne bar, Michelin-recognised cuisine, and a dedicated personal concierge elevate every stay to something truly memorable.",
            "A rooftop pool, sommelier-curated wine cellar, butler service, and a renowned thalassotherapy spa set the standard for excellence.",
            "Immaculate service from a high staff-to-guest ratio ensures every preference is anticipated. Dinner is a nightly event in the signature restaurant overlooking the sea.",
        ],
        "closing": [
            "Half-board and all-inclusive packages available. Exclusively for guests aged 18 and over.",
            "Room-only and half-board packages available. Adults only — minimum age 16.",
            "All-inclusive and full-board options available. An adults-only retreat for true connoisseurs of the finer things.",
            "Half-board available with upgrades to all-inclusive. No guests under 18. Best enjoyed at a leisurely pace.",
        ],
    },
    "luxury_family": {
        "opening": [
            "A premium family resort that refuses to compromise on luxury, delivering a flawless experience for parents and children alike.",
            "Where five-star refinement meets the infectious energy of a family holiday — this resort excels at both, effortlessly.",
            "An exceptional resort that has mastered the art of combining grown-up elegance with outstanding facilities for younger guests.",
            "For families who refuse to sacrifice quality, this is the definitive answer — luxurious, spacious, and brilliantly equipped for children.",
            "Sophisticated yet thoroughly child-friendly, this resort proves that a premium family holiday need not involve any trade-offs.",
        ],
        "setting": [
            "Spread across beautifully landscaped grounds leading to a calm, shallow beach perfect for young children.",
            "Positioned on a gently sloping beachfront, offering safe swimming conditions and direct access to a wide sandy bay.",
            "Set in expansive tropical gardens with multiple pool areas, including a dedicated children's splash zone and lagoon pool.",
            "Fronting a Blue Flag beach with calm, family-safe waters and a purpose-built beach club with children's activity stations.",
        ],
        "experience": [
            "A fully-staffed children's club with age-specific programmes runs alongside an adults' spa and multiple fine-dining venues, ensuring parents and children are equally well catered for.",
            "A supervised kids' club for toddlers through to teens, a waterpark, and nightly entertainment combine with a full-service spa and gourmet restaurants.",
            "Family suites with interconnecting rooms, a dedicated crèche for under-fives, and an extensive children's programme sit alongside a world-class spa for adults.",
            "The kids' village offers round-the-clock entertainment, while the adults' infinity pool, spa, and rooftop cocktail bar ensure parents enjoy an equally rewarding holiday.",
        ],
        "closing": [
            "All-inclusive packages available. An outstanding choice for families with children of all ages.",
            "Half-board and all-inclusive options. Children under 12 stay free in selected room categories — please check at time of booking.",
            "Full-board and all-inclusive available. Designed for families seeking premium quality without sacrificing fun or freedom.",
            "All-inclusive recommended. Two children under 16 stay free per adult room in peak season. A truly exceptional family hotel.",
        ],
    },
    "family_beach": {
        "opening": [
            "A bright and welcoming family resort in a superb beachfront setting, perfect for a classic summer holiday.",
            "Everything you need for a fantastic family beach holiday, in one lively and well-organised resort.",
            "A popular resort consistently praised for its friendly atmosphere, excellent beach position, and great value for families.",
            "The ideal base for a carefree family beach holiday, with something to keep everyone happy from early morning to evening.",
            "A favourite with British families year after year, offering reliable quality, a great beach, and a warm, lively atmosphere.",
        ],
        "setting": [
            "Located directly on a wide, gently shelving sandy beach — ideal for young children and confident swimmers alike.",
            "Set just a short stroll from a busy but beautiful resort beach with calm, clear waters and plenty of facilities.",
            "Fronting a popular sandy bay with safe swimming, watersports hire, and a good choice of beach bars nearby.",
            "A beachfront position with direct access to a long sandy beach and a calm, shallow sea — perfect for families.",
        ],
        "experience": [
            "A large main pool and dedicated children's pool, a lively kids' club, and organised evening entertainment keep families happily occupied throughout the week.",
            "Multiple pools including a splash park, a mini-club with daily activities, and a varied evening entertainment programme cater well to all ages.",
            "The animation team runs a packed programme of pool games, beach activities, and themed evenings, while the kids' club offers supervised fun for under-12s.",
            "Well-equipped pool areas, a cheerful children's entertainment team, a good buffet restaurant, and a relaxed pool bar make this a reliable, enjoyable family choice.",
        ],
        "closing": [
            "All-inclusive and half-board packages available. Family rooms and connecting options offered on request.",
            "Bed and breakfast through to all-inclusive. An excellent all-round choice for families with children of all ages.",
            "Half-board and all-inclusive options. Popular with families from across the UK seeking a hassle-free beach holiday.",
            "All-inclusive recommended for best value. Children under 2 stay free; under-12s at reduced rates.",
        ],
    },
    "quiet_retreat": {
        "opening": [
            "A beautifully peaceful retreat for travellers who prefer tranquillity and natural beauty over busy resort life.",
            "Far from the crowds, this intimate property offers a genuinely restorative escape in stunning natural surroundings.",
            "A charming boutique hotel that prizes serenity and simplicity — the perfect antidote to the pace of modern life.",
            "Hidden away from busier resort areas, this calm and characterful property is ideal for those in search of true rest.",
            "Small, serene, and thoughtfully run — this understated retreat delivers something the larger resorts simply cannot: genuine peace.",
        ],
        "setting": [
            "Set amid fragrant pine forests with views stretching across an unspoilt coastline to the sea below.",
            "Perched on a quiet hillside overlooking a secluded bay, accessible by a winding coastal path from the village.",
            "Nestled in an olive grove on the edge of a traditional village, far removed from the tourist trail.",
            "Occupying a restored historic building in a peaceful clifftop location above a small fishing harbour.",
            "Tucked into a natural cove, with terraced gardens descending to a private pebble beach shared by just a handful of guests.",
        ],
        "experience": [
            "A small heated pool, a yoga platform, a honesty bar, and a peaceful library terrace encourage genuine unwinding without distraction.",
            "The intimate spa, a shaded reading terrace, and carefully curated local wine list create an atmosphere of quiet contemplation.",
            "Thoughtfully curated facilities — a pool, a cookery room, and a garden lounge — keep the focus on rest, conversation, and the natural setting.",
            "Slow mornings, long lunches, and unhurried evenings define the pace here. The small pool, terrace restaurant, and personal service do the rest.",
        ],
        "closing": [
            "Bed and breakfast or half-board available. Adults preferred; minimum age 16.",
            "Room-only and bed and breakfast. Best suited to couples, solo travellers, and those seeking genuine quiet.",
            "Half-board available. Not suitable for young children. A genuine escape for those who value stillness over stimulation.",
            "Half-board only. No entertainment programme — that is precisely the point. Adults and older children welcome.",
        ],
    },
    "lively_beach": {
        "opening": [
            "A vibrant and energetic resort at the heart of the action — ideal for those who love sun, sea, and a social atmosphere.",
            "Right in the middle of one of the most popular beach resorts, this lively property delivers a classic, non-stop holiday.",
            "Full of energy from the moment the sun rises to well after it sets, this resort is a firm favourite with couples and groups.",
            "An upbeat, sociable hotel with an unbeatable beachfront position — perfect for those who want activity and fun throughout the day.",
            "A busy, buzzing resort where the pool party starts at noon and the beach bar does not stop until dark.",
        ],
        "setting": [
            "Steps from the most popular stretch of beach, with bars, restaurants, and water sports concessions within easy reach.",
            "On the main promenade, with direct beach access and the resort's liveliest venues immediately to hand.",
            "Positioned at the heart of the action — between the beach and the entertainment strip — with easy access to both.",
            "Fronting a wide, busy sandy beach lined with sunbeds, beach bars, and watersports operators for hire.",
        ],
        "experience": [
            "A buzzing pool bar, themed pool parties, live music afternoons, and a packed evening entertainment schedule keep the energy high all week.",
            "Regular foam parties, live acts, and resident DJs create a fun and social atmosphere that guests keep returning to.",
            "Sports facilities, an active pool area, watersports on the beach, and a varied entertainment programme suit sociable, active guests perfectly.",
            "The daytime programme — from aqua aerobics and beach volleyball to table tennis and crazy golf — gives way to livelier entertainment after dark.",
        ],
        "closing": [
            "All-inclusive and half-board packages available. A firm favourite with couples and friend groups.",
            "Half-board and room-only options. Popular with guests aged 18–40 looking for an active, sociable holiday.",
            "All-inclusive available and recommended. Perfect for those who want a holiday full of activity, sunshine, and good company.",
            "Half-board and all-inclusive options. A lively, well-managed resort that delivers exactly what it promises.",
        ],
    },
    "nightlife_hub": {
        "opening": [
            "At the epicentre of one of the Mediterranean's most famous nightlife destinations — the party, quite simply, starts here.",
            "A stylish and strategically placed base for those who live for the night, with the island's best clubs on the doorstep.",
            "Perfectly positioned for guests who prefer to sleep by day and dance by night in one of Europe's most celebrated party destinations.",
            "The resort of choice for those who regard the beach as somewhere to recover, not to start the day — this is nightlife tourism at its most unapologetic.",
        ],
        "setting": [
            "A stone's throw from the most legendary clubs and open-air beach bars the destination has to offer.",
            "Right in the heart of the resort's entertainment district, within walking distance of every venue worth visiting.",
            "Set on the main strip, with direct beach access and a front-row position opposite the most famous open-air club on the island.",
            "Surrounded by bars, restaurants, and clubs — everything within ten minutes on foot, nothing requiring a taxi.",
        ],
        "experience": [
            "A rooftop pool with weekly DJ sessions, a cocktail bar open until the early hours, and a dedicated pre-party package make this the obvious choice for night owls.",
            "The hotel's own club hosts international DJs and themed nights throughout the high season; the pool bar is a destination well before sundown.",
            "Pool parties, foam events, and exclusive guest passes to neighbouring clubs ensure the fun starts long before heading out into the night.",
            "An ice bar, outdoor sound system, and partnership with the island's leading promoters mean every night begins and often ends right here.",
        ],
        "closing": [
            "Bed and breakfast and room-only options. Adults only. Not suitable for families or guests seeking a quiet holiday.",
            "Room-only and bed and breakfast available. Adults only. Minimum age 18. This is the real deal for committed night owls.",
            "Half-board available. Strictly adults. The ideal base for an unforgettable party holiday — but pack earplugs if you want an early night.",
            "Room-only recommended. 18+ only. Guests are advised that noise levels reflect the resort's reputation as a world-class nightlife destination.",
        ],
    },
    "cultural_explorer": {
        "opening": [
            "A thoughtfully designed hotel that places guests at the heart of the destination's rich history and culture.",
            "For travellers who want more than a beach holiday, this hotel offers effortless access to extraordinary cultural experiences.",
            "An ideal base for exploring the arts, architecture, and local life of one of the region's most rewarding destinations.",
            "History, gastronomy, and local craftsmanship are as much a part of the stay here as the pool and the sunshine.",
            "Culture-led and independently spirited, this hotel attracts guests who prefer souks and galleries to sun loungers.",
        ],
        "setting": [
            "Housed in a beautifully restored historic building at the edge of the old quarter.",
            "Situated in the cultural heart of the destination, steps from museums, galleries, and the daily rhythms of local life.",
            "Set within the ancient medina, surrounded by the sights, sounds, and aromas of a living historic city.",
            "Occupying a converted merchant's house overlooking a traditional market square, in the oldest part of the town.",
        ],
        "experience": [
            "A rooftop terrace with panoramic views, a curated library of local history, and a restaurant serving authentic regional cuisine set the tone from the moment of arrival.",
            "Stylish rooms, a courtyard plunge pool, and a dedicated concierge specialising in cultural itineraries — museums, guided walks, cooking experiences — make the most of the setting.",
            "An intimate courtyard pool, a craft workshop, and curated partnerships with local artisan studios offer a genuine connection with the destination's creative community.",
            "Local guides, private gallery visits, cooking classes with resident chefs, and excursions to archaeological sites are all arranged through the hotel's cultural programme.",
        ],
        "closing": [
            "Bed and breakfast available. Suited to independent travellers, couples, and cultural enthusiasts of all ages.",
            "Room-only and bed and breakfast. An excellent base for exploring the destination thoroughly and at your own pace.",
            "Half-board available. Ideal for curious travellers seeking a genuine, unhurried connection with the local culture and history.",
            "Bed and breakfast. Older children and teenagers with an interest in culture and food are very welcome.",
        ],
    },
    "adventure_nature": {
        "opening": [
            "A well-positioned base for active travellers intent on making the most of the destination's spectacular natural landscape.",
            "For those who prefer mountain trails and open water to sun loungers, this outdoors-focused property hits every mark.",
            "Adventure-ready and perfectly placed, this hotel serves as the ideal starting point for exploring a breathtaking natural environment.",
            "Built for guests who measure a holiday in steps climbed, waves ridden, and trails completed — not hours on a poolside lounger.",
            "An energetic, activity-driven property that attracts hikers, cyclists, divers, and anyone who finds their best rest after a hard day outdoors.",
        ],
        "setting": [
            "Surrounded by dramatic volcanic terrain, with hiking trails leading directly from the hotel grounds into protected national parkland.",
            "Perched at altitude with panoramic views over rugged mountain peaks, forested valleys, and the coastline below.",
            "Set at the edge of a protected natural area, with some of the region's finest walking, cycling, and climbing routes immediately accessible.",
            "Positioned between a pine-forested hillside and a rocky coastline, offering both inland and coastal adventure from a single base.",
            "Directly accessible from a network of marked trails — the hotel is a recognised waypoint on the regional long-distance walking route.",
        ],
        "experience": [
            "A gear storage room, drying facilities, boot-cleaning station, and substantial breakfast buffet are designed specifically around the needs of active guests.",
            "An outdoor pool, a well-equipped gym, a recovery sauna, and a sports-nutrition menu cater to guests who push themselves hard by day and want to recover well by night.",
            "Trail maps, GPS units, and equipment hire are available at the activity desk, alongside a resident outdoor guide who leads daily excursions of varying difficulty.",
            "From sunrise yoga and guided forest walks to canyoning, open-water swimming, and multi-day trekking itineraries, the activity programme covers every level of ambition.",
        ],
        "closing": [
            "Half-board and bed and breakfast available. Best suited to active adults and older children aged 10 and over.",
            "Bed and breakfast. Suitable for families with adventurous children; some activities have minimum age requirements.",
            "Room-only and half-board. An outstanding choice for those who measure a holiday in trail metres and open-water hours.",
            "Half-board recommended. All fitness levels welcome — the activity desk will match the programme to your ability and ambition.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Name word pools (character-appropriate)
# ---------------------------------------------------------------------------

_NAME_PARTS: dict[str, dict[str, list[str]]] = {
    "luxury_adults": {
        "adj": ["Grand", "Royal", "Imperial", "Majestic", "Elite", "Prestige", "Premier", "Noble"],
        "noun": [
            "Palace",
            "Château",
            "Manor",
            "Residence",
            "Retreat",
            "Estates",
            "Villas",
            "Reserve",
        ],
        "suffix": ["", " Hotel", " Resort", " & Spa"],
    },
    "luxury_family": {
        "adj": ["Grand", "Royal", "Premium", "Prestige", "Superior", "Classic"],
        "noun": ["Palace", "Resort", "Bay", "Shores", "Palms", "Sands", "Cove"],
        "suffix": [" Resort", " Hotel & Spa", " Family Resort", " Beach Resort"],
    },
    "family_beach": {
        "adj": ["Sunny", "Blue", "Golden", "Happy", "Tropical", "Ocean", "Coral", "Azure"],
        "noun": ["Sands", "Bay", "Beach", "Palms", "Cove", "Shores", "Paradise", "Waves"],
        "suffix": [" Hotel", " Resort", " Beach Hotel", " Holiday Resort", ""],
    },
    "quiet_retreat": {
        "adj": [
            "Serene",
            "Tranquil",
            "Peaceful",
            "Hidden",
            "Secluded",
            "Quiet",
            "Intimate",
            "Verdant",
        ],
        "noun": ["Cove", "Garden", "Grove", "Terrace", "Vista", "Haven", "Nook", "Hideaway"],
        "suffix": [" Retreat", " Boutique Hotel", "", " House", " Hotel"],
    },
    "lively_beach": {
        "adj": ["Sol", "Aqua", "Sun", "Blue", "Brisa", "Viva", "Fun", "Wave"],
        "noun": ["Beach", "Bay", "Club", "Sands", "Shore", "Strip", "Costa", "March"],
        "suffix": [" Hotel", " Beach Club", " Resort", " Beach Hotel", ""],
    },
    "nightlife_hub": {
        "adj": ["Neon", "Club", "Urban", "Pulse", "Vibe", "Electric", "Nova", "Ultra"],
        "noun": ["Strip", "Plaza", "Hub", "Scene", "Night", "Beat", "Stage", "Deck"],
        "suffix": [" Hotel", " Beach Hotel", " Club Hotel", ""],
    },
    "cultural_explorer": {
        "adj": [
            "Heritage",
            "Classic",
            "Ancient",
            "Medina",
            "Historic",
            "Artisan",
            "Kasba",
            "Old Town",
        ],
        "noun": ["House", "Riad", "Palace", "Quarter", "Court", "Hall", "Gallery", "Square"],
        "suffix": [" Hotel", " Boutique", "", " Residence", " Suites"],
    },
    "adventure_nature": {
        "adj": ["Peak", "Trail", "Verde", "Wild", "Summit", "Eco", "Natura", "Forest"],
        "noun": ["Lodge", "Base", "Camp", "Ridge", "Point", "Station", "Park", "Cliff"],
        "suffix": [" Hotel", " Lodge", "", " Adventure Hotel", " Retreat"],
    },
}

# ---------------------------------------------------------------------------
# Amenity and activity pools
# ---------------------------------------------------------------------------

_AMENITIES: dict[str, list[str]] = {
    "basic": [
        "outdoor pool",
        "sun terrace",
        "bar",
        "restaurant",
        "free Wi-Fi",
        "air conditioning",
        "24-hour reception",
    ],
    "comfort": [
        "gym",
        "pool bar",
        "buffet restaurant",
        "room service",
        "laundry service",
        "concierge",
    ],
    "family": [
        "kids' club",
        "children's pool",
        "playground",
        "animation team",
        "family rooms",
        "babysitting service",
    ],
    "waterpark": ["water slides", "wave pool", "splash park", "aqua park"],
    "luxury": [
        "spa",
        "infinity pool",
        "fine-dining restaurant",
        "butler service",
        "private beach",
        "rooftop terrace",
        "wine cellar",
    ],
    "nightlife": ["nightclub", "DJ bar", "rooftop pool", "cocktail lounge", "foam party facility"],
    "wellness": [
        "yoga studio",
        "sauna",
        "jacuzzi",
        "steam room",
        "massage treatments",
        "meditation garden",
    ],
    "activity": [
        "tennis court",
        "water sports centre",
        "bicycle hire",
        "diving centre",
        "snorkelling equipment",
    ],
    "business": ["meeting rooms", "business centre"],
}

_ACTIVITIES: dict[str, list[str]] = {
    "beach": [
        "swimming",
        "snorkelling",
        "paddleboarding",
        "kayaking",
        "beach volleyball",
        "pedalos",
    ],
    "water_sports": ["jet skiing", "parasailing", "water skiing", "banana boat", "windsurfing"],
    "boat": [
        "boat trips",
        "glass-bottom boat",
        "catamaran cruises",
        "sunset sailing",
        "fishing trips",
    ],
    "outdoor": ["hiking", "cycling", "rock climbing", "horse riding", "quad biking"],
    "nature": [
        "bird watching",
        "whale watching",
        "dolphin spotting",
        "guided nature walks",
        "snorkelling tours",
    ],
    "cultural": [
        "guided city tours",
        "cooking classes",
        "wine tasting",
        "museum visits",
        "market tours",
    ],
    "wellness": ["yoga", "pilates", "meditation", "spa treatments"],
    "nightlife": ["beach bar crawl", "club entry packages", "boat parties"],
    "family": ["kids' club activities", "beach games", "mini golf", "table tennis"],
    "golf": ["golf (nearby course)"],
}

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

_STAR_WEIGHTS: list[list[float]] = [
    # weights for stars 1–5, indexed by luxury band 0–4
    [0.20, 0.35, 0.30, 0.12, 0.03],  # low luxury (0.0–0.2)
    [0.10, 0.25, 0.38, 0.22, 0.05],  # medium-low
    [0.04, 0.15, 0.38, 0.33, 0.10],  # medium
    [0.01, 0.07, 0.27, 0.42, 0.23],  # medium-high
    [0.00, 0.02, 0.13, 0.40, 0.45],  # high luxury (0.8–1.0)
]


def _luxury_band(luxury: float) -> int:
    return min(4, int(luxury * 5))


def _pick_star_rating(rng: random.Random, luxury: float) -> int:
    weights = _STAR_WEIGHTS[_luxury_band(luxury)]
    return rng.choices([1, 2, 3, 4, 5], weights=weights)[0]


def _pick_customer_rating(rng: random.Random, star_rating: int) -> float:
    base = 5.5 + star_rating * 0.7
    rating = base + rng.gauss(0, 0.5)
    return round(min(10.0, max(1.0, rating)), 1)


def _pick_board_types(rng: random.Random, dest: DestinationCluster, star_rating: int) -> list[str]:
    all_boards = ["room_only", "bed_and_breakfast", "half_board", "full_board", "all_inclusive"]
    # Lower stars and beach/family destinations lean all-inclusive
    ai_weight = 0.3 + dest.beach * 0.3 + dest.family * 0.2 - (star_rating - 3) * 0.05
    ai_weight = max(0.05, min(0.85, ai_weight))
    if rng.random() < ai_weight:
        boards = ["all_inclusive"]
        if rng.random() < 0.4:
            boards.append("half_board")
    else:
        boards = rng.sample(all_boards[:4], k=rng.randint(2, 4))
    return boards


def _pick_amenities(rng: random.Random, dest: DestinationCluster, star_rating: int) -> list[str]:
    chosen: list[str] = list(_AMENITIES["basic"])
    if star_rating >= 2:
        chosen += rng.sample(_AMENITIES["comfort"], k=min(3, len(_AMENITIES["comfort"])))
    if dest.family > 0.4 and rng.random() < dest.family:
        chosen += rng.sample(_AMENITIES["family"], k=rng.randint(2, 4))
    if dest.family > 0.5 and star_rating >= 4 and rng.random() < 0.4:
        chosen += rng.sample(_AMENITIES["waterpark"], k=rng.randint(1, 2))
    if star_rating >= 4:
        chosen += rng.sample(
            _AMENITIES["luxury"], k=rng.randint(2, min(4, len(_AMENITIES["luxury"])))
        )
    if dest.nightlife > 0.5 and rng.random() < dest.nightlife:
        chosen += rng.sample(_AMENITIES["nightlife"], k=rng.randint(1, 3))
    if star_rating >= 3:
        chosen += rng.sample(_AMENITIES["wellness"], k=rng.randint(1, 3))
    if dest.beach > 0.4 or dest.nature > 0.4:
        chosen += rng.sample(_AMENITIES["activity"], k=rng.randint(1, 3))
    return sorted(set(chosen))


def _pick_activities(rng: random.Random, dest: DestinationCluster) -> list[str]:
    chosen: list[str] = []
    if dest.beach > 0.3:
        chosen += rng.sample(_ACTIVITIES["beach"], k=rng.randint(2, 4))
    if dest.beach > 0.6:
        chosen += rng.sample(_ACTIVITIES["water_sports"], k=rng.randint(1, 3))
        chosen += rng.sample(_ACTIVITIES["boat"], k=rng.randint(1, 2))
    if dest.nature > 0.3:
        chosen += rng.sample(_ACTIVITIES["outdoor"], k=rng.randint(1, 3))
        chosen += rng.sample(_ACTIVITIES["nature"], k=rng.randint(1, 2))
    if dest.culture > 0.4:
        chosen += rng.sample(_ACTIVITIES["cultural"], k=rng.randint(1, 3))
    if dest.nightlife > 0.5:
        chosen += rng.sample(_ACTIVITIES["nightlife"], k=rng.randint(1, 2))
    chosen += rng.sample(_ACTIVITIES["wellness"], k=1)
    chosen += rng.sample(_ACTIVITIES["family"], k=rng.randint(1, 2))
    return sorted(set(chosen))


def _pick_tags(
    rng: random.Random, character: str, dest: DestinationCluster, star_rating: int
) -> list[str]:
    tags: set[str] = set()
    if dest.beach > 0.5:
        tags.add("beach")
    if dest.nature > 0.5:
        tags.update(["nature", "scenic"])
    if dest.culture > 0.5:
        tags.update(["culture", "history"])
    if dest.nightlife > 0.5:
        tags.add("nightlife")
    if star_rating >= 4:
        tags.add("luxury")
    if star_rating <= 2:
        tags.add("budget")
    character_tags: dict[str, list[str]] = {
        "luxury_adults": ["luxury", "adults-only", "spa", "romantic"],
        "luxury_family": ["luxury", "family", "premium", "kids-club"],
        "family_beach": ["family", "beach", "all-inclusive", "pool"],
        "quiet_retreat": ["quiet", "peaceful", "boutique", "adults-only"],
        "lively_beach": ["beach", "lively", "watersports", "pool-bar"],
        "nightlife_hub": ["nightlife", "adults-only", "party", "bars"],
        "cultural_explorer": ["culture", "history", "local-experience", "authentic"],
        "adventure_nature": ["adventure", "hiking", "nature", "active"],
    }
    tags.update(character_tags.get(character, []))
    return sorted(tags)


def _classify_character(
    rng: random.Random,
    star_rating: int,
    family_friendly: bool,
    adults_only: bool,
    dest: DestinationCluster,
) -> str:
    if adults_only and star_rating >= 4:
        return "luxury_adults"
    if family_friendly and star_rating >= 4:
        return "luxury_family"
    if dest.nightlife >= 0.75:
        return "nightlife_hub"
    if dest.culture >= 0.80:
        return "cultural_explorer"
    if dest.nature >= 0.70 and not family_friendly:
        return "adventure_nature"
    if adults_only:
        return "quiet_retreat" if rng.random() < 0.6 else "lively_beach"
    if family_friendly:
        return "family_beach"
    # Neither explicitly set — use destination weights to decide
    scores = {
        "lively_beach": dest.beach * 0.5 + dest.nightlife * 0.5,
        "quiet_retreat": (1 - dest.nightlife) * 0.5 + dest.nature * 0.3,
        "adventure_nature": dest.nature * 0.6 + (1 - dest.beach) * 0.2,
        "cultural_explorer": dest.culture * 0.7,
        "family_beach": dest.family * 0.6 + dest.beach * 0.2,
    }
    return max(scores, key=lambda k: scores[k])


def _pick_hotel_name(rng: random.Random, character: str, destination: str) -> str:
    parts = _NAME_PARTS[character]
    adj = rng.choice(parts["adj"])
    noun = rng.choice(parts["noun"])
    suffix = rng.choice(parts["suffix"])
    # Occasionally include the destination name for realism
    if rng.random() < 0.3:
        return f"{destination} {adj} {noun}{suffix}".strip()
    return f"{adj} {noun}{suffix}".strip()


def _build_description(rng: random.Random, character: str) -> str:
    tpl = _TEMPLATES[character]
    parts = [
        rng.choice(tpl["opening"]),
        rng.choice(tpl["setting"]),
        rng.choice(tpl["experience"]),
        rng.choice(tpl["closing"]),
    ]
    return " ".join(parts)


def _pick_available_months(rng: random.Random, dest: DestinationCluster) -> list[str]:
    peak = set(dest.peak_months)
    shoulder: list[str] = []
    for m in _ALL_MONTHS:
        if m not in peak:
            idx = _ALL_MONTHS.index(m)
            # Months adjacent to peak are likely shoulder season
            neighbours = {_ALL_MONTHS[(idx - 1) % 12], _ALL_MONTHS[(idx + 1) % 12]}
            if neighbours & peak and rng.random() < 0.55:
                shoulder.append(m)
    months = sorted(peak | set(shoulder), key=_ALL_MONTHS.index)
    return months


def _generate_one(rng: random.Random, dest: DestinationCluster, idx: int) -> TravelProduct:
    star_rating = _pick_star_rating(rng, dest.luxury)

    # Determine family/adults character
    # Destinations with high family weight lean family_friendly,
    # high luxury + low family lean adults_only
    adult_prob = (dest.luxury * 0.5 + (1 - dest.family) * 0.5) * 0.6
    family_prob = dest.family * 0.7
    roll = rng.random()
    if dest.beach == 0.0:
        # Non-beach destinations (Marrakech) never apply beach-driven rules
        family_friendly = roll < family_prob * 0.5
        adults_only = not family_friendly and roll > 0.6
    elif roll < family_prob and star_rating <= 4:
        family_friendly = True
        adults_only = False
    elif roll > (1 - adult_prob):
        family_friendly = False
        adults_only = True
    else:
        family_friendly = False
        adults_only = False

    character = _classify_character(rng, star_rating, family_friendly, adults_only, dest)

    lat = round(rng.uniform(*dest.lat_range), 5)
    lon = round(rng.uniform(*dest.lon_range), 5)

    # Beach distance: beach-heavy destinations mostly < 0.5 km
    if dest.beach > 0.8:
        beach_km = round(rng.triangular(0.02, 2.5, 0.2), 2)
    elif dest.beach > 0.4:
        beach_km = round(rng.triangular(0.1, 5.0, 1.0), 2)
    else:
        beach_km = round(rng.uniform(1.0, 15.0), 2)

    airport_km = round(rng.triangular(5.0, 60.0, 18.0), 1)

    board_types = _pick_board_types(rng, dest, star_rating)
    amenities = _pick_amenities(rng, dest, star_rating)
    activities = _pick_activities(rng, dest)
    tags = _pick_tags(rng, character, dest, star_rating)

    # Price: base × star multiplier × board multiplier × noise
    base = rng.uniform(*dest.base_price_gbp)
    star_mult = {1: 0.50, 2: 0.72, 3: 1.00, 4: 1.55, 5: 2.60}[star_rating]
    board_mult = (
        1.25
        if "all_inclusive" in board_types
        else 1.15
        if "full_board" in board_types
        else 1.00
        if "half_board" in board_types
        else 0.85
    )
    price = round(base * star_mult * board_mult * rng.uniform(0.82, 1.18), 2)

    customer_rating = _pick_customer_rating(rng, star_rating)

    # Airport selection: longer-haul destinations have fewer UK airports
    n_airports = rng.randint(1, min(len(dest.airports), 6))
    airports = sorted(rng.sample(dest.airports, k=n_airports))

    available_months = _pick_available_months(rng, dest)
    hotel_name = _pick_hotel_name(rng, character, dest.destination)
    description = _build_description(rng, character)

    return TravelProduct(
        id=f"hotel_{idx:06d}",
        hotel_name=hotel_name,
        hotel_description=description,
        destination=dest.destination,
        region=dest.region,
        country=dest.country,
        latitude=lat,
        longitude=lon,
        star_rating=star_rating,
        customer_rating=customer_rating,
        family_friendly=family_friendly,
        adults_only=adults_only,
        amenities=amenities,
        board_types=board_types,
        beach_distance_km=beach_km,
        airport_distance_km=airport_km,
        activities=activities,
        tags=tags,
        available_departure_airports=airports,
        price_per_person_gbp=price,
        available_months=available_months,
        climate_zone=dest.climate_zone,
        peak_season_months=dest.peak_months,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dataset(seed: int = 42) -> list[TravelProduct]:
    """Generate the full synthetic hotel dataset deterministically.

    Using a local Random instance (not the global random module) ensures
    the seed affects only this generator and nothing else in the process.
    """
    rng = random.Random(seed)
    products: list[TravelProduct] = []
    idx = 1
    for dest in _DESTINATIONS:
        for _ in range(dest.count):
            products.append(_generate_one(rng, dest, idx))
            idx += 1
    return products


def save_dataset(products: list[TravelProduct], output_path: Path) -> None:
    """Write products to a JSON Lines file (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for product in products:
            fh.write(product.model_dump_json() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic travel hotel dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "hotels.jsonl",
        help="Output path for the JSON Lines file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    print(f"Generating dataset with seed={args.seed} ...")
    products = generate_dataset(seed=args.seed)

    save_dataset(products, args.output)

    # Summary statistics
    from collections import Counter

    countries = Counter(p.country for p in products)
    stars = Counter(p.star_rating for p in products)
    n_family = sum(1 for p in products if p.family_friendly)
    n_adults = sum(1 for p in products if p.adults_only)

    print(
        f"\nGenerated {len(products):,} hotels across {len(set(p.destination for p in products))} destinations."
    )
    print(f"Saved to: {args.output}\n")

    print("By country:")
    for country, count in sorted(countries.items(), key=lambda x: -x[1]):
        print(f"  {country:<25} {count:>4}")

    print("\nBy star rating:")
    for star in range(1, 6):
        bar = "★" * star
        print(f"  {bar:<5} {stars[star]:>4}")

    print(
        f"\nFamily-friendly: {n_family:,}  |  Adults-only: {n_adults:,}  |  General: {len(products) - n_family - n_adults:,}"
    )


if __name__ == "__main__":
    main()
