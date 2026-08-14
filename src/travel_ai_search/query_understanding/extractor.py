"""Rule-based query understanding engine.

Extracts hard constraints from free-text travel queries using regex patterns and
keyword lookup tables.  No external API calls; deterministic and sub-millisecond.

Limitations (known and accepted for M9):
- "may" is always treated as the month May, even in non-temporal contexts.
- "march" (the verb) is treated as the month March.
- Only the first matching location / airport / month etc. is extracted.
- Destination-level extraction is skipped; island/region names map to country only,
  because dataset destinations are city-level and users rarely use exact city names.

Milestone 10 (LLM rewriting) and Milestone 12 (Bedrock) will add more capable engines.
"""

from __future__ import annotations

import re
import time

from travel_ai_search.query_understanding.models import QueryUnderstanding

# ── Month extraction ───────────────────────────────────────────────────────────

_MONTH_CANONICAL: dict[str, str] = {
    "january": "January",
    "jan": "January",
    "february": "February",
    "feb": "February",
    "march": "March",
    "mar": "March",
    "april": "April",
    "apr": "April",
    "may": "May",
    "june": "June",
    "jun": "June",
    "july": "July",
    "jul": "July",
    "august": "August",
    "aug": "August",
    "september": "September",
    "sep": "September",
    "sept": "September",
    "october": "October",
    "oct": "October",
    "november": "November",
    "nov": "November",
    "december": "December",
    "dec": "December",
}

# Sort alternation longest-first to avoid prefix shadowing (sept before sep, etc.)
_MONTH_ALTS = "|".join(sorted(_MONTH_CANONICAL, key=len, reverse=True))

# Optional leading preposition ensures "in October" is fully consumed
_MONTH_RE = re.compile(
    rf"(?:(?:in|during|for|this|next)\s+)?\b({_MONTH_ALTS})\b",
    re.IGNORECASE,
)

# ── Price extraction ────────────────────────────────────────────────────────────

# "under £2000", "up to 1500 pounds", "max £800", "budget of £1200 per person"
_PRICE_KEYWORD_RE = re.compile(
    r"(?:under|up\s+to|max(?:imum)?|below|less\s+than|budget(?:\s+of)?)\s+"
    r"[£$]?\s*(\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:pounds?|gbp|£|\$|per\s+person)?",
    re.IGNORECASE,
)

# "£1500" / "$2000" with an explicit currency symbol but no preceding keyword
_PRICE_SYMBOL_RE = re.compile(
    r"[£$]\s*(\d[\d,]*(?:\.\d+)?)\s*(?:pounds?|per\s+person)?",
    re.IGNORECASE,
)

# ── Star rating extraction ─────────────────────────────────────────────────────

_STAR_WORD_MAP: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}

_STAR_RE = re.compile(
    r"\b(one|two|three|four|five|[1-5])\s*-?\s*stars?\b",
    re.IGNORECASE,
)

# ── Departure airport extraction ───────────────────────────────────────────────
# Only the 10 UK airports present in the synthetic dataset.

_AIRPORT_MAP: dict[str, str] = {
    "london heathrow": "LHR",
    "heathrow": "LHR",
    "london gatwick": "LGW",
    "gatwick": "LGW",
    "manchester": "MAN",
    "birmingham": "BHX",
    "bristol": "BRS",
    "edinburgh": "EDI",
    "glasgow": "GLA",
    "leeds": "LBA",
    "bradford": "LBA",
    "liverpool": "LPL",
    "newcastle": "NCL",
}

_AIRPORT_ALTS = "|".join(re.escape(name) for name in sorted(_AIRPORT_MAP, key=len, reverse=True))

_AIRPORT_RE = re.compile(
    rf"(?:(?:from|departing(?:\s+from)?|leaving|via|out\s+of)\s+)?"
    rf"\b({_AIRPORT_ALTS})(?:\s+airport)?\b",
    re.IGNORECASE,
)

# ── Location extraction ────────────────────────────────────────────────────────
# Country names and island/region names that map to a country.
# Strategy: country → hard country filter; region → country filter only
# (dataset destinations are city-level; users refer to island/region names,
# not city names like "Playa de las Américas").

_COUNTRY_MAP: dict[str, str] = {
    "spain": "Spain",
    "spanish": "Spain",
    "greece": "Greece",
    "greek": "Greece",
    "turkey": "Turkey",
    "turkish": "Turkey",
    "portugal": "Portugal",
    "portuguese": "Portugal",
    "italy": "Italy",
    "italian": "Italy",
    "cyprus": "Cyprus",
    "malta": "Malta",
    "croatia": "Croatia",
    "croatian": "Croatia",
    "france": "France",
    "french": "France",
    "morocco": "Morocco",
    "moroccan": "Morocco",
    "egypt": "Egypt",
    "egyptian": "Egypt",
    "thailand": "Thailand",
    "thai": "Thailand",
    "indonesia": "Indonesia",
    "maldives": "Maldives",
    "uae": "UAE",
    "dubai": "UAE",
    "tunisia": "Tunisia",
    "tunisian": "Tunisia",
}

_REGION_TO_COUNTRY: dict[str, str] = {
    # Spain — Canary Islands
    "canary islands": "Spain",
    "tenerife": "Spain",
    "gran canaria": "Spain",
    "lanzarote": "Spain",
    "fuerteventura": "Spain",
    "canaries": "Spain",
    # Spain — Balearic Islands
    "balearic islands": "Spain",
    "mallorca": "Spain",
    "majorca": "Spain",
    "menorca": "Spain",
    "minorca": "Spain",
    "ibiza": "Spain",
    # Spain — mainland regions / cities
    "costa del sol": "Spain",
    "costa blanca": "Spain",
    "costa brava": "Spain",
    "benidorm": "Spain",
    "marbella": "Spain",
    "torremolinos": "Spain",
    # Greece — islands
    "santorini": "Greece",
    "mykonos": "Greece",
    "crete": "Greece",
    "rhodes": "Greece",
    "corfu": "Greece",
    "zakynthos": "Greece",
    "zante": "Greece",
    "kefalonia": "Greece",
    "kos": "Greece",
    "lefkada": "Greece",
    "skiathos": "Greece",
    # Portugal — regions / cities
    "algarve": "Portugal",
    "madeira": "Portugal",
    "azores": "Portugal",
    "lisbon": "Portugal",
    "porto": "Portugal",
    "albufeira": "Portugal",
    # Italy — regions / cities
    "amalfi coast": "Italy",
    "amalfi": "Italy",
    "sicily": "Italy",
    "sardinia": "Italy",
    "rome": "Italy",
    "venice": "Italy",
    "florence": "Italy",
    # Turkey — cities
    "bodrum": "Turkey",
    "antalya": "Turkey",
    "istanbul": "Turkey",
    "marmaris": "Turkey",
    "fethiye": "Turkey",
    # Egypt — resorts
    "sharm el sheikh": "Egypt",
    "sharm": "Egypt",
    "hurghada": "Egypt",
    # Morocco — cities
    "marrakech": "Morocco",
    "marrakesh": "Morocco",
    "agadir": "Morocco",
    "fes": "Morocco",
    # Thailand — resorts
    "phuket": "Thailand",
    "koh samui": "Thailand",
    "ko samui": "Thailand",
    "bangkok": "Thailand",
    "chiang mai": "Thailand",
    # UAE
    "abu dhabi": "UAE",
}

# Combined lookup: country names + region names (all map to a country)
_ALL_LOCATIONS: dict[str, str] = {**_COUNTRY_MAP, **_REGION_TO_COUNTRY}

_LOCATION_ALTS = "|".join(re.escape(name) for name in sorted(_ALL_LOCATIONS, key=len, reverse=True))

_LOCATION_RE = re.compile(
    rf"(?:(?:in|to|at|near|visiting)\s+)?(?:the\s+)?\b({_LOCATION_ALTS})\b",
    re.IGNORECASE,
)

# ── Trip-type flags ────────────────────────────────────────────────────────────

_FAMILY_RE = re.compile(
    r"\b(?:family(?:\s*-?\s*friendly)?|families|with\s+kids?|with\s+children?|for\s+kids?)\b",
    re.IGNORECASE,
)

_ADULTS_ONLY_RE = re.compile(
    r"\b(?:adults?\s*-?\s*only|no\s+kids?|no\s+children?|couples?\s+only|18\+)\b",
    re.IGNORECASE,
)

# ── Soft preference extraction ─────────────────────────────────────────────────
# Detected but NOT removed from semantic_query — they contribute to BM25/vector scoring.

_SOFT_PREF_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\ball.?inclusive\b", re.IGNORECASE), "all-inclusive"),
    (re.compile(r"\bbeach(?:front|side)?\b", re.IGNORECASE), "beach"),
    (re.compile(r"\bspa\b", re.IGNORECASE), "spa"),
    (re.compile(r"\bpool\b|\bswimming\s+pool\b", re.IGNORECASE), "pool"),
    (re.compile(r"\bsea\s+view\b|\bocean\s+view\b", re.IGNORECASE), "sea view"),
    (re.compile(r"\bnightlife\b|\bparty\b", re.IGNORECASE), "nightlife"),
    (re.compile(r"\bquiet\b|\bpeaceful\b|\brelaxing\b", re.IGNORECASE), "peaceful"),
    (re.compile(r"\bgolf\b", re.IGNORECASE), "golf"),
    (re.compile(r"\bski(?:ing)?\b", re.IGNORECASE), "ski"),
    (re.compile(r"\bhiking?\b|\btrekking?\b", re.IGNORECASE), "hiking"),
    (re.compile(r"\badventure\b", re.IGNORECASE), "adventure"),
    (re.compile(r"\bluxury\b|\bpremium\b|\bprestige\b", re.IGNORECASE), "luxury"),
    (re.compile(r"\bbudget\b|\baffordable\b|\bcheap\b", re.IGNORECASE), "budget"),
    (re.compile(r"\bromantic\b", re.IGNORECASE), "romantic"),
    (re.compile(r"\bboutique\b", re.IGNORECASE), "boutique"),
]

# ── Span-based semantic query builder ─────────────────────────────────────────

_DANGLING_PREP_RE = re.compile(
    r"^\s*\b(?:in|from|at|to|for|and|the|near|with|visiting)\b\s*",
    re.IGNORECASE,
)
_TRAILING_PREP_RE = re.compile(
    r"\s*\b(?:in|from|at|to|for|and|the|near|with|visiting)\b\s*$",
    re.IGNORECASE,
)


def _build_semantic_query(query: str, consumed: list[tuple[int, int]]) -> str:
    """Remove consumed spans from query, collapse whitespace, drop dangling prepositions."""
    if not consumed:
        return query
    # Merge overlapping spans
    spans: list[list[int]] = []
    for start, end in sorted(consumed):
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    parts: list[str] = []
    prev = 0
    for span in spans:
        start, end = span
        if start > prev:
            parts.append(query[prev:start])
        prev = end
    parts.append(query[prev:])
    text = " ".join("".join(parts).split())
    text = _DANGLING_PREP_RE.sub("", text)
    text = _TRAILING_PREP_RE.sub("", text)
    return text.strip()


# ── Engine ─────────────────────────────────────────────────────────────────────


class RuleBasedQueryUnderstandingEngine:
    """Pattern-matching query understanding engine.

    Parses a free-text travel query into hard constraints and a clean semantic
    query without any external API calls.  Sub-millisecond per query.
    """

    def understand(self, query: str) -> QueryUnderstanding:
        t0 = time.monotonic()
        consumed: list[tuple[int, int]] = []

        month = self._extract_month(query, consumed)
        max_price = self._extract_price(query, consumed)
        min_star_rating = self._extract_stars(query, consumed)
        departure_airport = self._extract_airport(query, consumed)
        country = self._extract_location(query, consumed)
        family_friendly = self._extract_family(query, consumed)
        adults_only = self._extract_adults_only(query, consumed)
        soft_preferences = self._extract_soft_prefs(query)

        semantic = _build_semantic_query(query, consumed) or query

        return QueryUnderstanding(
            original_query=query,
            semantic_query=semantic,
            month=month,
            departure_airport=departure_airport,
            max_price=max_price,
            min_star_rating=min_star_rating,
            family_friendly=family_friendly,
            adults_only=adults_only,
            country=country,
            soft_preferences=soft_preferences,
            understanding_took_ms=round((time.monotonic() - t0) * 1000),
        )

    # ── Extractors ─────────────────────────────────────────────────────────────

    def _extract_month(self, query: str, consumed: list[tuple[int, int]]) -> str | None:
        m = _MONTH_RE.search(query)
        if m is None:
            return None
        consumed.append((m.start(), m.end()))
        return _MONTH_CANONICAL[m.group(1).lower()]

    def _extract_price(self, query: str, consumed: list[tuple[int, int]]) -> float | None:
        m = _PRICE_KEYWORD_RE.search(query)
        if m is None:
            m = _PRICE_SYMBOL_RE.search(query)
        if m is None:
            return None
        consumed.append((m.start(), m.end()))
        return float(m.group(1).replace(",", ""))

    def _extract_stars(self, query: str, consumed: list[tuple[int, int]]) -> int | None:
        m = _STAR_RE.search(query)
        if m is None:
            return None
        consumed.append((m.start(), m.end()))
        return _STAR_WORD_MAP.get(m.group(1).lower())

    def _extract_airport(self, query: str, consumed: list[tuple[int, int]]) -> str | None:
        m = _AIRPORT_RE.search(query)
        if m is None:
            return None
        consumed.append((m.start(), m.end()))
        return _AIRPORT_MAP.get(m.group(1).lower())

    def _extract_location(self, query: str, consumed: list[tuple[int, int]]) -> str | None:
        m = _LOCATION_RE.search(query)
        if m is None:
            return None
        consumed.append((m.start(), m.end()))
        return _ALL_LOCATIONS.get(m.group(1).lower())

    def _extract_family(self, query: str, consumed: list[tuple[int, int]]) -> bool | None:
        m = _FAMILY_RE.search(query)
        if m is None:
            return None
        consumed.append((m.start(), m.end()))
        return True

    def _extract_adults_only(self, query: str, consumed: list[tuple[int, int]]) -> bool | None:
        m = _ADULTS_ONLY_RE.search(query)
        if m is None:
            return None
        consumed.append((m.start(), m.end()))
        return True

    def _extract_soft_prefs(self, query: str) -> list[str]:
        return [label for pattern, label in _SOFT_PREF_PATTERNS if pattern.search(query)]
