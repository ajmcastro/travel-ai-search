"""Query expansion for multi-query retrieval (Milestone 11).

Generates N variant queries from a single semantic query so that each
variant can be retrieved independently.  The union of candidates from
all variants is fused via RRF for higher recall than a single-query run.

Design
------
LocalQueryExpander produces up to three variants:
  1. Original query — always first; ensures the baseline retrieval is included.
  2. Synonym substitution — replace words with travel-domain synonyms to widen
     BM25 term matches (e.g. "beach" → "coastal", "luxury" → "premium").
  3. Context elaboration — append a domain phrase that mirrors hotel descriptions
     (e.g. "family holiday" → "... with kids club and children activities"), which
     improves ANN recall by moving the query embedding closer to relevant hotel texts.

The Protocol is `@runtime_checkable` so route handlers can use isinstance()
checks at startup without forcing a full import of concrete implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# ── Synonym table ─────────────────────────────────────────────────────────────
# Maps a lowercase word to its travel-domain synonym.  Applied word-by-word;
# only the first matching word in each query position is substituted.
# Purpose: produce a lexically distinct query that targets different BM25 term
# matches while preserving the same semantic intent.

_WORD_SYNONYMS: dict[str, str] = {
    # Ambience
    "quiet": "peaceful",
    "peaceful": "tranquil",
    "tranquil": "serene",
    "relaxing": "calm",
    "lively": "vibrant",
    # Geography / setting
    "beach": "coastal",
    "coastal": "seaside",
    "seaside": "beachfront",
    "island": "archipelago",
    # Family
    "family": "child-friendly",
    "child-friendly": "family-friendly",
    "children": "kids",
    "kids": "children",
    # Luxury
    "luxury": "premium",
    "premium": "upscale",
    "upscale": "five-star",
    "boutique": "charming",
    "charming": "intimate",
    # Wellness
    "spa": "wellness",
    "wellness": "health",
    # Couples / adults
    "romantic": "couples",
    "couples": "intimate",
    "adults": "adults-only",
    # Budget
    "budget": "affordable",
    "affordable": "value",
    "cheap": "budget-friendly",
    # Trip type
    "holiday": "vacation",
    "vacation": "break",
    "break": "getaway",
    "trip": "holiday",
    # Accommodation
    "resort": "hotel",
    "hotel": "property",
    "property": "accommodation",
    # Activities
    "adventure": "outdoor",
    "outdoor": "active",
    "nightlife": "entertainment",
    "party": "nightlife",
    "hiking": "trekking",
    "trekking": "walking",
    "golf": "golf course",
    "diving": "snorkelling",
    "snorkelling": "water sports",
}

# ── Context addition table ────────────────────────────────────────────────────
# Maps a keyword (found anywhere in the query) to a context phrase appended to
# the query.  Purpose: produce a richer, longer query whose embedding sits closer
# to hotel description vectors that naturally contain these phrases.

_CONTEXT_ADDITIONS: dict[str, str] = {
    "beach": "with beachfront access and swimming pool",
    "family": "with kids club and children activities",
    "spa": "with wellness treatments and relaxation facilities",
    "quiet": "in a peaceful setting away from nightlife",
    "luxury": "with premium amenities fine dining and butler service",
    "golf": "with golf course and putting green",
    "adults": "adults-only retreat with private pool and couples facilities",
    "ski": "with ski slopes and après-ski facilities",
    "dive": "with diving snorkelling and water sports",
    "hike": "with hiking trails and mountain scenery",
    "nightlife": "near bars clubs and entertainment venues",
    "all-inclusive": "full board unlimited drinks and activities included",
    "romantic": "with sea view couples suite and candlelit dining",
    "budget": "offering value for money with good facilities",
    "adventure": "with outdoor activities and sports facilities",
}


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class QueryExpander(Protocol):
    """Structural protocol for query expansion.

    Implementations return a list of exactly `n_queries` query strings.
    The first element MUST be `semantic_query` (or equivalent) so that
    the original retrieval path is always included in the fused results.
    """

    def expand(self, semantic_query: str, n_queries: int = 3) -> list[str]: ...


# ── Implementations ───────────────────────────────────────────────────────────


class IdentityQueryExpander:
    """Stub expander — always returns [semantic_query].

    Used in unit tests and as a no-op baseline: multi-query retrieval with
    the identity expander is equivalent to single-query hybrid RRF retrieval
    (same lists; no fusion benefit).  Useful for verifying that the
    multi_query_search interface is correct without adding recall.
    """

    def expand(self, semantic_query: str, n_queries: int = 3) -> list[str]:
        return [semantic_query]


class LocalQueryExpander:
    """Deterministic, rule-based query expander.  No API key required.

    Generates up to three distinct query variants from a semantic query:

    1. **Original** — always the first element; ensures baseline recall.
    2. **Synonym substitution** — each word replaced by a travel-domain synonym
       from `_WORD_SYNONYMS`.  Produces lexically different tokens for BM25.
       Falls back to original + " accommodation" if no synonyms found.
    3. **Context elaboration** — original query augmented with a descriptive
       phrase from `_CONTEXT_ADDITIONS` matching the first recognized keyword.
       Falls back to original + " hotel resort accommodation".

    For n_queries ≤ 3, returns exactly that many distinct variants.
    For n_queries > 3, fills extra slots with the original query (deduplication
    is left to the caller / fuser; RRF naturally handles duplicates by merging
    rank scores for the same document).
    """

    def expand(self, semantic_query: str, n_queries: int = 3) -> list[str]:
        if n_queries <= 0:
            return [semantic_query]
        if not semantic_query.strip():
            return [semantic_query] * max(1, n_queries)

        queries: list[str] = [semantic_query]

        if n_queries >= 2:
            queries.append(self._synonym_variant(semantic_query))

        if n_queries >= 3:
            queries.append(self._context_variant(semantic_query))

        while len(queries) < n_queries:
            queries.append(semantic_query)

        return queries[:n_queries]

    def _synonym_variant(self, query: str) -> str:
        words = query.split()
        substituted = [_WORD_SYNONYMS.get(w.lower(), w) for w in words]
        candidate = " ".join(substituted)
        # If no word was changed, append a generic travel term to ensure
        # the variant is distinct from the original (different BM25 tokens).
        return candidate if candidate != query else query + " accommodation"

    def _context_variant(self, query: str) -> str:
        query_lower = query.lower()
        for keyword, ctx in _CONTEXT_ADDITIONS.items():
            if keyword in query_lower:
                return f"{query} {ctx}"
        return f"{query} hotel resort accommodation"
