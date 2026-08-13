"""Score fusion and shared filter utilities for hybrid retrieval.

This module is the single authoritative source for:

  build_filter_clauses() — OpenSearch filter clause construction, shared by
    lexical.py, vector.py, and hybrid.py.  Previously duplicated; centralised
    here in Milestone 6.

  fuse_results() — min-max normalised weighted sum fusion of two ranked lists.
    Combines BM25 and vector scores onto a common [0, 1] scale before merging.

Design notes
------------
Score normalisation is necessary because BM25 and cosine similarity live in
different numeric ranges.  BM25 scores are unbounded (corpus- and query-
dependent); cosine similarity scores are in [0, 1] (normalised vectors).
Adding them directly lets BM25 dominate.  Min-max normalisation maps each
list independently to [0, 1] before combining.

Reciprocal Rank Fusion (RRF), an alternative that avoids score normalisation
altogether, is added in Milestone 7.
"""

from __future__ import annotations

from typing import Any

from travel_ai_search.retrieval.types import Hit

# ── Filter building ───────────────────────────────────────────────────────────


def build_filter_clauses(
    *,
    country: str | None = None,
    destination: str | None = None,
    family_friendly: bool | None = None,
    adults_only: bool | None = None,
    min_star_rating: int | None = None,
    max_price: float | None = None,
    month: str | None = None,
    airport: str | None = None,
) -> list[dict[str, Any]]:
    """Return OpenSearch filter clauses for the given filter parameters.

    Each active (non-None) parameter produces one filter clause.
    The returned list is ready to use as the value of `bool.filter`.
    """
    clauses: list[dict[str, Any]] = []
    if country is not None:
        clauses.append({"term": {"country": country}})
    if destination is not None:
        clauses.append({"term": {"destination": destination}})
    if family_friendly is not None:
        clauses.append({"term": {"family_friendly": family_friendly}})
    if adults_only is not None:
        clauses.append({"term": {"adults_only": adults_only}})
    if min_star_rating is not None:
        clauses.append({"range": {"star_rating": {"gte": min_star_rating}}})
    if max_price is not None:
        clauses.append({"range": {"price_per_person_gbp": {"lte": max_price}}})
    if month is not None:
        clauses.append({"term": {"available_months": month}})
    if airport is not None:
        clauses.append({"term": {"available_departure_airports": airport}})
    return clauses


# ── Score normalisation ───────────────────────────────────────────────────────


def _normalize_scores(hits: list[Hit]) -> dict[str, tuple[float, dict[str, Any]]]:
    """Min-max normalise hit scores to [0, 1].

    Returns a mapping from document id to (normalised_score, source).

    Edge cases:
      - Empty list → empty dict.
      - Single document → normalised score 1.0.
      - All identical scores → all normalised scores 1.0 (denominator = 0 → clamp).
    """
    if not hits:
        return {}

    scores = [h.score for h in hits]
    min_s = min(scores)
    max_s = max(scores)
    denom = max_s - min_s

    return {
        hit.id: (
            (hit.score - min_s) / denom if denom > 0.0 else 1.0,
            hit.source,
        )
        for hit in hits
    }


# ── Fusion ────────────────────────────────────────────────────────────────────


def fuse_results(
    lexical_hits: list[Hit],
    vector_hits: list[Hit],
    *,
    lexical_weight: float,
    vector_weight: float,
    top_k: int,
) -> list[Hit]:
    """Merge two ranked lists via min-max normalised weighted sum.

    Algorithm
    ---------
    1. Normalise each list independently to [0, 1].
    2. For every unique document id in the union of both lists:
         combined_score = lexical_weight × norm_bm25
                        + vector_weight  × norm_vector
       where norm_bm25 = 0.0 if the document did not appear in lexical results,
       and   norm_vector = 0.0 if the document did not appear in vector results.
    3. Sort by combined_score descending, return the top_k hits.

    Documents that appear in both lists and are ranked highly by both
    retrievers will score highest.  A document found by only one retriever
    is penalised (half score at most), reflecting lower confidence.
    """
    lex_norm = _normalize_scores(lexical_hits)
    vec_norm = _normalize_scores(vector_hits)

    all_ids = set(lex_norm) | set(vec_norm)

    merged: list[Hit] = []
    for doc_id in all_ids:
        lex_score, lex_source = lex_norm.get(doc_id, (0.0, {}))
        vec_score, vec_source = vec_norm.get(doc_id, (0.0, {}))
        source = lex_source if lex_source else vec_source

        combined = lexical_weight * lex_score + vector_weight * vec_score
        merged.append(Hit(id=doc_id, score=combined, source=source))

    merged.sort(key=lambda h: h.score, reverse=True)
    return merged[:top_k]
