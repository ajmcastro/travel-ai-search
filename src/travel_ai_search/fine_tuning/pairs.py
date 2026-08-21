"""Training pair utilities for bi-encoder fine-tuning — Milestone 19.

A TrainingPair is a (query, positive, negative) triplet used by
MultipleNegativesRankingLoss for contrastive training.  The negative must be
a document that appears relevant to the query (a *hard* negative) but is
actually NOT relevant — hotels that BM25 ranks highly for the query but are
absent from the graded relevance judgements.

Hard negatives force the model to learn fine-grained travel distinctions —
e.g. "adults-only spa resort" vs "family beach hotel with spa" — rather than
just separating hotels from noise.

All functions in this module are pure (no I/O, no side-effects) so they can
be unit-tested without any infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingPair:
    """Immutable (query, positive, negative) triplet for contrastive training."""

    query: str
    positive: str
    negative: str


def select_hard_negatives(
    bm25_hit_ids: list[str],
    known_relevant_ids: set[str],
) -> list[str]:
    """Filter BM25 hits to only those absent from the known-relevant set.

    A hotel retrieved by BM25 that is NOT in the golden graded set is a hard
    negative: the lexical model believes it is relevant but we know it is not.
    Including it in training forces the bi-encoder to distinguish between hotels
    that share surface keywords but differ semantically.

    Args:
        bm25_hit_ids: Ordered hotel IDs from BM25 retrieval (rank-preserving).
        known_relevant_ids: Hotel IDs with grade ≥ 1 in the golden dataset.

    Returns:
        Subset of bm25_hit_ids not in known_relevant_ids (original order preserved).
    """
    return [hid for hid in bm25_hit_ids if hid not in known_relevant_ids]


def build_training_pairs(
    query_text: str,
    positive_texts: list[str],
    negative_texts: list[str],
    max_pairs_per_query: int = 5,
) -> list[TrainingPair]:
    """Pair each positive text with a hard negative (round-robin over negatives).

    Produces at most ``min(len(positive_texts), max_pairs_per_query)`` pairs.
    When there are fewer negatives than positives, negatives cycle round-robin
    so every positive is covered.

    Args:
        query_text: The golden query string.
        positive_texts: Embedding texts for hotels with grade ≥ MIN_POSITIVE_GRADE.
        negative_texts: Embedding texts for hard-negative hotels.
        max_pairs_per_query: Cap on pairs produced per query — prevents queries
            with many positives from dominating the training corpus.

    Returns:
        List of TrainingPair instances; empty when either input list is empty.
    """
    if not positive_texts or not negative_texts:
        return []

    n_neg = len(negative_texts)
    pairs: list[TrainingPair] = []
    for i, pos in enumerate(positive_texts[:max_pairs_per_query]):
        neg = negative_texts[i % n_neg]
        pairs.append(TrainingPair(query=query_text, positive=pos, negative=neg))
    return pairs
