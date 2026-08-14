"""Reranker protocol — the contract every reranking backend must satisfy.

Any class with the right method qualifies structurally (no inheritance needed).
This makes mocking trivial: just build an object with the right attributes.

Implementations
---------------
LocalCrossEncoderReranker — sentence-transformers cross-encoder (local, offline)
BedrockReranker           — AWS Bedrock reranking API (Milestone 12, optional)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from travel_ai_search.retrieval.types import Hit


@runtime_checkable
class Reranker(Protocol):
    """Protocol for neural reranking backends.

    Takes a query and a list of candidate hits from the retrieval + fusion
    stage, re-scores them, and returns the top_k most relevant hits.

    Callers should pass at least top_k hits; passing fewer is safe — the
    implementation returns all hits if len(hits) <= top_k.

    Cross-encoder scores replace the original retrieval scores in the
    returned hits; the original scores are discarded.
    """

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        *,
        top_k: int,
    ) -> list[Hit]:
        """Re-score and return the top_k most relevant hits."""
        ...
