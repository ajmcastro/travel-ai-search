"""LocalCrossEncoderReranker — local CPU cross-encoder reranking.

Model choice: cross-encoder/ms-marco-MiniLM-L-6-v2
  • ~22 M parameters, ~86 MB download — runs on CPU in ≈ 15–25 ms per pair
  • Trained on MS-MARCO passage ranking; generalises well to travel queries
  • Raw logit output (typically −10 to +10); larger = more relevant
  • Faster alternatives: L-2-v2 (~12 M params); better: L-12-v2 (~66 M params)

The cross-encoder reads the full (query, document) pair jointly in one
forward pass. Attention heads can compare query tokens directly against
document tokens, capturing fine-grained relevance signals that a bi-encoder
(which encodes query and document independently) cannot.
"""

from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import CrossEncoder

from travel_ai_search.retrieval.types import Hit

logger = logging.getLogger(__name__)


def _build_reranking_text(source: dict[str, Any]) -> str:
    """Construct a compact text representation of a hotel for cross-encoder input.

    Includes the fields most likely to carry query-relevant signal:
    - hotel_name and location: primary identity
    - hotel_description: free-text semantic content
    - activities and tags: structured attributes that answer "what can I do?"

    Excludes structural / numeric fields (price, star_rating, airport codes)
    that carry no semantic content for a relevance model.
    """
    activities = ", ".join(source.get("activities", []))
    tags = ", ".join(source.get("tags", []))
    parts = [
        source.get("hotel_name", ""),
        f"{source.get('destination', '')}, {source.get('country', '')}",
        source.get("hotel_description", ""),
        f"Activities: {activities}" if activities else "",
        f"Tags: {tags}" if tags else "",
    ]
    return " | ".join(p for p in parts if p.strip())


class LocalCrossEncoderReranker:
    """Cross-encoder reranker backed by a local sentence-transformers model.

    Typical usage in the pipeline:
      1. Retrieval + fusion produces rerank_k candidates (e.g. 50).
      2. This reranker scores each (query, candidate) pair.
      3. Candidates are re-sorted by cross-encoder score; top_k are returned.

    The cross-encoder score replaces the original retrieval score. Scores are
    raw logits — comparable within one query, not across queries.

    Load this once at application startup and reuse across requests, since
    model loading (~1 s) dominates individual inference (~20 ms per pair).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model: CrossEncoder = CrossEncoder(model_name)
        logger.info("Loaded cross-encoder reranker: %s", model_name)

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        *,
        top_k: int,
    ) -> list[Hit]:
        """Score each (query, document) pair and return the top_k highest-scoring hits.

        Cross-encoder scores replace the original retrieval scores in the
        returned Hit objects so the caller always sees a consistent score field.
        """
        if not hits:
            return []

        texts = [_build_reranking_text(hit.source) for hit in hits]
        pairs = [[query, text] for text in texts]
        raw_scores: list[float] = self._model.predict(pairs).tolist()  # type: ignore[arg-type]

        scored = sorted(
            zip(hits, raw_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [Hit(id=h.id, score=s, source=h.source) for h, s in scored[:top_k]]
