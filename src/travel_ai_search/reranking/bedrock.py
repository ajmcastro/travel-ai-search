"""Cohere Rerank v3.5 provider via AWS Bedrock.

Model: cohere.rerank-v3-5:0
  Access: invoke_model with the same bedrock-runtime client used for embeddings
  Scores: calibrated 0–1 relevance scores (unlike raw logits from cross-encoder)
  Multilingual: supports 100+ languages out of the box

API shape (invoke_model body):
  request:  {"query": str, "documents": list[str], "top_n": int, "return_documents": false}
  response: {"results": [{"index": int, "relevance_score": float}, ...]}

The document text is built from the hit's source dict using the same
_build_reranking_text() helper as LocalCrossEncoderReranker — both rerankers
see identical document representations, enabling fair A/B quality comparison.

For local reranking without AWS credentials, use LocalCrossEncoderReranker.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from travel_ai_search.retrieval.types import Hit

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "cohere.rerank-v3-5:0"


def _build_reranking_text(source: dict[str, Any]) -> str:
    """Construct a compact text representation of a hotel for the reranker.

    Identical field selection and pipe-separated format to LocalCrossEncoderReranker
    so both rerankers receive the same document view — enables direct score comparison.
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


class BedrockReranker:
    """Reranker backed by Cohere Rerank v3.5 on AWS Bedrock.

    Satisfies Reranker Protocol (structural typing — no import needed).
    """

    def __init__(
        self,
        client: Any,
        *,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        self._client = client
        self._model_id = model_id

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        *,
        top_k: int,
    ) -> list[Hit]:
        """Rerank hits using Cohere Rerank on Bedrock.

        Builds document text from each hit's source dict, sends all (query, docs)
        to Bedrock in one call, then maps the returned indices back to original hits
        with Cohere relevance scores replacing the original retrieval scores.

        Exceptions propagate — the calling pipeline catches and falls back to
        the pre-reranking fused results.
        """
        if not hits:
            return []

        doc_texts = [_build_reranking_text(hit.source) for hit in hits]
        body = json.dumps(
            {
                "query": query,
                "documents": doc_texts,
                "top_n": top_k,
                "return_documents": False,
            }
        )
        response = self._client.invoke_model(
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result: dict[str, Any] = json.loads(response["body"].read())

        return [
            Hit(
                id=hits[item["index"]].id,
                score=float(item["relevance_score"]),
                source=hits[item["index"]].source,
            )
            for item in result["results"]
        ]
