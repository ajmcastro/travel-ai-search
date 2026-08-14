"""KnowledgeRetriever: semantic search over the destination knowledge base."""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch

from travel_ai_search.embeddings.base import EmbeddingProvider
from travel_ai_search.rag.index import KNOWLEDGE_INDEX_NAME
from travel_ai_search.rag.knowledge import DestinationKnowledge

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Retrieves relevant destination knowledge documents for a query.

    Uses the same embedding model as the hotel retriever — the embedding
    dimension must match the knowledge index's knn_vector field.

    Optionally filters by country when the query understanding engine has
    extracted a country entity, preventing a "beach holiday in Greece" query
    from retrieving knowledge about Caribbean beaches based on pure semantic
    similarity.
    """

    def __init__(
        self,
        client: OpenSearch,
        embedding_provider: EmbeddingProvider,
        *,
        index: str = KNOWLEDGE_INDEX_NAME,
        top_k: int = 3,
    ) -> None:
        self._client = client
        self._embedding_provider = embedding_provider
        self._index = index
        self._top_k = top_k

    def retrieve(
        self,
        query: str,
        *,
        country: str | None = None,
    ) -> list[DestinationKnowledge]:
        """Return up to top_k destination knowledge documents for the query.

        Embeds the query and performs knn ANN search on the knowledge index.
        When country is supplied, applies it as a pre-filter so only destinations
        in that country are considered — the k nearest among matching docs.

        Raises on OpenSearch errors (e.g. index not found); callers are
        responsible for catching and degrading gracefully.
        """
        vector = self._embedding_provider.embed(query)

        knn_clause: dict[str, Any] = {"vector": vector, "k": self._top_k}
        if country:
            knn_clause["filter"] = {"term": {"country": country}}

        body: dict[str, Any] = {
            "_source": {"excludes": ["embedding_vector"]},
            "query": {"knn": {"embedding_vector": knn_clause}},
            "size": self._top_k,
        }
        response = self._client.search(index=self._index, body=body)
        hits: list[dict[str, Any]] = response["hits"]["hits"]
        return [DestinationKnowledge.model_validate(hit["_source"]) for hit in hits]
