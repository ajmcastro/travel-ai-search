"""OpenSearch knowledge index definition for travel destinations.

The knowledge index stores one document per destination island/region
(~30 documents) and is searched semantically using the same embedding model
as the hotel index.  The embedding_vector dimension must therefore match
Settings.embedding_dimension — if you switch embedding providers, recreate
both the hotel index and the knowledge index.
"""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

KNOWLEDGE_INDEX_NAME = "travel_destinations"


def build_knowledge_index_body(embedding_dimension: int) -> dict[str, Any]:
    """Return the knowledge index mapping for the given embedding dimension."""
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "knn": True,
            "knn.algo_param.ef_search": 100,
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "destination": {
                    "type": "keyword",
                    "fields": {"text": {"type": "text", "analyzer": "standard"}},
                },
                "country": {"type": "keyword"},
                "region": {"type": "keyword"},
                "description": {"type": "text", "analyzer": "english"},
                "climate": {"type": "text", "analyzer": "english"},
                "geographic_note": {"type": "text", "analyzer": "english"},
                "best_months": {"type": "keyword"},
                "activities": {"type": "keyword"},
                "character_tags": {"type": "keyword"},
                "similar_destinations": {"type": "keyword"},
                "family_suitability": {"type": "keyword"},
                "nightlife_level": {"type": "keyword"},
                "beach_quality": {"type": "keyword"},
                "embedding_vector": {
                    "type": "knn_vector",
                    "dimension": embedding_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 16,
                        },
                    },
                },
            }
        },
    }


def knowledge_index_exists(client: OpenSearch, index: str = KNOWLEDGE_INDEX_NAME) -> bool:
    """Return True if the knowledge index exists."""
    return bool(client.indices.exists(index=index))


def create_knowledge_index(
    client: OpenSearch,
    embedding_dimension: int,
    *,
    index: str = KNOWLEDGE_INDEX_NAME,
    recreate: bool = False,
) -> None:
    """Create the knowledge index with the given embedding dimension.

    If the index already exists and recreate=False, this is a no-op.
    If recreate=True, the existing index is deleted first (all data lost).
    """
    if knowledge_index_exists(client, index):
        if not recreate:
            return
        delete_knowledge_index(client, index)
    client.indices.create(index=index, body=build_knowledge_index_body(embedding_dimension))


def delete_knowledge_index(client: OpenSearch, index: str = KNOWLEDGE_INDEX_NAME) -> None:
    """Delete the knowledge index, silently ignoring a missing index."""
    try:
        client.indices.delete(index=index)
    except NotFoundError:
        pass
