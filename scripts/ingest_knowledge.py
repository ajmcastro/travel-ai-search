"""Ingest destination knowledge documents into OpenSearch.

Creates the knowledge index (if it does not exist), embeds each destination
document, and bulk-indexes the results.  Only ~30 documents so a single pass
without batching is fine.

Usage:
    uv run python scripts/ingest_knowledge.py

Options (via environment variables / .env):
    OPENSEARCH_HOST, OPENSEARCH_PORT  — default localhost:9200
    EMBEDDING_MODEL_NAME              — default all-MiniLM-L6-v2
    EMBEDDING_DIMENSION               — must match the model (default 384)
    EMBEDDING_PROVIDER                — "local" or "bedrock"
    KNOWLEDGE_INDEX_NAME              — default travel_destinations

Run generate_knowledge.py first if data/knowledge/destinations.jsonl does not exist.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Allow importing from src/ without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_ai_search.config.settings import get_settings
from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.rag.index import create_knowledge_index
from travel_ai_search.rag.knowledge import DestinationKnowledge, build_knowledge_embedding_text

DATA_PATH = Path(__file__).parent.parent / "data" / "knowledge" / "destinations.jsonl"


def load_destinations() -> list[DestinationKnowledge]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run: uv run python scripts/generate_knowledge.py"
        )
    destinations = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                destinations.append(DestinationKnowledge.model_validate(json.loads(line)))
    return destinations


def main() -> None:
    settings = get_settings()
    client = create_client(settings)

    # --- Embedding provider ---
    if settings.embedding_provider == "bedrock":
        try:
            from travel_ai_search.embeddings.bedrock import BedrockEmbeddingProvider
            from travel_ai_search.infrastructure.bedrock import create_bedrock_client

            bedrock = create_bedrock_client(settings.aws_region)
            embedding_provider = BedrockEmbeddingProvider(
                bedrock,
                model_id=settings.bedrock_embedding_model_id,
                dimension=settings.bedrock_embedding_dimension,
            )
            embedding_dimension = settings.bedrock_embedding_dimension
            logger.info(
                "Using Bedrock embedding provider: %s (dim=%d)",
                settings.bedrock_embedding_model_id,
                embedding_dimension,
            )
        except Exception as exc:
            logger.warning("Bedrock provider failed (%s) — falling back to local.", exc)
            embedding_provider = LocalEmbeddingProvider(settings.embedding_model_name)
            embedding_dimension = settings.embedding_dimension
    else:
        embedding_provider = LocalEmbeddingProvider(settings.embedding_model_name)
        embedding_dimension = settings.embedding_dimension
        logger.info(
            "Using local embedding provider: %s (dim=%d)",
            settings.embedding_model_name,
            embedding_dimension,
        )

    # --- Index ---
    index_name = settings.knowledge_index_name
    create_knowledge_index(client, embedding_dimension, index=index_name)
    logger.info("Knowledge index '%s' ready.", index_name)

    # --- Load and embed ---
    destinations = load_destinations()
    logger.info("Loaded %d destination documents.", len(destinations))

    actions = []
    for dest in destinations:
        text = build_knowledge_embedding_text(dest)
        vector = embedding_provider.embed(text)
        doc = {**dest.model_dump(), "embedding_vector": vector}
        actions.append({"index": {"_index": index_name, "_id": dest.id}})
        actions.append(doc)
        logger.info("  Embedded: %s (%s)", dest.destination, dest.country)

    # --- Bulk index ---
    response = client.bulk(body=actions)
    if response.get("errors"):
        for item in response["items"]:
            if "error" in item.get("index", {}):
                logger.error("Indexing error: %s", item["index"]["error"])
        raise RuntimeError("Bulk indexing completed with errors — check logs above.")

    logger.info(
        "Indexed %d destination documents into '%s'.",
        len(destinations),
        index_name,
    )
    client.close()


if __name__ == "__main__":
    main()
