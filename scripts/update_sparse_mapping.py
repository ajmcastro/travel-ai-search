"""Non-destructive mapping update: add the splade_vector rank_features field — Milestone 17.

Use this when the index already exists (created before Milestone 17) and you want
to add SPLADE support WITHOUT recreating or re-ingesting the index.

    make update-sparse-mapping

After running, generate the sparse embeddings:
    make generate-sparse-embeddings

OpenSearch allows adding new fields to an existing mapping at any time; existing
documents are unaffected (the new field is simply absent until updated).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_ai_search.config.settings import get_settings
from travel_ai_search.infrastructure.opensearch import create_client


def main() -> None:
    settings = get_settings()
    index = settings.opensearch_index_name

    print(f"\nConnecting to OpenSearch at {settings.opensearch_host}:{settings.opensearch_port} …")
    client = create_client(settings)
    try:
        client.info()
    except Exception as exc:
        print(f"ERROR: {exc}")
        print("  Start OpenSearch with: make up")
        sys.exit(1)

    if not client.indices.exists(index=index):
        print(f"ERROR: Index '{index}' does not exist.")
        print("  Create and ingest first: make create-index && make ingest")
        sys.exit(1)

    print(f"Adding splade_vector (rank_features) field to index '{index}' …")
    client.indices.put_mapping(
        index=index,
        body={"properties": {"splade_vector": {"type": "rank_features"}}},
    )
    print("  Done — existing documents are unaffected (splade_vector field absent until encoded).")
    print("  Next: make generate-sparse-embeddings")
    print()


if __name__ == "__main__":
    main()
