"""Offline script: generate dense embeddings for all hotels and update OpenSearch.

Run after indexing the dataset:
    make ingest
    make generate-embeddings

The script reads hotels.jsonl, generates embeddings via LocalEmbeddingProvider
(in batches for efficiency), and updates each document in OpenSearch with its
embedding_vector field.  Existing embeddings are overwritten (idempotent).

Why offline rather than at ingest time?
  Embedding generation is CPU-bound and takes ~5–10 minutes for 5,000 hotels
  on CPU.  Separating it from ingestion means the BM25 index is available
  immediately and embedding can happen in the background or be rerun with a
  different model without recreating the schema.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make src importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opensearchpy.helpers import bulk  # type: ignore[import-untyped]

from travel_ai_search.config.settings import get_settings
from travel_ai_search.domain.models import build_embedding_text
from travel_ai_search.embeddings.local import LocalEmbeddingProvider
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.ingestion.ingestor import load_products

HOTELS_PATH = Path("data/processed/hotels.jsonl")
BATCH_SIZE = 64  # documents per embedding batch


def main() -> None:
    settings = get_settings()

    # ── Connectivity check ────────────────────────────────────────────────────
    print(f"\nConnecting to OpenSearch at {settings.opensearch_host}:{settings.opensearch_port} …")
    client = create_client(settings)
    try:
        client.info()
    except Exception as exc:
        print(f"ERROR: {exc}")
        print("  Start OpenSearch with: make up")
        sys.exit(1)

    # ── Load hotels ───────────────────────────────────────────────────────────
    if not HOTELS_PATH.exists():
        print(f"ERROR: Dataset not found at {HOTELS_PATH}")
        print("  Generate it with: make generate-data")
        sys.exit(1)

    print(f"Loading hotels from {HOTELS_PATH} …")
    hotels = load_products(HOTELS_PATH)
    print(f"  {len(hotels):,} hotels loaded")

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"\nLoading embedding model '{settings.embedding_model_name}' …")
    print("  (Downloads ~80 MB on first run; cached thereafter)")
    t0 = time.perf_counter()
    provider = LocalEmbeddingProvider(settings.embedding_model_name)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s  (dimension={provider.dimension})")

    if provider.dimension != settings.embedding_dimension:
        print(
            f"WARNING: model dimension ({provider.dimension}) does not match "
            f"settings.embedding_dimension ({settings.embedding_dimension}). "
            f"The index mapping uses {settings.embedding_dimension} — recreate "
            f"the index if you changed the model."
        )

    # ── Generate and index embeddings ─────────────────────────────────────────
    print(f"\nGenerating embeddings in batches of {BATCH_SIZE} …")
    index = settings.opensearch_index_name
    total = len(hotels)
    indexed = 0
    t_start = time.perf_counter()

    for batch_start in range(0, total, BATCH_SIZE):
        batch = hotels[batch_start : batch_start + BATCH_SIZE]
        texts = [build_embedding_text(h) for h in batch]
        vectors = provider.embed_batch(texts)

        actions = [
            {
                "_op_type": "update",
                "_index": index,
                "_id": hotel.id,
                "doc": {"embedding_vector": vector},
            }
            for hotel, vector in zip(batch, vectors)
        ]
        bulk(client, actions, raise_on_error=True)
        indexed += len(batch)

        elapsed = time.perf_counter() - t_start
        rate = indexed / elapsed
        remaining = (total - indexed) / rate if rate > 0 else 0
        print(
            f"  {indexed:>5,}/{total:,}  "
            f"({100 * indexed / total:5.1f}%)  "
            f"{rate:.0f} docs/s  "
            f"ETA {remaining:.0f}s"
        )

    # Force refresh so embeddings are immediately searchable
    client.indices.refresh(index=index)

    elapsed_total = time.perf_counter() - t_start
    print(
        f"\nDone. {indexed:,} documents updated in {elapsed_total:.1f}s  "
        f"({indexed / elapsed_total:.0f} docs/s)"
    )
    print(f"  Index: {index}")
    print()


if __name__ == "__main__":
    main()
