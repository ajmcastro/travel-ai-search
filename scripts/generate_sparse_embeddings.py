"""Offline script: generate SPLADE sparse embeddings for all hotels — Milestone 17.

Run after ingesting the dataset:
    make ingest
    make generate-sparse-embeddings

The script loads all hotels from hotels.jsonl, encodes each hotel's text with
LocalSparseProvider (in batches), and updates each OpenSearch document with the
resulting splade_vector rank_features field.  Existing sparse vectors are overwritten
(idempotent: safe to re-run after changing top_k or model).

Pre-requisite: the index must have the splade_vector rank_features field in its
mapping.  If you created the index before Milestone 17, run:
    make update-sparse-mapping
Or recreate the index from scratch:
    make create-index && make ingest && make generate-embeddings
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opensearchpy.helpers import bulk  # type: ignore[import-untyped]

from travel_ai_search.config.settings import get_settings
from travel_ai_search.domain.models import build_embedding_text
from travel_ai_search.embeddings.sparse import LocalSparseProvider
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.ingestion.ingestor import load_products

HOTELS_PATH = Path("data/processed/hotels.jsonl")
BATCH_SIZE = 16  # smaller than dense embeddings — SPLADE is more memory-intensive


def _validate_mapping(client: object, index: str) -> None:
    """Abort early if splade_vector is not mapped as rank_features.

    OpenSearch dynamic mapping stores {str: float} dicts as nested object/float fields,
    which the rank_feature query cannot use.  The index must be recreated to fix the type.
    """
    from opensearchpy import OpenSearch  # noqa: PLC0415

    assert isinstance(client, OpenSearch)
    mapping = client.indices.get_mapping(index=index)
    props = mapping[index]["mappings"].get("properties", {})
    sv = props.get("splade_vector", {})
    actual_type = sv.get("type")
    if actual_type == "rank_features":
        return  # correct
    if actual_type is None and not sv:
        # Field not yet in mapping — add it now.
        print("  splade_vector field not in mapping; adding rank_features mapping …")
        client.indices.put_mapping(
            index=index,
            body={"properties": {"splade_vector": {"type": "rank_features"}}},
        )
        return
    # Field exists with the wrong type (most likely object/float from dynamic mapping).
    print("\nERROR: splade_vector is mapped as an incompatible type in OpenSearch.")
    print(f"  Current type: {actual_type or 'object (dynamic)'}")
    print("  Required type: rank_features")
    print()
    print("  This happens when generate-sparse-embeddings runs before the mapping")
    print("  is declared.  OpenSearch cannot change a field type in place.")
    print()
    print("  Fix: recreate the index and re-run the pipeline:")
    print("    uv run python scripts/create_index.py --recreate")
    print("    make ingest")
    print("    make generate-embeddings")
    print("    make generate-sparse-embeddings")
    sys.exit(1)


def main() -> None:
    settings = get_settings()

    print(f"\nConnecting to OpenSearch at {settings.opensearch_host}:{settings.opensearch_port} …")
    client = create_client(settings)
    try:
        client.info()
    except Exception as exc:
        print(f"ERROR: {exc}")
        print("  Start OpenSearch with: make up")
        sys.exit(1)

    if not HOTELS_PATH.exists():
        print(f"ERROR: Dataset not found at {HOTELS_PATH}")
        print("  Generate it with: make generate-data && make ingest")
        sys.exit(1)

    print(f"Loading hotels from {HOTELS_PATH} …")
    hotels = load_products(HOTELS_PATH)
    print(f"  {len(hotels):,} hotels loaded")

    print(f"\nLoading SPLADE model '{settings.splade_model_name}' …")
    print("  (Downloads ~300 MB on first run; cached in ~/.cache/huggingface/)")
    t0 = time.perf_counter()
    provider = LocalSparseProvider(
        settings.splade_model_name,
        top_k=settings.splade_top_k_terms,
    )
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    _validate_mapping(client, settings.opensearch_index_name)

    print(f"\nGenerating sparse embeddings in batches of {BATCH_SIZE} …")
    index = settings.opensearch_index_name
    total = len(hotels)
    indexed = 0
    t_start = time.perf_counter()

    for batch_start in range(0, total, BATCH_SIZE):
        batch = hotels[batch_start : batch_start + BATCH_SIZE]
        texts = [build_embedding_text(h) for h in batch]
        sparse_vecs = provider.encode_batch(texts)

        actions = [
            {
                "_op_type": "update",
                "_index": index,
                "_id": hotel.id,
                "doc": {"splade_vector": sparse_vec},
            }
            for hotel, sparse_vec in zip(batch, sparse_vecs)
        ]
        bulk(client, actions, raise_on_error=True)
        indexed += len(batch)

        elapsed = time.perf_counter() - t_start
        rate = indexed / elapsed
        remaining = (total - indexed) / rate if rate > 0 else 0
        print(
            f"  {indexed:>5,}/{total:,}  "
            f"({100 * indexed / total:5.1f}%)  "
            f"{rate:.1f} docs/s  "
            f"ETA {remaining:.0f}s"
        )

    client.indices.refresh(index=index)

    elapsed_total = time.perf_counter() - t_start
    print(
        f"\nDone. {indexed:,} documents updated in {elapsed_total:.1f}s  "
        f"({indexed / elapsed_total:.1f} docs/s)"
    )
    print(f"  Index: {index}")
    print(f"  SPLADE field: splade_vector  (top_k_terms={settings.splade_top_k_terms})")
    print()


if __name__ == "__main__":
    main()
