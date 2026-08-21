"""Offline script: generate ColBERT token embeddings for all hotels.

Run this after dense embeddings have been generated:
    make generate-embeddings
    make generate-colbert-embeddings

The script reads hotels.jsonl, encodes each hotel description with the ColBERT
encoder (all token embeddings, not a single pooled vector), and saves the result
as a NumPy array at:

    data/processed/colbert_embeddings/<hotel_id>.npy
    shape: (d_tokens, dim) float32, L2-normalised per token

These files are loaded at query time by ColBERTReranker.rerank() for MaxSim scoring.

Why .npy files rather than an OpenSearch field?
  ColBERT v2 produces 128-dimensional embeddings.  With 128 tokens per hotel,
  that is 128 × 128 = 16,384 floats per document — 65 KB per hotel and ~350 MB
  for 5,470 hotels.  OpenSearch dense_vector fields are optimised for ANN search,
  not for fetching raw float matrices; loading from disk is faster and cheaper.

Why offline rather than at query time?
  A forward pass through a 110 M-parameter BERT model takes ~50 ms on CPU per
  document.  Re-encoding at query time across 50 candidates per query would add
  2.5 s to every search request — unacceptable for a search API.

Idempotent: existing .npy files are overwritten on each run.  To regenerate only
missing files, add a ``--skip-existing`` flag (not implemented here — full reruns
are the safe default after a model change).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make src importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_ai_search.config.settings import get_settings
from travel_ai_search.ingestion.ingestor import load_products
from travel_ai_search.reranking.colbert import ColBERTEncoder, build_colbert_document_text

HOTELS_PATH = Path("data/processed/hotels.jsonl")
OUTPUT_DIR = Path("data/processed/colbert_embeddings")
PROGRESS_EVERY = 100


def main() -> None:
    settings = get_settings()

    # ── Load hotels ───────────────────────────────────────────────────────────
    if not HOTELS_PATH.exists():
        print(f"ERROR: Dataset not found at {HOTELS_PATH}")
        print("  Generate it with: make generate-data")
        sys.exit(1)

    print(f"\nLoading hotels from {HOTELS_PATH} …")
    hotels = load_products(HOTELS_PATH)
    print(f"  {len(hotels):,} hotels loaded")

    # ── Load ColBERT encoder ──────────────────────────────────────────────────
    model_name = settings.colbert_model_name
    print(f"\nLoading ColBERT encoder '{model_name}' …")
    print("  colbert-ir/colbertv2.0 downloads ~400 MB on first run; cached thereafter.")
    print("  Alternatively, set COLBERT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2")
    print("  to reuse the already-cached MiniLM weights (384-dim instead of 128).")
    t0 = time.perf_counter()
    encoder = ColBERTEncoder(
        model_name,
        doc_maxlen=settings.colbert_doc_maxlen,
        query_maxlen=settings.colbert_query_maxlen,
    )
    print(
        f"  Encoder ready in {time.perf_counter() - t0:.1f}s  "
        f"(dim={encoder.dim}, doc_maxlen={settings.colbert_doc_maxlen})"
    )

    # ── Prepare output directory ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving token embeddings to {OUTPUT_DIR}/ …")

    # ── Encode and save ───────────────────────────────────────────────────────
    import numpy as np

    total = len(hotels)
    t_start = time.perf_counter()
    skipped = 0

    for i, hotel in enumerate(hotels):
        text = build_colbert_document_text(
            {
                "hotel_name": hotel.hotel_name,
                "destination": hotel.destination,
                "country": getattr(hotel, "country", ""),
                "hotel_description": hotel.hotel_description,
                "activities": getattr(hotel, "activities", []),
                "tags": getattr(hotel, "tags", []),
            }
        )

        if not text.strip():
            skipped += 1
            continue

        embs = encoder.encode_document(text)
        out_path = OUTPUT_DIR / f"{hotel.id}.npy"
        np.save(str(out_path), embs)

        if (i + 1) % PROGRESS_EVERY == 0 or i == total - 1:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  {i + 1:>6,}/{total:,}  "
                f"({100 * (i + 1) / total:5.1f}%)  "
                f"{rate:.1f} docs/s  "
                f"ETA {remaining:.0f}s  "
                f"shape={embs.shape}"
            )

    elapsed_total = time.perf_counter() - t_start
    saved = total - skipped
    print(
        f"\nDone. {saved:,} embeddings saved to {OUTPUT_DIR}/"
        f"  ({elapsed_total:.1f}s, {saved / elapsed_total:.1f} docs/s)"
    )
    if skipped:
        print(f"  WARNING: {skipped} hotels had empty text and were skipped.")
    storage_mb = sum(p.stat().st_size for p in OUTPUT_DIR.glob("*.npy")) / 1e6
    print(f"  Total storage: {storage_mb:.1f} MB")
    print(f"  Per-embedding shape: ({settings.colbert_doc_maxlen}, {encoder.dim}) float32")
    print()
    print("Next step: uv run python scripts/evaluate.py --strategy colbert")
    print()


if __name__ == "__main__":
    main()
