#!/usr/bin/env python
"""Ingest the synthetic travel dataset into OpenSearch.

Usage:
    make ingest
    uv run python scripts/ingest_data.py [--input PATH] [--batch-size N]

Requires the index to exist first:
    make create-index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_ai_search.config.settings import get_settings
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.ingestion.index import index_exists
from travel_ai_search.ingestion.ingestor import ingest, load_products

_DEFAULT_INPUT = Path(__file__).parent.parent / "data" / "processed" / "hotels.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest travel hotels into OpenSearch")
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        metavar="PATH",
        help="JSON Lines file to ingest (default: data/processed/hotels.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        metavar="N",
        help="Documents per bulk request (default: 500)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}")
        print("Run 'make generate-data' first.")
        sys.exit(1)

    settings = get_settings()
    client = create_client(settings)
    index = settings.opensearch_index_name

    if not index_exists(client, index):
        print(f"Error: index '{index}' does not exist.")
        print("Run 'make create-index' first.")
        sys.exit(1)

    print(f"Loading products from {args.input} ...")
    products = load_products(args.input)
    print(f"Loaded {len(products):,} products.")

    print(f"Ingesting into '{index}' in batches of {args.batch_size} ...")
    result = ingest(client, products, batch_size=args.batch_size, index=index)

    print("\nIngestion complete:")
    print(f"  Indexed:  {result.indexed:,}")
    print(f"  Errors:   {result.errors}")
    print(f"  Duration: {result.elapsed_seconds:.1f}s")

    if result.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
