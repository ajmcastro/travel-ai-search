#!/usr/bin/env python
"""Create the OpenSearch hotels index.

Usage:
    make create-index
    uv run python scripts/create_index.py [--recreate]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_ai_search.config.settings import get_settings
from travel_ai_search.infrastructure.opensearch import create_client
from travel_ai_search.ingestion.index import create_index, index_exists


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the OpenSearch hotels index")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the existing index first (WARNING: all indexed data will be lost)",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = create_client(settings)
    index = settings.opensearch_index_name

    if index_exists(client, index) and not args.recreate:
        print(f"Index '{index}' already exists. Use --recreate to overwrite.")
        return

    create_index(client, index=index, recreate=args.recreate)
    print(f"Index '{index}' created successfully.")


if __name__ == "__main__":
    main()
