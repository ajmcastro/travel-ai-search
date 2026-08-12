"""Bulk ingestion of TravelProduct documents into OpenSearch."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from travel_ai_search.domain.models import TravelProduct
from travel_ai_search.ingestion.index import INDEX_NAME


@dataclass
class IngestResult:
    indexed: int
    errors: int
    elapsed_seconds: float


def load_products(path: Path) -> list[TravelProduct]:
    """Load all TravelProducts from a JSON Lines file (one JSON object per line)."""
    products: list[TravelProduct] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                products.append(TravelProduct.model_validate_json(line))
    return products


def to_document(product: TravelProduct) -> dict[str, Any]:
    """Serialise a TravelProduct to an OpenSearch document dict.

    Adds a 'location' geo_point derived from latitude/longitude so that
    geo-distance queries can be used in Milestone 5+.
    """
    doc: dict[str, Any] = product.model_dump()
    doc["location"] = {"lat": product.latitude, "lon": product.longitude}
    return doc


def _actions(products: list[TravelProduct], index: str) -> Iterator[dict[str, Any]]:
    """Yield one bulk action dict per product."""
    for product in products:
        yield {"_index": index, "_id": product.id, "_source": to_document(product)}


def ingest(
    client: OpenSearch,
    products: list[TravelProduct],
    *,
    batch_size: int = 500,
    index: str = INDEX_NAME,
) -> IngestResult:
    """Bulk-index products into OpenSearch and return a result summary.

    Using the product's own id as the document _id makes re-ingestion idempotent:
    running ingest twice updates existing documents rather than creating duplicates.

    A forced index refresh after bulk ensures documents are immediately searchable
    (OpenSearch normally refreshes every second by default).
    """
    start = time.perf_counter()
    success, error_info = bulk(
        client,
        _actions(products, index),
        chunk_size=batch_size,
        raise_on_error=False,
    )
    client.indices.refresh(index=index)
    error_count = len(error_info) if isinstance(error_info, list) else int(error_info)
    return IngestResult(
        indexed=int(success),
        errors=error_count,
        elapsed_seconds=time.perf_counter() - start,
    )
