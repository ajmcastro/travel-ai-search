"""Deep health-check endpoint (Milestone 15).

GET /health returns a structured status report for each system component:
  - opensearch   : cluster ping (is the database reachable?)
  - index        : does the hotel index exist?
  - embedding    : is an embedding provider loaded?
  - reranker     : is a reranker loaded? (may be "disabled" — not an error)
  - graph        : is the destination graph built?

HTTP status codes
-----------------
200  status = "ok" or "degraded" (degraded = non-critical optional
     component missing/disabled, e.g. reranker or graph)
503  status = "unavailable" (OpenSearch unreachable — the API cannot
     serve any search requests)

Liveness vs readiness
---------------------
Kubernetes uses two checks:
  - Liveness:  is the process alive?  → A simple GET /health returning 2xx.
  - Readiness: can it serve traffic?  → A deep check like this one.
This endpoint functions as both in a single call; split into /health/live
and /health/ready if separate Kubernetes probes are needed.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response
from opensearchpy import OpenSearch

from travel_ai_search.api.deps import get_os_client, get_settings
from travel_ai_search.config.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


def _component_status(components: dict[str, dict[str, str]]) -> str:
    """Derive overall status from individual component statuses."""
    statuses = {c["status"] for c in components.values()}
    if "error" in statuses:
        return "unavailable"
    if "missing" in statuses:
        return "degraded"
    return "ok"


@router.get("")
def health(
    request: Request,
    response: Response,
    client: OpenSearch = Depends(get_os_client),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Return detailed health status for each system component.

    The response always contains a top-level ``status`` field and a
    ``components`` map with one entry per subsystem.  Each entry has at
    minimum a ``status`` key (one of: ok, disabled, missing, error).
    """
    components: dict[str, dict[str, str]] = {}

    # ── OpenSearch cluster ping ────────────────────────────────────────────
    try:
        info = client.info()
        version = info.get("version", {}).get("number", "unknown")
        components["opensearch"] = {"status": "ok", "version": version}
    except Exception as exc:
        logger.warning("Health check: OpenSearch ping failed: %s", exc)
        components["opensearch"] = {"status": "error", "detail": str(exc)}

    # ── Hotel index existence ──────────────────────────────────────────────
    try:
        exists = client.indices.exists(index=settings.opensearch_index_name)
        if exists:
            components["index"] = {"status": "ok", "name": settings.opensearch_index_name}
        else:
            components["index"] = {
                "status": "missing",
                "name": settings.opensearch_index_name,
                "detail": "run 'make create-index && make ingest' to populate",
            }
    except Exception as exc:
        logger.warning("Health check: index existence check failed: %s", exc)
        components["index"] = {"status": "error", "detail": str(exc)}

    # ── Embedding provider ─────────────────────────────────────────────────
    if hasattr(request.app.state, "embedding_provider"):
        provider = request.app.state.embedding_provider
        provider_name = type(provider).__name__
        components["embedding"] = {"status": "ok", "provider": provider_name}
    else:
        components["embedding"] = {"status": "missing"}

    # ── Reranker (optional) ────────────────────────────────────────────────
    reranker = getattr(request.app.state, "reranker", None)
    if reranker is not None:
        components["reranker"] = {"status": "ok", "provider": type(reranker).__name__}
    else:
        components["reranker"] = {
            "status": "disabled",
            "detail": "set RERANKING_ENABLED=true to activate",
        }

    # ── Destination graph (optional) ───────────────────────────────────────
    graph = getattr(request.app.state, "destination_graph", None)
    if graph is not None:
        components["graph"] = {
            "status": "ok",
            "nodes": str(graph.node_count()),
            "edges": str(graph.edge_count()),
        }
    else:
        components["graph"] = {
            "status": "disabled",
            "detail": "set GRAPH_ENABLED=true to activate",
        }

    overall = _component_status(components)

    if overall == "unavailable":
        response.status_code = 503

    return {"status": overall, "components": components}
