#!/usr/bin/env python
"""Verify connectivity to OpenSearch and print cluster info.

Usage:
    uv run python scripts/healthcheck.py
"""

import sys

from opensearchpy import ConnectionError, TransportError

from travel_ai_search.config.settings import Settings
from travel_ai_search.infrastructure.opensearch import create_client


def main() -> int:
    settings = Settings()
    print(f"Connecting to OpenSearch at {settings.opensearch_host}:{settings.opensearch_port} ...")

    client = create_client(settings)
    try:
        info = client.info()
        health = client.cluster.health()
    except ConnectionError as exc:
        print(f"ERROR: could not connect — {exc}")
        print("Is OpenSearch running? Try: docker compose up -d")
        return 1
    except TransportError as exc:
        print(f"ERROR: transport error — {exc}")
        return 1

    version = info["version"]["number"]
    cluster = health["cluster_name"]
    status = health["status"]
    nodes = health["number_of_nodes"]

    print(f"  OpenSearch version : {version}")
    print(f"  Cluster name       : {cluster}")
    print(f"  Cluster status     : {status}")
    print(f"  Nodes              : {nodes}")

    if status not in ("green", "yellow"):
        print(f"WARNING: cluster status is '{status}' — expected green or yellow")
        return 1

    print("Health check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
