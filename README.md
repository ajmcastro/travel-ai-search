# Travel AI Search

An educational, production-quality project demonstrating modern AI search architectures — from BM25 lexical retrieval through dense vector search, hybrid fusion, neural reranking, and query understanding — applied to a synthetic travel/hotel dataset.

> This project is built incrementally. Each milestone adds one concept cleanly, with tests and explanation. See `docs/PROJECT_SPEC.md` for the full learning objectives.

---

## Current status: Milestone 0 — Project scaffold

Completed milestones: **0**

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | ≥ 0.5 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Docker](https://docs.docker.com/get-docker/) | ≥ 24 | docker.com |

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone <repo-url> travel-ai-search && cd travel-ai-search

# 2. Install Python dependencies
uv sync

# 3. Start OpenSearch
docker compose up -d

# 4. Wait ~30 s for OpenSearch to initialise, then verify connectivity
uv run python scripts/healthcheck.py
```

Expected output:
```
Connecting to OpenSearch at localhost:9200 ...
  OpenSearch version : 2.15.0
  Cluster name       : docker-cluster
  Cluster status     : green
  Nodes              : 1
Health check passed.
```

---

## Development commands

```bash
# Install / sync dependencies
uv sync

# Unit tests (no infrastructure required)
uv run pytest

# Integration tests (requires OpenSearch running)
uv run pytest tests/integration

# Linting
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Type checking
uv run mypy src
```

---

## Configuration

Copy `.env.example` to `.env` and adjust if needed. All settings have sensible defaults for local development:

```bash
cp .env.example .env
```

The most common override — if OpenSearch runs on a different host or port:
```bash
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
```

---

## Project structure (Milestone 0)

```
src/travel_ai_search/
├── config/
│   └── settings.py          # Pydantic settings from env vars
└── infrastructure/
    └── opensearch.py        # OpenSearch client factory

scripts/
└── healthcheck.py           # Verify OpenSearch connectivity

tests/
├── unit/                    # Always runnable, no infrastructure
└── integration/             # Requires OpenSearch
```

Modules for retrieval, embeddings, reranking etc. are introduced in later milestones.

---

## Stopping OpenSearch

```bash
docker compose down          # stop containers, keep data volume
docker compose down -v       # stop containers and delete data volume
```

---

## Architecture decisions

See `docs/` for architecture notes and ADRs (added progressively with each milestone).
