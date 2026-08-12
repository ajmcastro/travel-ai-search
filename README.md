# Travel AI Search

An educational, production-quality project demonstrating modern AI search architectures — from BM25 lexical retrieval through dense vector search, hybrid fusion, neural reranking, and query understanding — applied to a synthetic travel/hotel dataset.

> Built incrementally by milestone. Each milestone adds one concept cleanly, with tests and explanation. See [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) for the full learning objectives.

---

## Current status: Milestone 2 — OpenSearch mappings and ingestion

| # | Milestone | Status |
|---|---|---|
| 0 | Project scaffold | Complete |
| 1 | Synthetic travel dataset | Complete |
| 2 | OpenSearch mappings and ingestion | **Complete** |
| 3 | BM25 lexical retrieval | Pending |
| 4 | Evaluation framework and BM25 baseline | Pending |
| 5 | Embeddings and vector retrieval | Pending |
| 6 | Hybrid retrieval | Pending |
| 7 | RRF and alternative fusion | Pending |
| 8 | Cross-encoder reranking | Pending |
| 9 | Query understanding and structured constraints | Pending |
| 10 | Query rewriting | Pending |
| 11 | Multi-query retrieval | Pending |
| 12 | AWS Bedrock providers | Pending |
| 13 | RAG / travel knowledge base | Pending |
| 14 | Graph-enhanced retrieval prototype | Pending |
| 15 | Production API, observability, resilience | Pending |

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
git clone https://github.com/ajmcastro/travel-ai-search.git && cd travel-ai-search

# 2. Install Python dependencies
make install

# 3. Copy environment config
make env

# 4. Start OpenSearch
make up

# 5. Wait ~30 s for OpenSearch to initialise, then verify connectivity
make health
```

Expected output from `make health`:
```
Connecting to OpenSearch at localhost:9200 ...
  OpenSearch version : 2.15.0
  Cluster name       : docker-cluster
  Cluster status     : green
  Nodes              : 1
Health check passed.
```

---

## Data pipeline

These steps build the full dataset and index it into OpenSearch.

```bash
# Generate the synthetic hotel dataset (~5,470 hotels, deterministic seed)
make generate-data

# Create the OpenSearch index with the correct field mappings
make create-index

# Bulk-index all hotels into OpenSearch
make ingest
```

Expected output from `make ingest`:
```
Loaded 5,470 products.
Ingesting into 'travel_hotels' in batches of 500 ...

Ingestion complete:
  Indexed:  5,470
  Errors:   0
  Duration: 0.9s
```

To recreate the index from scratch (drops all indexed data):

```bash
uv run python scripts/create_index.py --recreate
make ingest
```

---

## Development commands

```bash
make check            # full quality gate: lint + format check + types + unit tests
make test             # unit tests only (no infrastructure required)
make test-integration # integration tests (requires: make up)
make test-all         # unit + integration
make lint             # ruff check
make lint-fix         # ruff check --fix
make fmt              # ruff format
make typecheck        # mypy
make serve            # FastAPI dev server  [Milestone 3+]
make evaluate         # run search evaluation  [Milestone 4+]
```

---

## Configuration

Copy `.env.example` to `.env` (or run `make env`) and adjust as needed. All settings have sensible defaults for local development.

| Environment variable | Default | Description |
|---|---|---|
| `OPENSEARCH_HOST` | `localhost` | OpenSearch hostname |
| `OPENSEARCH_PORT` | `9200` | OpenSearch port |
| `OPENSEARCH_USE_SSL` | `false` | Enable TLS |
| `OPENSEARCH_VERIFY_CERTS` | `false` | Verify TLS certificates |
| `OPENSEARCH_INDEX_NAME` | `travel_hotels` | Index name (override for staging/test) |
| `LOG_LEVEL` | `INFO` | Application log level |
| `ENVIRONMENT` | `development` | `development` or `production` |

---

## Project structure

```
src/travel_ai_search/
├── config/
│   └── settings.py          # Pydantic settings, loaded from env vars / .env
├── domain/
│   └── models.py            # TravelProduct model + build_embedding_text()
├── ingestion/
│   ├── index.py             # OpenSearch index mapping and CRUD
│   └── ingestor.py          # Bulk ingestion: load_products(), ingest()
└── infrastructure/
    └── opensearch.py        # OpenSearch client factory

scripts/
├── generate_dataset.py      # Generate synthetic JSONL dataset
├── create_index.py          # Create OpenSearch index
├── ingest_data.py           # Bulk-index dataset into OpenSearch
└── healthcheck.py           # Verify OpenSearch connectivity

data/
├── processed/
│   └── hotels.jsonl         # Generated dataset (gitignored)
└── evaluation/              # Evaluation results (Milestone 4+)

tests/
├── unit/                    # No infrastructure required (81 tests)
└── integration/             # Requires OpenSearch running (19 tests)
```

---

## Index mapping

The `travel_hotels` index maps each hotel field to the OpenSearch type that best serves the query patterns for that field:

| Field | Type | Role |
|---|---|---|
| `hotel_name` | `text` (english) + `.keyword` | BM25 search + exact match |
| `hotel_description` | `text` (english) | BM25 full-text search |
| `amenities`, `activities` | `text` (english) + `.keyword` | BM25 search + facets |
| `destination`, `tags` | `keyword` + `.text` | Exact filter + BM25 |
| `country`, `region`, `board_types` | `keyword` | Exact filters, aggregations |
| `available_months`, `climate_zone` | `keyword` | Exact filters, aggregations |
| `star_rating` | `integer` | Range queries |
| `price_per_person_gbp`, `customer_rating` | `float` | Range queries |
| `beach_distance_km`, `airport_distance_km` | `float` | Range queries |
| `family_friendly`, `adults_only` | `boolean` | Boolean filters |
| `location` | `geo_point` | Geo-distance queries (Milestone 5+) |

---

## Stopping OpenSearch

```bash
make down         # stop containers, keep data volume
make down-v       # stop containers and delete data volume
```

---

## Architecture decisions

See [`docs/`](docs/) for architecture notes and ADRs (added progressively with each milestone). Experiment logs are recorded in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) from Milestone 3 onwards.
