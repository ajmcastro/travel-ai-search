# Travel AI Search

An educational, production-quality project demonstrating modern AI search architectures — from BM25 lexical retrieval through dense vector search, hybrid fusion, neural reranking, and query understanding — applied to a synthetic travel/hotel dataset.

> Built incrementally by milestone. Each milestone adds one concept cleanly, with tests and explanation. See [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) for the full learning objectives.

---

## Current status: Milestone 5 — Embeddings and vector retrieval

| # | Milestone | Status |
|---|---|---|
| 0 | Project scaffold | ✅ Complete |
| 1 | Synthetic travel dataset | ✅ Complete |
| 2 | OpenSearch mappings and ingestion | ✅ Complete |
| 3 | BM25 lexical retrieval | ✅ Complete |
| 4 | Evaluation framework and BM25 baseline | ✅ Complete |
| 5 | Embeddings and vector retrieval | ✅ **Complete** |
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

### BM25 vs Vector baseline (K=10, 62 queries, 10 query classes)

| Metric | BM25 | Vector | Δ |
|---|---|---|---|
| NDCG@10 | 0.5007 | **0.6940** | +38.6% |
| MRR | 0.6842 | **0.8688** | +27.0% |
| HitRate@10 | 0.8226 | **1.0000** | +21.6% |
| Precision@10 | 0.6145 | **0.7790** | +26.8% |
| Latency p50 | 24 ms | 11 ms | faster |
| Latency p95 | 45 ms | 135 ms | more variable |

**Key findings:**
- Vector retrieval completely closes the destination gap: `exact_destination` NDCG jumped from 0.18 → 0.84 (+358%). Embedding models encode place names as geometric concepts; "Tenerife" is close to Canary Island hotels even without the word in their descriptions.
- HitRate@10 = 1.000 for vector — every query in the golden set now finds at least one relevant hotel.
- BM25 remains competitive for exact-term queries and has lower p95 latency.
- Hybrid retrieval (M6) will aim to combine both.

Full details in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

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

# Create the OpenSearch index with the correct field mappings (knn enabled)
make create-index

# Bulk-index all hotels into OpenSearch
make ingest

# Generate dense embeddings and write to OpenSearch (Milestone 5+)
# Downloads all-MiniLM-L6-v2 (~80 MB) on first run; ~14 s for 5,470 hotels
make generate-embeddings
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
make generate-embeddings
```

---

## Evaluation

```bash
# Build the golden relevance dataset (one-time; produces 62 queries, 48,675 judgments)
uv run python scripts/build_golden_dataset.py

# Run BM25 evaluation (prints table + saves JSON to data/evaluation/results/)
make evaluate

# Run vector evaluation (requires: make generate-embeddings first)
uv run python scripts/evaluate.py --strategy vector
```

---

## Search API

```bash
# Start the API server
make serve
```

Available endpoints:

| Endpoint | Description |
|---|---|
| `GET /health` | OpenSearch connectivity check |
| `GET /search/lexical?q=...` | BM25 lexical search |
| `GET /search/vector?q=...` | Dense vector (ANN) search |

Example searches:

```bash
# BM25 — keyword match
curl "localhost:8000/search/lexical?q=family+beach+resort&country=Spain&family_friendly=true&max_price=1000"

# Vector — semantic match (finds Tenerife hotels even without the word in description)
curl "localhost:8000/search/vector?q=sunny+beach+holiday+in+the+Canary+Islands"
```

Response shape:
```json
{
  "hits": [
    {
      "id": "hotel_001234",
      "score": 12.4,
      "hotel_name": "Playa Familiar Resort",
      "destination": "Benidorm",
      "country": "Spain",
      "star_rating": 4,
      "price_per_person_gbp": 849.0,
      "family_friendly": true
    }
  ],
  "total": 1847,
  "took_ms": 22,
  "facets": {
    "countries": [{"key": "Spain", "count": 1241}, ...],
    "star_ratings": [{"key": "4", "count": 612}, ...],
    "board_types": [...],
    "climate_zones": [...]
  }
}
```

Supported query parameters:

| Parameter | Type | Description |
|---|---|---|
| `q` | string | Free-text search query |
| `top_k` | int | Number of results (default: 10) |
| `country` | string | Exact country filter |
| `destination` | string | Exact destination filter |
| `family_friendly` | bool | Family-friendly hotels only |
| `adults_only` | bool | Adults-only hotels only |
| `min_star_rating` | int | Minimum star rating (1–5) |
| `max_price` | float | Maximum price per person (GBP) |
| `month` | string | Available month (e.g. `July`) |
| `airport` | string | Departure airport code (e.g. `MAN`) |

---

## Development commands

```bash
make check              # full quality gate: lint + format check + types + unit tests
make test               # unit tests only (no infrastructure required)
make test-integration   # integration tests (requires: make up + make ingest)
make test-all           # unit + integration
make lint               # ruff check
make lint-fix           # ruff check --fix
make fmt                # ruff format
make typecheck          # mypy
make serve              # FastAPI dev server
make evaluate           # run BM25 evaluation against golden dataset
make generate-embeddings # generate dense embeddings for all hotels (Milestone 5+)
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
| `TOP_K` | `10` | Default result count for search and evaluation |
| `LOG_LEVEL` | `INFO` | Application log level |
| `ENVIRONMENT` | `development` | `development` or `production` |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model for dense embeddings |
| `EMBEDDING_DIMENSION` | `384` | Embedding dimension (must match model and index mapping) |

---

## Project structure

```
src/travel_ai_search/
├── api/
│   ├── app.py               # FastAPI app, lifespan (OpenSearch + embedding provider)
│   ├── deps.py              # FastAPI dependency injection
│   └── routes/
│       └── search.py        # GET /search/lexical, GET /search/vector
├── config/
│   └── settings.py          # Pydantic settings, loaded from env vars / .env
├── domain/
│   └── models.py            # TravelProduct model + build_embedding_text()
├── embeddings/
│   ├── base.py              # EmbeddingProvider Protocol
│   └── local.py             # LocalEmbeddingProvider (sentence-transformers)
├── evaluation/
│   ├── dataset.py           # GoldenQuery, GoldenDataset, load_dataset()
│   ├── evaluator.py         # evaluate(), EvaluationReport, SearchFn type
│   └── metrics.py           # P@K, Recall@K, HitRate@K, RR, AP, NDCG@K
├── ingestion/
│   ├── index.py             # OpenSearch index mapping and CRUD (knn_vector)
│   └── ingestor.py          # Bulk ingestion: load_products(), ingest()
├── retrieval/
│   ├── lexical.py           # BM25 multi-match search: lexical_search()
│   └── vector.py            # ANN search: vector_search(), _build_vector_query()
└── infrastructure/
    └── opensearch.py        # OpenSearch client factory

scripts/
├── generate_dataset.py      # Generate synthetic JSONL dataset
├── create_index.py          # Create OpenSearch index (knn enabled)
├── ingest_data.py           # Bulk-index dataset into OpenSearch
├── generate_embeddings.py   # Offline: embed all hotels, update OpenSearch
├── build_golden_dataset.py  # Build golden evaluation dataset (one-time)
├── evaluate.py              # Run evaluation: --strategy bm25 | vector
└── healthcheck.py           # Verify OpenSearch connectivity

data/
├── processed/
│   └── hotels.jsonl         # Generated dataset (gitignored)
└── evaluation/
    ├── golden_queries.jsonl # 62 queries, 48,675 graded judgments
    └── results/             # JSON evaluation results per run

tests/
├── unit/                    # No infrastructure required (202 tests)
└── integration/             # Requires OpenSearch running (69 tests)
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
| `location` | `geo_point` | Geo-distance queries |
| `embedding_vector` | `knn_vector` (384-dim, HNSW/lucene) | ANN search (Milestone 5+) |

---

## Stopping OpenSearch

```bash
make down         # stop containers, keep data volume
make down-v       # stop containers and delete data volume
```

---

## Architecture decisions

See [`docs/RATIONALE_PER_MILESTONE.md`](docs/RATIONALE_PER_MILESTONE.md) for the IR concepts, design choices, and rationale behind each milestone. Experiment logs and measured results are in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
