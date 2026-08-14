# Travel AI Search

An educational, production-quality project demonstrating modern AI search architectures — from BM25 lexical retrieval through dense vector search, hybrid fusion, neural reranking, and query understanding — applied to a synthetic travel/hotel dataset.

> Built incrementally by milestone. Each milestone adds one concept cleanly, with tests and explanation. See [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) for the full learning objectives.

---

## Current status: Milestone 10 — Query rewriting

| # | Milestone | Status |
|---|---|---|
| 0 | Project scaffold | ✅ Complete |
| 1 | Synthetic travel dataset | ✅ Complete |
| 2 | OpenSearch mappings and ingestion | ✅ Complete |
| 3 | BM25 lexical retrieval | ✅ Complete |
| 4 | Evaluation framework and BM25 baseline | ✅ Complete |
| 5 | Embeddings and vector retrieval | ✅ Complete |
| 6 | Hybrid retrieval | ✅ Complete |
| 7 | RRF and alternative fusion | ✅ Complete |
| 8 | Cross-encoder reranking | ✅ Complete |
| 9 | Query understanding and structured constraints | ✅ Complete |
| 10 | Query rewriting | ✅ **Complete** |
| 11 | Multi-query retrieval | Pending |
| 12 | AWS Bedrock providers | Pending |
| 13 | RAG / travel knowledge base | Pending |
| 14 | Graph-enhanced retrieval prototype | Pending |
| 15 | Production API, observability, resilience | Pending |

### BM25 vs Vector vs Hybrid vs RRF vs Rerank vs Understand vs Rewrite (K=10, 62 queries, 10 query classes)

| Metric | BM25 | Vector | Hybrid (50/50) | RRF | Rerank | Understand | Rewrite |
|---|---|---|---|---|---|---|---|
| NDCG@10 | 0.5007 | 0.6940 | 0.6003 | 0.6239 | **0.6830** | 0.6312 | 0.6130 |
| MRR | 0.6842 | 0.8688 | 0.8542 | 0.8449 | 0.8191 | **0.8620** | 0.8226 |
| HitRate@10 | 0.8226 | **1.0000** | 0.9355 | 0.9516 | 0.9516 | 0.9355 | **0.9677** |
| Precision@10 | 0.6145 | 0.7790 | 0.6823 | 0.7210 | **0.7935** | 0.7290 | 0.7242 |
| Latency p50 | 24 ms | 11 ms | 57 ms | 56 ms | 113 ms | **45 ms** | 54 ms |
| Latency p95 | 45 ms | 135 ms | 90 ms | 84 ms | 142 ms | **71 ms** | 98 ms |

**Key findings (cumulative):**
- Vector (M5): NDCG +38.6% vs BM25; `exact_destination` NDCG jumped from 0.18 → 0.84 (+358%); HitRate = 1.000.
- Hybrid (M6) — weighted sum, 50/50: overall NDCG is between BM25 and vector (0.60). Naive 50/50 fusion can *regress* from the best individual retriever when one retriever produces meaningless scores for a query class. `exact_destination` NDCG drops from 0.84 (vector) to 0.48.
- Hybrid (M7) — RRF (k=60): beats weighted-sum (+3.9% NDCG, +5.7% Precision) by using rank positions instead of raw scores. `exact_destination` recovers from 0.48 to 0.53; `activities` beats vector (0.40 vs 0.38). Still below pure vector overall.
- Rerank (M8) — RRF + cross-encoder (`ms-marco-MiniLM-L-6-v2`, 50 candidates): **highest NDCG overall (0.683)**. `exact_destination` jumps from 0.53 to 0.79 (+48%); `activities` from 0.40 to 0.57 (+43%). Cost: ~57 ms extra latency.
- Understand (M9) — rule-based QU + RRF: beats RRF on NDCG (+1.2%) and MRR (+2.0%) while being **20% faster** (45 ms vs 56 ms p50). `adults_couples` NDCG +8.9% (correct `adults_only` filter extracted). Main failure: false-positive constraints on `budget` queries (−18.4%).
- Rewrite (M10) — QU + `LocalLLMProvider` keyword expansion + RRF: **HitRate improves +3.4%** (more relevant hotels in top-10) but NDCG and MRR regress vs Understand (−2.9%, −4.6%). Classic precision-recall tradeoff: naive synonym expansion broadens recall but dilutes the ranking signal. `activities` class +14.4%. Architecture ready for real LLM (M12).

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

# Run hybrid weighted-sum evaluation
uv run python scripts/evaluate.py --strategy hybrid

# Run hybrid RRF evaluation (Milestone 7)
uv run python scripts/evaluate.py --strategy rrf

# Run reranking evaluation: RRF + cross-encoder (Milestone 8)
# Downloads cross-encoder/ms-marco-MiniLM-L-6-v2 (~86 MB) on first run
uv run python scripts/evaluate.py --strategy rerank

# Run query understanding evaluation: rule-based QU + RRF (Milestone 9)
# Ignores ground-truth filters; extracts constraints from query text only
uv run python scripts/evaluate.py --strategy understand

# Run query rewriting evaluation: QU + LocalLLMProvider keyword expansion + RRF (Milestone 10)
uv run python scripts/evaluate.py --strategy rewrite
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
| `GET /search/hybrid?q=...` | Hybrid BM25 + vector (weighted-sum or RRF fusion) |
| `POST /search` | Full pipeline: QU → optional rewriting → hybrid RRF → optional reranking |
| `POST /query/understand` | Inspect query understanding extraction result |

Example searches:

```bash
# BM25 — keyword match
curl "localhost:8000/search/lexical?q=family+beach+resort&country=Spain&family_friendly=true&max_price=1000"

# Vector — semantic match (finds Tenerife hotels even without the word in description)
curl "localhost:8000/search/vector?q=sunny+beach+holiday+in+the+Canary+Islands"

# Hybrid — weighted-sum fusion (default)
curl "localhost:8000/search/hybrid?q=romantic+adults-only+resort+with+infinity+pool&country=Spain&adults_only=true"

# Hybrid — RRF fusion (Milestone 7); rank-based, robust to score-scale differences
curl "localhost:8000/search/hybrid?q=hotels+in+Tenerife&fusion=rrf"

# Hybrid — RRF with custom k (smaller k amplifies top-rank advantage)
curl "localhost:8000/search/hybrid?q=luxury+spa+retreat&fusion=rrf&rrf_k=10"

# Hybrid — RRF + cross-encoder reranking (Milestone 8)
# Requires: RERANKING_ENABLED=true in .env (loads ~86 MB model at startup)
curl "localhost:8000/search/hybrid?q=adults+luxury+spa&fusion=rrf&rerank=true"

# Reranking with custom candidate pool size (default: 50)
curl "localhost:8000/search/hybrid?q=hotels+in+Tenerife&fusion=rrf&rerank=true&rerank_k=30"

# Full orchestrated pipeline: QU → hybrid RRF → response with query_understanding (Milestone 9)
curl -X POST "localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "family beach holiday in Greece July from Manchester under £2000"}'

# Inspect what the QU engine extracts from a query
curl -X POST "localhost:8000/query/understand" \
  -H "Content-Type: application/json" \
  -d '{"query": "adults only luxury spa resort in Santorini"}'

# Full pipeline with query rewriting enabled (Milestone 10)
# Requires: QUERY_REWRITING_ENABLED=true in .env
curl -X POST "localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "something quiet with a pool", "rewrite": true}'
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
| `candidate_k` | int | (hybrid only) Candidates per retriever before fusion (default: 50) |
| `fusion` | string | (hybrid only) Fusion method: `weighted` (default) or `rrf` |
| `lexical_weight` | float | (hybrid/weighted only) BM25 score weight (default: 0.5) |
| `vector_weight` | float | (hybrid/weighted only) Vector score weight (default: 0.5) |
| `rrf_k` | int | (hybrid/rrf only) RRF smoothing constant k (default: 60) |
| `rerank` | bool | (hybrid only) Apply cross-encoder reranking to top candidates (default: false) |
| `rerank_k` | int | (hybrid/rerank only) Candidates to pass to the cross-encoder (default: 50) |
| `rewrite` | bool | (POST /search only) Apply query rewriting before retrieval; requires `QUERY_REWRITING_ENABLED=true` (default: false) |

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
| `RERANKING_ENABLED` | `false` | Load cross-encoder at startup (set `true` to enable `rerank=true` on API) |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model (~86 MB download) |
| `RERANK_K` | `50` | Default candidates to pass to the cross-encoder |
| `QUERY_UNDERSTANDING_ENABLED` | `true` | Enable rule-based QU engine at startup (pure Python; no download needed) |
| `QUERY_REWRITING_ENABLED` | `false` | Enable query rewriter at startup (set `true` to allow `rewrite=true` on POST /search) |
| `LLM_PROVIDER` | `local` | LLM backend: `local` (keyword expansion), `echo` (identity stub), `bedrock` (M12) |

---

## Project structure

```
src/travel_ai_search/
├── api/
│   ├── app.py               # FastAPI app, lifespan (OpenSearch + embedding provider)
│   ├── deps.py              # FastAPI dependency injection
│   └── routes/
│       ├── search.py        # GET /search/lexical, GET /search/vector, POST /search
│       └── query.py         # POST /query/understand
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
├── llm/
│   ├── base.py              # LLMProvider Protocol (runtime_checkable)
│   └── local.py             # EchoLLMProvider (identity stub), LocalLLMProvider (keyword expansion)
├── query_understanding/
│   ├── base.py              # QueryUnderstandingEngine Protocol (runtime_checkable)
│   ├── models.py            # QueryUnderstanding dataclass + to_search_filters()
│   ├── extractor.py         # RuleBasedQueryUnderstandingEngine (regex + keyword lookup)
│   └── rewriter.py          # QueryRewriter (wraps LLMProvider; graceful fallback)
├── reranking/
│   ├── base.py              # Reranker Protocol (runtime_checkable, structural typing)
│   └── local.py             # LocalCrossEncoderReranker (sentence-transformers CrossEncoder)
├── retrieval/
│   ├── types.py             # Shared Hit dataclass (imported by all retrieval modules)
│   ├── fusion.py            # FusionMethod enum, fuse_results() (weighted), rrf_fuse() (RRF)
│   ├── lexical.py           # BM25 multi-match search: lexical_search()
│   ├── vector.py            # ANN search: vector_search(), _build_vector_query()
│   └── hybrid.py            # Hybrid orchestration: hybrid_search(), HybridSearchParams
└── infrastructure/
    └── opensearch.py        # OpenSearch client factory

scripts/
├── generate_dataset.py      # Generate synthetic JSONL dataset
├── create_index.py          # Create OpenSearch index (knn enabled)
├── ingest_data.py           # Bulk-index dataset into OpenSearch
├── generate_embeddings.py   # Offline: embed all hotels, update OpenSearch
├── build_golden_dataset.py  # Build golden evaluation dataset (one-time)
├── evaluate.py              # Run evaluation: --strategy bm25|vector|hybrid|rrf|rerank|understand|rewrite
└── healthcheck.py           # Verify OpenSearch connectivity

data/
├── processed/
│   └── hotels.jsonl         # Generated dataset (gitignored)
└── evaluation/
    ├── golden_queries.jsonl # 62 queries, 48,675 graded judgments
    └── results/             # JSON evaluation results per run

tests/
├── unit/                    # No infrastructure required (342 tests)
└── integration/             # Requires OpenSearch running (124 tests)
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

## Development

This project was designed and implemented by Antonio de Castro, with AI-assisted development using Claude Code (model Sonnet4.6).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
