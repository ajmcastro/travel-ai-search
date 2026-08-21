# Travel AI Search

An educational, production-quality project demonstrating modern AI search architectures — from BM25 lexical retrieval through dense vector search, hybrid fusion, neural reranking, and query understanding — applied to a synthetic travel/hotel dataset.

> Built incrementally by milestone. Each milestone adds one concept cleanly, with tests and explanation. See [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) for the full learning objectives.

---

## Current status: Milestone 19 — Two-tower fine-tuning ✅ Complete

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
| 10 | Query rewriting | ✅ Complete |
| 11 | Multi-query retrieval | ✅ Complete |
| 12 | AWS Bedrock providers | ✅ Complete |
| 13 | RAG / travel knowledge base | ✅ Complete |
| 14 | Graph-enhanced retrieval prototype | ✅ Complete |
| 15 | Production API, observability, resilience, final evaluation | ✅ Complete |
| 16 | LLM-as-judge evaluation | ✅ Complete |
| 17 | Learned sparse retrieval (SPLADE) | ✅ Complete |
| 18 | ColBERT late-interaction reranking | ✅ Complete |
| 19 | Two-tower fine-tuning | ✅ **Complete** |

### Evaluation results — all strategies (K=10, 62 queries, 10 query classes)

| Metric | BM25 | Vector | RRF | Rerank | SPLADE | ColBERT | **Fine-tuned** |
|---|---|---|---|---|---|---|---|
| NDCG@10 | 0.5021 | 0.6940 | 0.6239 | 0.6830 | 0.7195 | 0.6294 | **0.7388** |
| MRR | 0.6874 | 0.8688 | 0.8449 | 0.8191 | 0.8370 | 0.7608 | **0.9000** |
| HitRate@10 | 0.8387 | **1.0000** | 0.9516 | 0.9516 | 0.9355 | 0.9516 | 0.9839 |
| Precision@10 | 0.6161 | 0.7790 | 0.7210 | **0.8935** | 0.8290 | 0.7484 | 0.8194 |
| Latency p50 | 26 ms | **10 ms** | 50 ms | 109 ms | 22 ms | 75 ms | 11 ms |
| Latency p95 | 46 ms | 292 ms | 86 ms | 178 ms | **30 ms** | 362 ms | 21 ms |

*Fine-tuned vector (M19) is the highest NDCG@10 single strategy across all milestones.*

**Key findings (cumulative):**
- Vector (M5): NDCG +38.6% vs BM25; `exact_destination` NDCG jumped from 0.18 → 0.84 (+358%); HitRate = 1.000.
- Hybrid (M6) — weighted sum, 50/50: overall NDCG is between BM25 and vector (0.60). Naive 50/50 fusion can *regress* from the best individual retriever when one retriever produces meaningless scores for a query class. `exact_destination` NDCG drops from 0.84 (vector) to 0.48.
- Hybrid (M7) — RRF (k=60): beats weighted-sum (+3.9% NDCG, +5.7% Precision) by using rank positions instead of raw scores. `exact_destination` recovers from 0.48 to 0.53; `activities` beats vector (0.40 vs 0.38). Still below pure vector overall.
- Rerank (M8) — RRF + cross-encoder (`ms-marco-MiniLM-L-6-v2`, 50 candidates): **highest Precision@10 overall (0.89)**. `exact_destination` jumps from 0.53 to 0.79 (+48%); `activities` from 0.40 to 0.57 (+43%). Cost: ~83 ms extra latency.
- Understand (M9) — rule-based QU + RRF: beats RRF on NDCG (+1.2%) and MRR (+2.0%) while being **8.7% faster** (46 ms vs 50 ms p50). `adults_couples` NDCG +8.9% (correct `adults_only` filter extracted). Main failure: false-positive constraints on `budget` queries (−18.4%).
- Rewrite (M10) — QU + `LocalLLMProvider` keyword expansion + RRF: **HitRate improves +3.4%** (more relevant hotels in top-10) but NDCG and MRR regress vs Understand (−2.9%, −4.6%). Classic precision-recall tradeoff: naive synonym expansion broadens recall but dilutes the ranking signal. Architecture ready for real LLM (M12).
- Expand (M11) — QU + `LocalQueryExpander` (N=3 variants) + 6-list RRF: beats rewrite on NDCG (0.629 vs 0.613) because the original query is preserved as the first variant. `activities` +20.9%, `budget` +21.1%. Cost: 3.6× latency (167 ms). Architecture in place for LLM-generated diverse expansion variants (M12).
- Bedrock (M12) — `BedrockEmbeddingProvider` (Titan V2), `BedrockLLMProvider` (Claude via Converse API), `BedrockReranker` (Cohere Rerank v3.5): all three provider slots now support AWS Bedrock as a drop-in replacement for local providers. Graceful degradation: any Bedrock initialisation failure logs a warning and falls back to local. AWS credentials are never required to run the system.
- RAG (M13) — destination knowledge base: 30 documents (one per island/region), stored in a separate `travel_destinations` OpenSearch index and retrieved semantically alongside hotel search. `POST /search` with `rag=true` returns `knowledge_context` (structured destination facts) and optionally `rag_summary` (LLM-synthesized recommendation). Hotel ranking is unchanged — RAG is purely additive.
- Graph (M14) — in-memory destination graph: 38 nodes (30 destinations + 8 UK airports) and 309 edges built from the knowledge JSONL at startup. Two edge types: `SIMILAR_TO` (curated editorial similarity, bidirectional) and `FLIES_TO` (directed, airport → destination, with realistic long-haul hub restriction). Three exploration endpoints: `GET /graph/similar` (BFS SIMILAR_TO), `GET /graph/destinations` (FLIES_TO from airport), `GET /graph/airports` (reverse FLIES_TO). Demonstrates what graph traversal provides that vector search cannot: exact structural reachability and multi-hop curated similarity chains.
- Observability (M15) — per-stage pipeline timing (`qu_took_ms`, `rewrite_took_ms`, `lexical_took_ms`, `vector_took_ms`, `reranking_took_ms`, `rag_took_ms`), structured request logging, in-memory Prometheus-compatible metrics (`GET /metrics`), deep health check (`GET /health`), and runtime resilience (rewriter fail → original query; embedding fail → BM25 fallback). 9 new resilience unit tests verify all fallback paths.
- LLM-as-judge (M16) — `JudgeProvider` Protocol + `EchoJudgeProvider` (no AWS needed) + `BedrockJudgeProvider` (`amazon.nova-lite-v1:0` — different family from the Anthropic Claude generator, to avoid common-mode bias). `LLMEvaluator` scores each retrieved hotel 0–3 with a rationale. Spearman ρ and Kendall τ (from first principles) measure the generator-effect gap between synthetic and human-written query slices. 20 human-written queries in `data/evaluation/human_queries.jsonl`. `POST /evaluate/judge` endpoint + `make evaluate-judge` CLI.
- SPLADE (M17) — `LearnedSparseProvider` Protocol + `LocalSparseProvider` (HuggingFace SPLADE model). **NDCG@10 = 0.7195 — the highest single-strategy score measured so far**, beating pure vector (0.6940) and cross-encoder reranking (0.6830). Encodes queries and documents as sparse vocabulary-weight vectors (SPLADE formula: `w_t = max_i log(1 + ReLU(MLM_logit_{i,t}))`). Stored as `rank_features` in OpenSearch; queried via `bool.should[rank_feature]` — one clause per non-zero vocabulary term. `exact_destination` class is perfect (NDCG=1.000); `luxury` and `nightlife` classes are weakest (NDCG≈0.55–0.60), likely due to vocabulary mismatch between query and document encodings. p95 latency = 30 ms — faster than RRF (86 ms) and reranking (178 ms).
- ColBERT (M18) — `ColBERTReranker` implementing the `Reranker` Protocol via MaxSim late interaction: `Score(q,d) = Σᵢ max_j (qᵢ·dⱼ)`. Document token embeddings pre-computed offline by `generate_colbert_embeddings.py` and stored as `data/processed/colbert_embeddings/<hotel_id>.npy` (shape `(≤128, 128)`, L2-normalised float32). At query time: RRF retrieves candidates → query is encoded into token embeddings → MaxSim scores each candidate against its `.npy` file → sorted, top-K returned. Drop-in via `RERANKER_PROVIDER=colbert`. NDCG@10 = 0.6294, slightly above RRF (0.6239) but below cross-encoder (0.6830) and SPLADE (0.7195). Key finding: the hypothesis that MaxSim outperforms cross-encoder did not hold — file I/O loading 50 `.npy` files per query drove p95 to 362 ms (worse than cross-encoder 178 ms), and running `colbert-ir/colbertv2.0` without its custom `[Q]`/`[D]` prefix tokens degraded quality vs. using the full ColBERT library. `multi_constraint` class best (MRR=1.000); `activities` class worst (NDCG=0.368). ColBERT does recover the SPLADE luxury/nightlife gap (NDCG +0.10), as MaxSim is not blocked by vocabulary sparsity.
- Fine-tuned vector (M19) — domain-adapted `all-MiniLM-L6-v2` via contrastive learning. `MultipleNegativesRankingLoss` (InfoNCE variant) with in-batch negatives + BM25-mined explicit hard negatives. Training pipeline: `prepare_fine_tuning_data.py` → `fine_tune_embeddings.py` (3 epochs, batch=16, manual PyTorch loop to avoid the `datasets` dependency) → `data/models/bi-encoder-travel/`. **NDCG@10 = 0.7388 — highest single-strategy result across all milestones** (+6.5% vs base vector, +2.7% vs SPLADE). Biggest class gains: nightlife (+31.6%), family (+15.1%), activities (+21.6%). Regression on luxury (−11.2%) — attributed to the small training corpus (~300 triplets). Evaluated via `--strategy fine-tuned-vector`; loads the checkpoint via `LocalEmbeddingProvider` (local path = same code path as HuggingFace Hub).

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

# Generate destination knowledge documents (Milestone 13)
make generate-knowledge

# Create knowledge index and ingest 30 destination documents (Milestone 13)
# Requires: make up + make generate-knowledge first
make ingest-knowledge

# SPLADE sparse embeddings (Milestone 17) — run after ingest
# Option A: add rank_features field to existing index (non-destructive)
make update-sparse-mapping
# Option B: recreate index from scratch (includes rank_features in mapping)
# uv run python scripts/create_index.py --recreate && make ingest && make generate-embeddings
#
# Then encode all hotel descriptions with the SPLADE model (~300 MB download, ~5-10 min on CPU)
make generate-sparse-embeddings

# ColBERT token embeddings (Milestone 18) — run after generate-embeddings
# Default model: colbert-ir/colbertv2.0 (~400 MB download on first run)
# Quick alternative (no new download): set COLBERT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 in .env
make generate-colbert-embeddings
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

# LLM-as-judge evaluation (Milestone 16)
# Dry run with EchoJudge — no AWS credentials required; verifies the pipeline end-to-end
make evaluate-judge

# All strategies on both slices + generator-effect gap table (Spearman ρ, Kendall τ)
make evaluate-judge-all

# With real Bedrock judge (requires AWS credentials and JUDGE_PROVIDER=bedrock in .env)
uv run python scripts/evaluate_judge.py --all-strategies --slice both --judge-provider bedrock

# SPLADE evaluation (Milestone 17)
# Requires: make generate-sparse-embeddings first (downloads ~300 MB model on first run)
make evaluate-splade

# ColBERT late-interaction reranking evaluation (Milestone 18)
# Requires: make generate-embeddings + make generate-colbert-embeddings first
make evaluate-colbert
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
| `GET /health` | Deep health check: OpenSearch ping, index existence, model load states |
| `GET /metrics` | Prometheus-compatible counters and histograms (request counts, latency) |
| `GET /search/lexical?q=...` | BM25 lexical search |
| `GET /search/vector?q=...` | Dense vector (ANN) search |
| `GET /search/hybrid?q=...` | Hybrid BM25 + vector (weighted-sum or RRF fusion) |
| `GET /search/sparse?q=...` | SPLADE learned sparse search (vocabulary expansion via MLM head) |
| `POST /search` | Full pipeline: QU → optional rewriting → hybrid RRF → optional reranking |
| `POST /query/understand` | Inspect query understanding extraction result |
| `POST /evaluate/judge` | LLM-as-judge: score top-K hotels per query (0–3) with rationale |

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

# RAG: knowledge retrieval + optional LLM synthesis (Milestone 13)
# Requires: RAG_ENABLED=true in .env + make ingest-knowledge
# Returns knowledge_context (destination facts) + rag_summary (if LLM configured)
curl -X POST "localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "relaxed beach holiday in Greece", "rag": true}'

# Graph exploration: curated SIMILAR_TO traversal (Milestone 14)
# No extra setup needed — graph is built at startup (GRAPH_ENABLED=true by default)
curl "localhost:8000/graph/similar?destination=Mallorca&hops=1"
curl "localhost:8000/graph/similar?destination=Mallorca&hops=2"

# Graph exploration: airport → reachable destinations (FLIES_TO edges)
# Compare GLA (regional, short-haul only) vs LHR (hub, includes long-haul)
curl "localhost:8000/graph/destinations?airport=GLA"
curl "localhost:8000/graph/destinations?airport=LHR"

# Graph exploration: which airports serve a long-haul destination?
# Barbados → only LGW, LHR, MAN; Tenerife → all 8 UK airports
curl "localhost:8000/graph/airports?destination=Barbados"
curl "localhost:8000/graph/airports?destination=Tenerife"

# Deep health check (Milestone 15) — shows component status, not just "ok"
curl "localhost:8000/health"

# Prometheus metrics scrape — counters, latency histogram in text exposition format
curl "localhost:8000/metrics"

# LLM-as-judge evaluation (Milestone 16)
# EchoJudgeProvider (default, no AWS): fixed score=2, verifies pipeline wiring
curl -X POST "localhost:8000/evaluate/judge" \
  -H "Content-Type: application/json" \
  -d '{"queries":[{"query_text":"beach holiday with kids","query_class":"family"}],"strategy":"rrf","k":5}'

# Full search with complete per-stage timing in response (Milestone 15)
# Response now includes: qu_took_ms, rewrite_took_ms, lexical_took_ms,
# vector_took_ms, reranking_took_ms, rag_took_ms, strategy, fallback_used
curl -X POST "localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "family beach holiday Greece July from Manchester"}'

# SPLADE sparse search (Milestone 17) — vocabulary expansion via MLM head
# Requires: SPLADE_ENABLED=true in .env + make generate-sparse-embeddings
# "quiet adults retreat" may activate "peaceful", "tranquil", "serene" in the index
curl "localhost:8000/search/sparse?q=quiet+adults+retreat+near+the+sea"
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
make final-eval         # run final evaluation across all strategies + save results (Milestone 15)
make evaluate-judge     # LLM-as-judge dry run (EchoJudge, rrf, generated slice)
make evaluate-judge-all # LLM-as-judge: all strategies on both slices + generator-effect gap
make generate-embeddings # generate dense embeddings for all hotels (Milestone 5+)
make update-sparse-mapping    # add rank_features field to existing index (non-destructive, M17)
make generate-sparse-embeddings # encode all hotels with SPLADE model (~300 MB, M17)
make evaluate-splade    # SPLADE evaluation against golden dataset (M17)
make generate-colbert-embeddings # generate ColBERT token embeddings for all hotels (M18)
make evaluate-colbert   # ColBERT late-interaction reranking evaluation (M18)
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
| `LOG_FORMAT` | `text` | Log format: `text` (human-readable) or `json` (NDJSON for production log aggregators) |
| `ENVIRONMENT` | `development` | `development` or `production` |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model for dense embeddings |
| `EMBEDDING_DIMENSION` | `384` | Embedding dimension (must match model and index mapping) |
| `RERANKING_ENABLED` | `false` | Load cross-encoder at startup (set `true` to enable `rerank=true` on API) |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model (~86 MB download) |
| `RERANK_K` | `50` | Default candidates to pass to the cross-encoder |
| `QUERY_UNDERSTANDING_ENABLED` | `true` | Enable rule-based QU engine at startup (pure Python; no download needed) |
| `QUERY_REWRITING_ENABLED` | `false` | Enable query rewriter at startup (set `true` to allow `rewrite=true` on POST /search) |
| `LLM_PROVIDER` | `local` | LLM backend: `local` (keyword expansion), `echo` (identity stub), `bedrock` (Claude via Converse API) |
| `QUERY_EXPANSION_ENABLED` | `false` | Enable multi-query expander at startup |
| `NUM_EXPANSION_QUERIES` | `3` | Number of query variants to generate per request |
| `EMBEDDING_PROVIDER` | `local` | Embedding backend: `local` (sentence-transformers) or `bedrock` (Titan V2) |
| `RERANKER_PROVIDER` | `local` | Reranker backend: `local` (cross-encoder), `bedrock` (Cohere Rerank v3.5), or `colbert` (M18) |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock API calls |
| `BEDROCK_EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Titan Embeddings model ID |
| `BEDROCK_EMBEDDING_DIMENSION` | `1024` | Titan output dimension (256/512/1024) — must also update `EMBEDDING_DIMENSION` and recreate the index |
| `BEDROCK_LLM_MODEL_ID` | `anthropic.claude-haiku-4-5-20251001` | Bedrock model for query rewriting and RAG synthesis |
| `BEDROCK_RERANKER_MODEL_ID` | `cohere.rerank-v3-5:0` | Cohere Rerank model ID |
| `RAG_ENABLED` | `false` | Enable knowledge retrieval + synthesis at startup (set `true` after `make ingest-knowledge`) |
| `KNOWLEDGE_INDEX_NAME` | `travel_destinations` | OpenSearch index for destination knowledge documents |
| `RAG_CONTEXT_K` | `3` | Number of destination knowledge docs to retrieve per RAG query |
| `GRAPH_ENABLED` | `true` | Build in-memory destination graph at startup (set `false` to disable `/graph/*` endpoints) |
| `KNOWLEDGE_FILE_PATH` | `data/knowledge/destinations.jsonl` | JSONL file used to seed the destination graph |
| `JUDGE_PROVIDER` | `echo` | LLM judge backend: `echo` (fixed score=2, no AWS) or `bedrock` (M16) |
| `BEDROCK_JUDGE_MODEL_ID` | `amazon.nova-lite-v1:0` | Judge model — must differ from the generator model family to avoid common-mode bias (M16) |
| `SPLADE_ENABLED` | `false` | Load SPLADE sparse encoder at startup; when false, `GET /search/sparse` returns 503 (M17) |
| `SPLADE_MODEL_NAME` | `naver/splade-cocondenser-ensemble-distil` | HuggingFace masked-language model for sparse encoding (~300 MB, cached in `~/.cache/huggingface/`) |
| `SPLADE_TOP_K_TERMS` | `64` | Max non-zero vocabulary terms per query vector sent to OpenSearch (higher = better recall, slower) |
| `COLBERT_MODEL_NAME` | `colbert-ir/colbertv2.0` | HuggingFace ColBERT checkpoint (~400 MB); alt: `sentence-transformers/all-MiniLM-L6-v2` (no new download, 384-dim) (M18) |
| `COLBERT_EMBEDDINGS_DIR` | `data/processed/colbert_embeddings` | Directory of `<hotel_id>.npy` token embedding files produced by `make generate-colbert-embeddings` (M18) |
| `COLBERT_DOC_MAXLEN` | `128` | Max document tokens; must match what was used during `generate-colbert-embeddings` (M18) |
| `COLBERT_QUERY_MAXLEN` | `32` | Max query tokens; ColBERT v2 default (M18) |

---

## Project structure

```
src/travel_ai_search/
├── api/
│   ├── app.py               # FastAPI app, lifespan (OpenSearch + embedding provider)
│   ├── deps.py              # FastAPI dependency injection
│   ├── schemas/
│   │   └── evaluate.py      # JudgeRequest, JudgeResponse, JudgedHit, JudgeQueryOutput (M16)
│   └── routes/
│       ├── search.py        # GET /search/lexical, GET /search/vector, POST /search
│       ├── query.py         # POST /query/understand
│       ├── health.py        # GET /health — deep health check (M15)
│       ├── graph.py         # GET /graph/similar, GET /graph/destinations, GET /graph/airports (M14)
│       └── evaluate.py      # POST /evaluate/judge — LLM-as-judge scoring endpoint (M16)
├── config/
│   └── settings.py          # Pydantic settings, loaded from env vars / .env
├── domain/
│   └── models.py            # TravelProduct model + build_embedding_text()
├── embeddings/
│   ├── base.py              # EmbeddingProvider Protocol
│   ├── local.py             # LocalEmbeddingProvider (sentence-transformers)
│   ├── bedrock.py           # BedrockEmbeddingProvider (Titan V2, M12)
│   └── sparse.py            # LearnedSparseProvider Protocol + LocalSparseProvider (SPLADE, M17)
├── evaluation/
│   ├── dataset.py           # GoldenQuery, GoldenDataset, load_dataset()
│   ├── evaluator.py         # evaluate(), EvaluationReport, SearchFn type
│   ├── metrics.py           # P@K, Recall@K, HitRate@K, RR, AP, NDCG@K
│   ├── judge.py             # JudgeProvider Protocol, JudgeVerdict, EchoJudgeProvider, prompt/parser (M16)
│   ├── judge_bedrock.py     # BedrockJudgeProvider (amazon.nova-lite-v1:0, M16)
│   └── judge_evaluator.py   # LLMEvaluator, JudgeReport, spearman_rho, kendall_tau, generator_effect_gap (M16)
├── ingestion/
│   ├── index.py             # OpenSearch index mapping and CRUD (knn_vector)
│   └── ingestor.py          # Bulk ingestion: load_products(), ingest()
├── llm/
│   ├── base.py              # LLMProvider Protocol (runtime_checkable)
│   ├── local.py             # EchoLLMProvider (identity stub), LocalLLMProvider (keyword expansion)
│   └── bedrock.py           # BedrockLLMProvider (Claude via Converse API, M12)
├── query_understanding/
│   ├── base.py              # QueryUnderstandingEngine Protocol (runtime_checkable)
│   ├── models.py            # QueryUnderstanding dataclass + to_search_filters()
│   ├── extractor.py         # RuleBasedQueryUnderstandingEngine (regex + keyword lookup)
│   ├── rewriter.py          # QueryRewriter (wraps LLMProvider; graceful fallback)
│   └── expander.py          # LocalQueryExpander — generates N query variants for multi-query retrieval (M11)
├── reranking/
│   ├── base.py              # Reranker Protocol (runtime_checkable, structural typing)
│   ├── local.py             # LocalCrossEncoderReranker (sentence-transformers CrossEncoder)
│   ├── bedrock.py           # BedrockReranker (Cohere Rerank v3.5, M12)
│   └── colbert.py           # ColBERTReranker + ColBERTEncoder + maxsim() (M18)
├── retrieval/
│   ├── types.py             # Shared Hit dataclass (imported by all retrieval modules)
│   ├── fusion.py            # FusionMethod enum, fuse_results() (weighted), rrf_fuse() (RRF)
│   ├── lexical.py           # BM25 multi-match search: lexical_search()
│   ├── vector.py            # ANN search: vector_search(), _build_vector_query()
│   ├── hybrid.py            # Hybrid orchestration: hybrid_search(), HybridSearchParams
│   └── splade.py            # SPLADE sparse retrieval: splade_search(), _build_splade_query() (M17)
├── rag/                     # RAG / destination knowledge (M13)
│   ├── __init__.py
│   ├── knowledge.py         # DestinationKnowledge model, build_knowledge_embedding_text()
│   ├── index.py             # Knowledge index mapping, create_knowledge_index()
│   ├── retriever.py         # KnowledgeRetriever (knn + optional country filter)
│   └── synthesizer.py       # RAGSynthesizer (prompt construction + LLM call)
├── graph/                   # Graph-enhanced retrieval (M14)
│   ├── __init__.py
│   ├── models.py            # NodeType, EdgeType, GraphNode, DestinationGraph (adjacency-list)
│   └── builder.py           # build_destination_graph(), load_knowledge_docs()
├── observability/           # Observability utilities (M15)
│   ├── metrics.py           # Counter, Histogram, MetricsRegistry — Prometheus text format
│   └── logging.py           # StructuredFormatter — NDJSON log format (LOG_FORMAT=json)
└── infrastructure/
    ├── opensearch.py        # OpenSearch client factory
    └── bedrock.py           # boto3 bedrock-runtime client factory (M12)

scripts/
├── generate_dataset.py      # Generate synthetic JSONL dataset
├── create_index.py          # Create OpenSearch index (knn enabled)
├── ingest_data.py           # Bulk-index dataset into OpenSearch
├── generate_embeddings.py   # Offline: embed all hotels, update OpenSearch
├── generate_knowledge.py    # Generate destination knowledge documents (M13)
├── ingest_knowledge.py      # Create knowledge index and ingest 30 docs (M13)
├── build_golden_dataset.py  # Build golden evaluation dataset (one-time)
├── evaluate.py              # Run evaluation: --strategy bm25|vector|hybrid|rrf|rerank|understand|rewrite|splade|colbert
├── evaluate_judge.py        # LLM-as-judge CLI: --strategy, --slice, --judge-provider, generator-effect gap (M16)
├── update_sparse_mapping.py # Add rank_features field to existing index (non-destructive, M17)
├── generate_sparse_embeddings.py # Offline SPLADE encoding: all hotels → OpenSearch bulk update (M17)
├── generate_colbert_embeddings.py # Offline ColBERT encoding: all hotels → data/processed/colbert_embeddings/ (M18)
├── export_chat.py           # Export Claude Code chat history to docs/CHAT_HISTORY.md
└── healthcheck.py           # Verify OpenSearch connectivity

data/
├── processed/
│   ├── hotels.jsonl         # Generated dataset (gitignored)
│   └── colbert_embeddings/  # Per-hotel ColBERT token embeddings: <hotel_id>.npy (M18, gitignored)
├── knowledge/
│   └── destinations.jsonl   # 30 destination knowledge documents (M13)
└── evaluation/
    ├── golden_queries.jsonl  # 62 queries, 48,675 graded judgments
    ├── human_queries.jsonl   # 20 human-written queries for generator-effect measurement (M16)
    └── results/              # JSON evaluation results per run

docs/
├── PROJECT_SPEC.md          # Full learning objectives and milestone definitions
├── EXPERIMENTS.md           # Hypothesis, results, and observations per milestone
├── RATIONALE_PER_MILESTONE.md # IR concepts, design choices, and why behind each milestone
├── CHAT_HISTORY.md          # Exported Claude Code session history (auto-generated)
└── annotation_guide.md      # Guide for creating and grading human query annotations (M16)

tests/
├── unit/                    # No infrastructure required (653 tests)
└── integration/             # Requires OpenSearch running (132 tests)
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
| `splade_vector` | `rank_features` | SPLADE sparse encoding: `{vocabulary_term: weight}` (Milestone 17) |

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
