# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**Travel AI Search** is an educational, production-quality Python project demonstrating modern AI search architectures (BM25 → vector → hybrid → reranking → query understanding) applied to a synthetic travel/hotel dataset. The full project spec is in [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md).

The project is being built **incrementally by milestone**. Always check which milestone is currently complete before adding new capabilities. Do not jump ahead.

---

## Package management — uv only

Use `uv` for everything Python. Never use pip, Poetry, pipenv, or conda.

```bash
uv sync                  # install/sync dependencies
uv add <package>         # add a dependency
uv remove <package>      # remove a dependency
uv run <command>         # run a command in the venv
```

Python version: **3.12** (unless a specific dependency forces otherwise).
Dependencies live in `pyproject.toml`; always commit `uv.lock`.

---

## Common commands

```bash
# Infrastructure
docker compose up -d         # start OpenSearch

# Development
uv run uvicorn travel_ai_search.api.app:app --reload

# Tests
uv run pytest                        # all tests
uv run pytest tests/unit/            # unit tests only
uv run pytest tests/integration/     # integration tests (requires OpenSearch)
uv run pytest -k "test_name"         # single test

# Quality
uv run ruff check .
uv run ruff format .
uv run mypy src
```

---

## Architecture

### Stack
- **OpenSearch** (via Docker Compose) — primary search backend
- **opensearch-py** — Python client
- **FastAPI + Pydantic** — HTTP API layer
- **sentence-transformers** — local dense embeddings
- **cross-encoder** model — local neural reranking
- **boto3** — optional AWS Bedrock integration (embeddings, LLM, reranking)

### Source layout
```
src/travel_ai_search/
├── api/            # FastAPI app, routes, request/response schemas
├── config/         # settings via environment variables / .env
├── domain/         # core domain models and interfaces (no external deps)
├── ingestion/      # data loading, index creation
├── embeddings/     # EmbeddingProvider: LocalEmbeddingProvider, BedrockEmbeddingProvider
├── retrieval/      # lexical (BM25), vector (HNSW), hybrid, fusion (RRF, weighted)
├── reranking/      # LocalCrossEncoderReranker, BedrockReranker
├── query_understanding/  # intent, entity extraction, rewrite, expansion
├── evaluation/     # metrics (P@K, Recall@K, MRR, NDCG@K), evaluator, golden dataset
├── orchestration/  # SearchService — coordinates the full pipeline
└── infrastructure/ # OpenSearch client, Bedrock client
```

### Key design rules
- **Provider abstractions** for all AI capabilities: `EmbeddingProvider`, `LLMProvider`, `Reranker`. AWS Bedrock is always optional; the full system must run locally without AWS credentials.
- **Domain layer** (`domain/`) has no infrastructure dependencies — no OpenSearch, no HTTP calls.
- **Configuration** is loaded from environment variables via a settings class. Use `.env` locally (`.env.example` committed, real `.env` gitignored).
- Feature flags (`query_rewriting_enabled`, `query_expansion_enabled`, `reranking_enabled`, etc.) must be in config so search strategies can be swapped without code changes.
- **Structured logging** throughout; design metrics for easy Prometheus integration later.
- **Graceful degradation**: if LLM rewriting fails → use original query; if embedding fails → fall back to BM25; if reranker fails → return fused results.

### Search pipeline (when fully built)
```
User query
  → query_understanding (intent, entities, hard constraints, soft preferences)
  → [optional] query rewriting / expansion
  → retrieval stage A: lexical BM25
  → retrieval stage B: vector ANN (HNSW)
  → fusion (RRF or weighted score)
  → [optional] cross-encoder reranking (top N candidates → top K)
  → response with per-stage timing metadata
```

### API endpoints (target)
```
GET  /health
GET  /search/lexical?q=...
GET  /search/vector?q=...
GET  /search/hybrid?q=...
POST /search                  # full orchestrated pipeline
POST /query/understand
POST /evaluate
```

---

## Data and evaluation

- Synthetic dataset: ~5,000–10,000 travel products generated with a deterministic seed; stored under `data/`.
- Evaluation golden set: ≥50 queries with graded relevance (0–3) across query classes (exact destination, semantic discovery, family, luxury, multi-constraint, etc.).
- Metrics implemented from first principles: Precision@K, Recall@K, HitRate@K, MRR, DCG, NDCG@K.
- Evaluation CLI: `uv run python scripts/evaluate.py` (or a proper entry point).
- Document experiments in `docs/EXPERIMENTS.md` — hypothesis, config, metrics, and surprises.

---

## Milestone tracking

Milestones (from the spec):
0. Project scaffold (uv + Docker Compose + OpenSearch connectivity + Ruff + pytest) ← **start here**
1. Synthetic travel dataset
2. OpenSearch mappings and ingestion
3. BM25 lexical retrieval
4. Evaluation framework and BM25 baseline
5. Embeddings and vector retrieval
6. Hybrid retrieval
7. RRF and alternative fusion
8. Cross-encoder reranking
9. Query understanding and structured constraints
10. Query rewriting
11. Multi-query retrieval
12. AWS Bedrock providers
13. RAG / travel knowledge base
14. Graph-enhanced retrieval prototype
15. Production API, observability, resilience, final evaluation

Only introduce modules and files as they are actually needed for the current milestone.
