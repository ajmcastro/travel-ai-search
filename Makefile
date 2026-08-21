.DEFAULT_GOAL := help

# ── Setup ──────────────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install / sync all dependencies (production + dev)
	uv sync

.PHONY: env
env: ## Copy .env.example → .env (skips if .env already exists)
	@test -f .env && echo ".env already exists, skipping." || (cp .env.example .env && echo "Created .env from .env.example")

# ── Infrastructure ─────────────────────────────────────────────────────────────

.PHONY: up
up: ## Start OpenSearch in the background
	docker compose up -d

.PHONY: down
down: ## Stop OpenSearch, keep data volume
	docker compose down

.PHONY: down-v
down-v: ## Stop OpenSearch and delete the data volume
	docker compose down -v

.PHONY: logs
logs: ## Tail OpenSearch container logs
	docker compose logs -f opensearch

.PHONY: health
health: ## Verify connectivity to OpenSearch
	uv run python scripts/healthcheck.py

# ── Testing ────────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run unit tests (no infrastructure required)
	uv run pytest -v

.PHONY: test-integration
test-integration: ## Run integration tests (requires: make up)
	uv run pytest tests/integration -v

.PHONY: test-all
test-all: ## Run unit + integration tests
	uv run pytest tests/ -v

.PHONY: test-k
test-k: ## Run a single test by name  →  make test-k k=test_settings
	uv run pytest -k "$(k)" -v

# ── Code quality ───────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Lint with ruff
	uv run ruff check .

.PHONY: lint-fix
lint-fix: ## Lint with ruff and auto-fix issues
	uv run ruff check --fix .

.PHONY: fmt
fmt: ## Format code with ruff
	uv run ruff format .

.PHONY: fmt-check
fmt-check: ## Check formatting without modifying files
	uv run ruff format --check .

.PHONY: typecheck
typecheck: ## Type-check with mypy
	uv run mypy src

.PHONY: check
check: lint fmt-check typecheck test ## Full quality gate: lint + format + types + unit tests

# ── Application ────────────────────────────────────────────────────────────────

.PHONY: serve
serve: ## Start the FastAPI dev server with auto-reload  [Milestone 3+]
	uv run uvicorn travel_ai_search.api.app:app --reload

# ── Data pipeline ──────────────────────────────────────────────────────────────

.PHONY: generate-data
generate-data: ## Generate the synthetic travel dataset  [Milestone 1]
	uv run python scripts/generate_dataset.py

.PHONY: create-index
create-index: ## Create OpenSearch index with mappings  [Milestone 2]
	uv run python scripts/create_index.py

.PHONY: ingest
ingest: ## Ingest travel dataset into OpenSearch  [Milestone 2]
	uv run python scripts/ingest_data.py

.PHONY: generate-embeddings
generate-embeddings: ## Generate dense embeddings and update OpenSearch  [Milestone 5]
	uv run python scripts/generate_embeddings.py

.PHONY: embed
embed: generate-embeddings ## Alias for generate-embeddings

.PHONY: generate-knowledge
generate-knowledge: ## Generate destination knowledge documents  [Milestone 13]
	uv run python scripts/generate_knowledge.py

.PHONY: ingest-knowledge
ingest-knowledge: ## Create knowledge index and ingest destination docs  [Milestone 13]
	uv run python scripts/ingest_knowledge.py

# ── Graph exploration (Milestone 14) ───────────────────────────────────────────
# The destination graph is built automatically at server startup from the
# knowledge JSONL file — no separate ingestion step is required.
# Use these curl examples to explore the graph endpoints after `make serve`:
#
#   Graph similarity (SIMILAR_TO traversal):
#     curl "http://localhost:8000/graph/similar?destination=Mallorca&hops=1"
#     curl "http://localhost:8000/graph/similar?destination=Mallorca&hops=2"
#
#   Reachable destinations from a departure airport (FLIES_TO traversal):
#     curl "http://localhost:8000/graph/destinations?airport=GLA"
#     curl "http://localhost:8000/graph/destinations?airport=LHR"
#
#   Airports serving a specific destination (reverse FLIES_TO lookup):
#     curl "http://localhost:8000/graph/airports?destination=Barbados"
#     curl "http://localhost:8000/graph/airports?destination=Tenerife"

# ── Evaluation ─────────────────────────────────────────────────────────────────

.PHONY: evaluate
evaluate: ## Run search evaluation across all strategies  [Milestone 4+]
	uv run python scripts/evaluate.py

.PHONY: final-eval
final-eval: ## Run final evaluation across all strategies and save results  [Milestone 15]
	uv run python scripts/evaluate.py --output data/evaluation/final_results.json

.PHONY: evaluate-judge
evaluate-judge: ## LLM-as-judge evaluation (EchoJudge, rrf, generated slice)  [Milestone 16]
	uv run python scripts/evaluate_judge.py --strategy rrf --slice generated

.PHONY: evaluate-judge-all
evaluate-judge-all: ## LLM-as-judge: all strategies on both slices + generator-effect gap  [Milestone 16]
	uv run python scripts/evaluate_judge.py --all-strategies --slice both

# ── SPLADE / learned sparse retrieval (Milestone 17) ──────────────────────────
# 1. Add the rank_features field to an existing index (skip if you recreate it):
#      make update-sparse-mapping
# 2. Encode all hotel descriptions with the SPLADE model:
#      make generate-sparse-embeddings
# 3. Evaluate SPLADE against the golden dataset:
#      make evaluate-splade

.PHONY: update-sparse-mapping
update-sparse-mapping: ## Add splade_vector rank_features field to existing index  [Milestone 17]
	uv run python scripts/update_sparse_mapping.py

.PHONY: generate-sparse-embeddings
generate-sparse-embeddings: ## Encode hotels with SPLADE and update OpenSearch  [Milestone 17]
	uv run python scripts/generate_sparse_embeddings.py

.PHONY: evaluate-splade
evaluate-splade: ## Evaluate SPLADE retrieval against the golden dataset  [Milestone 17]
	uv run python scripts/evaluate.py --strategy splade

# ── ColBERT late-interaction reranking (Milestone 18) ─────────────────────────
# 1. Generate dense embeddings first (for RRF candidate stage):
#      make generate-embeddings
# 2. Generate per-document ColBERT token embeddings (one .npy file per hotel):
#      make generate-colbert-embeddings
# 3. Evaluate ColBERT reranking against the golden dataset:
#      make evaluate-colbert
#
# Quick start without downloading colbert-ir/colbertv2.0 (~400 MB):
#   Set COLBERT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 in .env
#   to reuse the already-cached MiniLM weights (384-dim, same MaxSim mechanism).

.PHONY: generate-colbert-embeddings
generate-colbert-embeddings: ## Generate ColBERT token embeddings for all hotels  [Milestone 18]
	uv run python scripts/generate_colbert_embeddings.py

.PHONY: evaluate-colbert
evaluate-colbert: ## Evaluate ColBERT late-interaction reranking against the golden dataset  [Milestone 18]
	uv run python scripts/evaluate.py --strategy colbert

# ── Help ───────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@printf "\nUsage: make \033[36m<target>\033[0m\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@printf "\n  Targets marked \033[90m[Milestone N+]\033[0m are stubs — the underlying script does not\n"
	@printf "  exist yet and will be added in the indicated milestone.\n\n"
