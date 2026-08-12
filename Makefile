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

# ── Evaluation ─────────────────────────────────────────────────────────────────

.PHONY: evaluate
evaluate: ## Run search evaluation across all strategies  [Milestone 4+]
	uv run python scripts/evaluate.py

# ── Help ───────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@printf "\nUsage: make \033[36m<target>\033[0m\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@printf "\n  Targets marked \033[90m[Milestone N+]\033[0m are stubs — the underlying script does not\n"
	@printf "  exist yet and will be added in the indicated milestone.\n\n"
