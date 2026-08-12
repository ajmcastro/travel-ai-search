# Travel AI Search — Project Implementation Prompt

I want you to act as a Senior Machine Learning Engineer and Search/Information Retrieval architect and help me build an educational but production-quality project called:

**Travel AI Search**

The purpose of this project is to learn and demonstrate modern AI Search architectures similar to those used in large travel/e-commerce platforms.

This is NOT a TUI project and must not use TUI proprietary data, APIs, branding, code, or intellectual property. Use synthetic or openly available travel/hotel data only.

I want to understand the implementation, not merely have you generate code. Therefore work incrementally, explain important architectural choices, and avoid unnecessary complexity.

## Main learning objectives

The project must progressively demonstrate:

1. Traditional lexical information retrieval
2. BM25
3. OpenSearch
4. Dense embeddings
5. Vector search
6. Approximate Nearest Neighbour search / HNSW
7. Hybrid lexical + semantic retrieval
8. Reciprocal Rank Fusion
9. Alternative score-fusion strategies
10. Neural reranking
11. Search evaluation
12. Intent recognition
13. Entity and constraint extraction
14. Query rewriting
15. Query expansion
16. Multi-query retrieval
17. AWS Bedrock integration
18. RAG concepts
19. Graph-enhanced retrieval concepts
20. Production search architecture and observability

The final application should allow natural-language travel queries such as:

"Find me a family holiday somewhere warm in October, departing from Manchester, under £2,000, preferably all-inclusive and close to the beach."

or:

"Somewhere like Mallorca but quieter and good for small children."

The system should distinguish semantic search requirements from structured constraints and soft preferences.

---

# Technology requirements

Use Python.

IMPORTANT: Use `uv` for ALL Python version, virtual environment and package management.

Do NOT use Poetry, pipenv, Conda, requirements.txt as the primary dependency mechanism, or direct pip-based project management.

Use:

* `uv python`
* `uv init`
* `uv add`
* `uv remove`
* `uv sync`
* `uv run`

as appropriate.

Maintain dependencies in:

`pyproject.toml`

and commit:

`uv.lock`

Choose a modern Python version that is well supported by the required libraries. Prefer Python 3.12 unless dependency compatibility gives a good reason to use another version.

Use:

* OpenSearch
* opensearch-py
* Docker Compose
* FastAPI
* Pydantic
* sentence-transformers or an equivalent suitable embedding library
* a suitable cross-encoder for local reranking
* boto3 for AWS Bedrock integration
* pytest
* Ruff
* mypy where useful

Avoid adding large frameworks such as LangChain unless they solve a concrete requirement. Prefer transparent Python implementations so the retrieval architecture is easy to understand.

---

# Architecture principles

Follow these principles:

* clean architecture without overengineering
* strong separation between domain logic and infrastructure
* dependency injection where useful
* provider abstractions for external AI capabilities
* async APIs when beneficial
* typed Python
* Pydantic models for API/domain boundaries
* structured logging
* configuration through environment variables/settings
* `.env.example`, never real credentials
* unit tests
* integration tests where appropriate
* reproducible Docker environment
* Makefile required; every milestone must add its relevant targets to it
* clear README
* architecture documentation
* ADRs for important architectural decisions when useful

Do not create abstractions merely for the sake of design patterns.

---

# Target repository structure

Start with approximately this structure, but improve it if there is a good architectural reason:

travel-ai-search/
├── README.md
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   └── decisions/
│
├── notebooks/
│
├── scripts/
│   ├── generate_dataset.py
│   ├── create_index.py
│   ├── ingest_data.py
│   └── evaluate.py
│
├── src/
│   └── travel_ai_search/
│       ├── **init**.py
│       │
│       ├── api/
│       │   ├── app.py
│       │   ├── routes/
│       │   └── schemas/
│       │
│       ├── config/
│       │
│       ├── domain/
│       │   ├── models.py
│       │   └── search.py
│       │
│       ├── ingestion/
│       │
│       ├── embeddings/
│       │   ├── base.py
│       │   ├── local.py
│       │   └── bedrock.py
│       │
│       ├── retrieval/
│       │   ├── lexical.py
│       │   ├── vector.py
│       │   ├── hybrid.py
│       │   └── fusion.py
│       │
│       ├── reranking/
│       │   ├── base.py
│       │   ├── local.py
│       │   └── bedrock.py
│       │
│       ├── query_understanding/
│       │   ├── intent.py
│       │   ├── entities.py
│       │   ├── rewrite.py
│       │   └── expansion.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── evaluator.py
│       │   └── datasets.py
│       │
│       ├── orchestration/
│       │   └── search_service.py
│       │
│       └── infrastructure/
│           ├── opensearch.py
│           └── bedrock.py
│
└── tests/
├── unit/
└── integration/

Do not create empty files/modules just to satisfy this target tree. Introduce modules as we actually need them.

---

# Travel domain model

Design a useful travel product model with fields such as:

* id
* hotel_name
* hotel_description
* destination
* region
* country
* latitude
* longitude
* star_rating
* customer_rating
* amenities
* board_types
* family_friendly
* adults_only
* beach_distance
* airport_distance
* activities
* tags
* available_departure_airports
* price
* currency
* available_months
* climate characteristics

Separate product attributes from specific offer/availability attributes if appropriate.

The design should demonstrate that some fields are:

* searchable text
* exact filterable metadata
* numeric/range filters
* categorical facets
* semantic embedding content

Create a function that explicitly constructs the textual representation used to generate product embeddings rather than blindly concatenating every database field.

---

# Dataset

Generate a deterministic synthetic dataset containing approximately 5,000–10,000 travel products initially.

It should contain realistic variety across destinations such as:

* Portugal
* Spain
* Canary Islands
* Balearic Islands
* Greece
* Cyprus
* Turkey
* Italy
* Croatia
* Morocco
* Caribbean
* Maldives
* Thailand

Include meaningful differences in:

* family suitability
* luxury
* nightlife
* quietness
* beaches
* sports
* activities
* board basis
* price
* ratings
* climate
* departure airports

Use a deterministic random seed.

The generated descriptions must be sufficiently different for semantic retrieval experiments to be meaningful.

Do not scrape websites.

---

# Search versions

Implement the system progressively.

## V1 — BM25 lexical retrieval

Implement OpenSearch BM25 retrieval.

Support:

* multi-field queries
* field boosting
* filters
* aggregations/facets

Provide an endpoint similar to:

GET /search/lexical?q=family+hotel+tenerife

Explain mappings and analyzers.

---

## V2 — Vector retrieval

Generate product embeddings.

Store vectors in OpenSearch.

Implement ANN search using HNSW.

Provide:

GET /search/vector?q=quiet+family+holiday+by+the+sea

Expose relevant HNSW configuration.

Document the recall/latency tradeoffs of HNSW parameters.

---

## V3 — Hybrid retrieval

Combine lexical BM25 and semantic vector retrieval.

Provide:

GET /search/hybrid?q=...

Use OpenSearch native hybrid functionality where appropriate.

Keep lexical and vector retrieval implementations independently testable.

---

## V4 — Reciprocal Rank Fusion

Implement and/or configure RRF.

Also create a small pure-Python RRF implementation so the algorithm is understandable and unit testable.

Compare:

* lexical only
* vector only
* hybrid with RRF

---

## V5 — Alternative fusion

Implement at least one alternative:

* normalized weighted score fusion

Allow configuration of lexical/vector weights.

Do not assume RRF is optimal.

Make fusion strategy configurable.

---

## V6 — Neural reranking

Introduce a reranking stage.

Architecture:

candidate generation
→ fusion
→ top N candidates
→ cross-encoder reranker
→ final top K

Implement a local cross-encoder reranker first.

Make `candidate_pool_size` and final `top_k` configurable.

Explain why reranking every document would be inefficient.

---

# Query understanding

Introduce a structured query model approximately like:

```python
class SearchIntent(...):
    semantic_query: str
    destination: str | None
    departure_airport: str | None
    departure_month: str | None
    max_price: float | None
    min_star_rating: int | None
    hard_constraints: ...
    soft_preferences: ...
```

Do not use this exact schema blindly; design it properly.

The system should identify:

* user intent
* entities
* hard constraints
* soft preferences
* semantic concepts

Example:

"Family holiday somewhere warm in October from Manchester under £2000, preferably all-inclusive and near a beach."

Should result approximately in:

semantic:
"warm family beach holiday"

hard constraints:
departure_airport = MAN
month = October
price <= 2000

soft preferences:
family-friendly
all-inclusive
near beach

Keep the original query for observability and evaluation.

---

# Query rewriting

Implement optional query rewriting.

Original:

"Something like Mallorca but quieter"

Possible rewritten semantic query:

"quiet Mediterranean island beach destination similar to Mallorca with lower tourist density"

Do NOT replace the original query permanently.

Record both.

Query rewriting must be feature-toggleable and evaluatable.

---

# Query expansion / multi-query retrieval

Implement optional generation of several retrieval queries from one user query.

Example:

"quiet Mediterranean family holiday"

could generate:

* quiet Mediterranean family resort
* peaceful child-friendly beach destinations southern Europe
* family coastal resorts away from nightlife

Retrieve candidates independently and fuse the candidate sets.

This capability must be optional because we want to measure whether it actually improves search quality.

---

# Evaluation framework

This is a critical project requirement.

Create a manually defined/synthetic golden relevance dataset containing at least 50 search queries initially, ideally expanding toward 100.

Include several query classes:

* exact destination
* exact hotel
* semantic discovery
* family
* couples
* luxury
* budget
* nightlife
* quiet
* multi-constraint
* vague discovery
* similarity queries
* natural-language queries

Represent graded relevance where possible:

0 = irrelevant
1 = somewhat relevant
2 = relevant
3 = highly relevant

Implement from first principles, with unit tests:

* Precision@K
* Recall@K
* HitRate@K
* Reciprocal Rank
* MRR
* DCG
* NDCG@K

MAP may also be implemented.

Where practical, validate metric implementations against a trusted library.

Create an evaluation CLI such as:

`uv run python scripts/evaluate.py`

or preferably a proper CLI entry point.

It should compare:

* BM25
* vector
* hybrid
* hybrid + RRF
* hybrid + weighted fusion
* hybrid + reranker
* query rewrite + hybrid
* multi-query + hybrid

Output a table such as:

Strategy | Recall@10 | MRR | NDCG@10 | p50 latency | p95 latency

Also save machine-readable experiment results.

Do not evaluate only global averages.

Support segmentation by query class.

---

# AWS Bedrock integration

AWS must be OPTIONAL.

The whole system must run locally without AWS credentials.

Implement provider interfaces.

For example:

EmbeddingProvider

with implementations:

* LocalEmbeddingProvider
* BedrockEmbeddingProvider

LLMProvider:

* optional local/mock provider
* BedrockLLMProvider

Reranker:

* LocalCrossEncoderReranker
* BedrockReranker

Use boto3.

Prefer current AWS Bedrock APIs such as the Converse API for supported conversational model interactions.

Use Bedrock for appropriate experiments involving:

* query understanding
* query rewriting
* embeddings
* reranking

Do not hard-code model IDs throughout the application.

Put model selection in configuration.

Never commit AWS credentials.

Document required IAM permissions at a high level.

---

# RAG

Add a small optional destination knowledge base.

Examples of knowledge:

* destination characteristics
* seasonal climate
* activities
* suitability for families
* nightlife characteristics
* geographic facts

Demonstrate the conceptual difference between:

product retrieval

and:

knowledge retrieval / RAG.

Do not turn the entire search engine into a chatbot.

Search remains the core architecture.

---

# Graph concepts

Create a lightweight representation showing relationships such as:

Airport -> FLIES_TO -> Destination
Destination -> CONTAINS -> Resort
Resort -> CONTAINS -> Hotel
Hotel -> NEAR -> Beach
Hotel -> HAS_AMENITY -> KidsClub
Destination -> SIMILAR_TO -> Destination

A full graph database is optional.

Start with a simple implementation or documentation demonstrating use cases where graph traversal provides something vector similarity cannot.

Only introduce Neo4j or another graph system if there is clear learning value.

---

# FastAPI

Expose useful APIs.

At minimum consider:

GET /health

GET /search/lexical

GET /search/vector

GET /search/hybrid

POST /search

POST /query/understand

POST /evaluate

The final `/search` endpoint should orchestrate the complete pipeline.

Use proper request/response Pydantic schemas.

Expose timings for pipeline stages in development/debug responses where useful.

---

# Observability

Instrument the search pipeline.

Capture at least:

* total request latency
* query-understanding latency
* lexical retrieval latency
* vector retrieval latency
* reranking latency
* result count
* fallback usage
* errors
* selected search strategy

Use structured logging.

Design metrics so that Prometheus integration could be added easily.

Do not add a giant observability platform during the early stages.

---

# Resilience

Design sensible fallback behaviour.

For example:

If LLM query rewriting fails:
→ use original query.

If vector embedding generation fails:
→ degrade to BM25.

If reranker fails:
→ return fused retrieval results.

A search service should degrade gracefully where possible.

Add tests for relevant fallbacks.

---

# Configuration and feature flags

Make these configurable:

* embedding provider
* embedding model
* reranker provider
* reranker model
* top_k
* candidate pool
* lexical weight
* vector weight
* fusion algorithm
* query rewriting enabled
* query expansion enabled
* reranking enabled

This should allow search experiments without changing application code.

---

# Development workflow

Use `uv` exclusively for Python project management.

A `Makefile` is required and must be kept up to date. Every milestone must add its new commands as named targets. Developers should never need to remember raw `uv` or Docker invocations — `make <target>` is the interface.

The canonical workflow after cloning:

```bash
make install      # uv sync
make env          # copy .env.example → .env
make up           # docker compose up -d
make health       # verify OpenSearch connectivity
make test         # unit tests (no infrastructure required)
```

Key targets that must always work:

```bash
make check        # full quality gate: lint + format check + types + unit tests
make test         # unit tests
make test-integration  # integration tests (requires make up)
make serve        # FastAPI dev server (Milestone 3+)
```

Use Docker Compose for OpenSearch. Do not require Python to run inside Docker for local development unless there is a compelling reason.

Each milestone's implementation instructions should reference `make` targets, not raw commands.

---

# Testing strategy

Add tests progressively.

Use:

unit tests:

* metrics
* RRF
* score fusion
* query parsing
* domain logic

integration tests:

* OpenSearch indexing
* lexical search
* vector search
* hybrid search

Avoid tests that rely unnecessarily on live AWS infrastructure.

Mock provider boundaries for most Bedrock tests.

---

# README

The README should eventually contain:

1. Project motivation
2. Architecture
3. Search pipeline diagram
4. Technologies
5. Installation using uv
6. Starting OpenSearch
7. Dataset generation
8. Indexing
9. Running the API
10. Example searches
11. Evaluation
12. Experiment results
13. AWS Bedrock optional configuration
14. Architecture decisions
15. Future work

Include Mermaid diagrams where useful.

---

# Important learning behaviour

This project is primarily educational.

Therefore, whenever implementing a major capability:

1. Briefly explain what problem it solves.
2. Explain the main design choice.
3. Explain important alternatives.
4. Implement the smallest clean version.
5. Add tests.
6. Show me how to run it.
7. Give me 2–3 experiments I should manually perform to understand its behaviour.
8. keep an EXPERIMENTS.md in docs folder. Every time you compare BM25/vector/hybrid/reranking/query rewriting, write down the hypothesis, configuration, metrics and what surprised you.

Do NOT generate every feature in this prompt immediately.

---

# Implementation sequence

Work incrementally using the following milestones:

Milestone 0
Project scaffolding with uv + Docker Compose + OpenSearch + basic tests.

Milestone 1
Synthetic travel dataset.

Milestone 2
OpenSearch mappings and ingestion.

Milestone 3
BM25 lexical retrieval.

Milestone 4
Evaluation framework and BM25 baseline.

Milestone 5
Embeddings and vector retrieval.

Milestone 6
Hybrid retrieval.

Milestone 7
RRF and alternative fusion.

Milestone 8
Cross-encoder reranking.

Milestone 9
Query understanding and structured constraints.

Milestone 10
Query rewriting.

Milestone 11
Multi-query retrieval.

Milestone 12
AWS Bedrock provider implementations.

Milestone 13
RAG / travel knowledge.

Milestone 14
Graph-enhanced retrieval prototype/concept.

Milestone 15
Production API, observability, resilience and final evaluation.

---

# Your first task

DO NOT implement all milestones now.

Start with **Milestone 0 only**.

Before writing code:

1. Inspect the current repository if one already exists.
2. Propose the concrete Milestone 0 architecture.
3. State the Python version you recommend and why.
4. Identify the minimal dependencies required now.
5. Show the proposed initial repository structure.
6. Identify any important OpenSearch/Docker compatibility considerations.
7. Then implement Milestone 0.

Milestone 0 is complete when:

* the project is initialized and managed by `uv`;
* `pyproject.toml` and `uv.lock` exist;
* OpenSearch can be started with Docker Compose;
* Python can connect to OpenSearch;
* a health-check script or endpoint verifies connectivity;
* pytest works;
* Ruff works;
* configuration is cleanly handled;
* `.env.example` exists;
* README contains exact setup/run commands;
* `uv run pytest` succeeds.

After implementing it, stop.

Give me:

* what you created;
* the architecture decisions you made;
* commands to run it;
* tests to execute;
* anything I should inspect manually;
* three short questions that test whether I understood what we just built.

Wait for me to tell you to proceed to Milestone 1.
