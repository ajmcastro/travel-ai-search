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
21. LLM-as-judge evaluation and methodology validity
22. Learned sparse retrieval and neural vocabulary expansion (SPLADE)
23. Late interaction retrieval with multi-vector representations (ColBERT)
24. Two-tower model fine-tuning and domain adaptation

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

# LLM-as-judge evaluation

## Purpose

The golden relevance dataset used throughout Milestones 4–15 is attribute-based: a hotel is marked relevant for a query when its structured fields (country, `family_friendly`, price, star rating, `adults_only`, etc.) match the query's constraints programmatically. This is fast, deterministic, and reproducible, but it creates a circularity risk — the retrieval methods are ranked against a notion of relevance defined by the same pipeline that generated the corpus.

An LLM-as-judge evaluation layer replaces (or supplements) the attribute-based labels with natural-language relevance judgments: given a query and a retrieved hotel description, ask an LLM to score how relevant the hotel is. This is a fundamentally different signal — the judge reads the text, not the structured fields.

## The common-mode bias problem

**If the LLM used to judge relevance is from the same model family that generated the synthetic hotel descriptions, the judge and the generator share the same linguistic biases and internal knowledge representation. Their agreement reads as high precision, but part of that agreement is structural (shared bias) rather than a genuine signal of retrieval quality.**

Concretely: if Claude generated the hotel descriptions and Claude judges the results, both will tend to prefer documents that reflect Claude's internal sense of what "beach hotel" or "luxury spa" means. A retrieval method that happens to surface documents whose phrasing aligns with that internal representation will score well — not because it serves users better, but because it mirrors the generator.

## Design requirements

**Judge model must differ from the generator model.** Use a different provider or a provably independent model family (e.g., a Cohere or Mistral judge if Claude generated the data, or vice versa). Document the model choice and the reasoning in `EXPERIMENTS.md`.

**Human-annotated held-out slice.** Collect a small set of queries written by a person who read the hotel documents but never saw the generation prompts or the golden dataset construction code. Evaluate all retrieval strategies on this human slice separately from the generated golden queries. The gap in relative method rankings between the two slices is the empirical size of the generator effect.

**Do not replace the golden dataset.** LLM judge scores are an additional evaluation layer, not a replacement. The existing attribute-based golden dataset remains the primary metric for reproducibility. Report both.

**Report statistical uncertainty.** At 62 generated queries and a small human slice, confidence intervals are wide. Every reported comparison must include an indication of the sample size per slice. Do not present method rankings as definitive if the slices are too small to support significance claims.

## What to implement

- `JudgeProvider` Protocol with at least two implementations:
  - `EchoJudgeProvider` — returns a fixed score (for tests and dry runs)
  - `BedrockJudgeProvider` — calls a Bedrock LLM (different family from the generator) with a structured scoring prompt; returns a score in [0, 3] and a brief rationale
- A scoring prompt that presents the query, the hotel name, and the hotel description, asks for a relevance score (0 = irrelevant, 1 = somewhat relevant, 2 = relevant, 3 = highly relevant), and requests a one-sentence rationale. The prompt must not reveal that the data is synthetic.
- `LLMEvaluator` — runs judge scoring over a set of (query, hit) pairs, computes per-query mean judge score, and compares against golden labels (Pearson/Spearman correlation, agreement rate at grade ≥ 2).
- A human-annotated query file (`data/evaluation/human_queries.jsonl`) with a documented annotation guide (`docs/annotation_guide.md`). The annotation process must be described so a collaborator can independently produce judgments.
- `scripts/evaluate_judge.py` — CLI that accepts `--strategy`, `--judge-provider`, `--slice` (generated | human | both), and outputs a comparison table with sample sizes and correlation coefficients.
- A new `POST /evaluate/judge` API endpoint that runs judge scoring on a live search result set.
- Makefile target: `make evaluate-judge`.

## What to measure

| Metric | Description |
|---|---|
| Mean judge score per strategy | Average LLM relevance score across all (query, hit) pairs at K |
| Kendall's τ / Spearman ρ | Rank correlation between method ordering under golden labels vs LLM judge |
| Label agreement rate | Fraction of (query, hotel) pairs where judge grade ≥ 2 matches golden grade ≥ 2 |
| Generator-effect gap | Difference in relative method rankings between generated-query slice and human-query slice |

## Methodology validity note

Publish the judge model identity (provider, model ID, version) alongside every set of LLM judge results so readers can assess independence from the generator. If two evaluations use different judge models, their scores are not directly comparable — always report the judge identity.

---

# Learned sparse retrieval (SPLADE)

## Purpose

BM25 fails when the query and the document use different words for the same concept — "child-friendly" vs "family hotel", "budget" vs "affordable", "peaceful" vs "quiet".  Dense bi-encoders (Milestone 5) solve the vocabulary mismatch problem, but they produce dense vectors where every dimension is active, which requires approximate nearest-neighbour search and makes exact top-K retrieval expensive.

Learned sparse retrieval combines the best properties of both worlds: the vocabulary-level interpretability and inverted-index efficiency of BM25, plus the semantic expansion of neural models.  A transformer model (typically a BERT-based MLM head) takes a query or document and outputs a sparse vector over the full vocabulary (~30,000 terms), where each non-zero value is a learned importance weight.  The model can assign weight to terms that do not appear in the original text, effectively expanding the vocabulary in a data-driven way.

The canonical implementation is **SPLADE** (Sparse Lexical AnD Expansion, Formal et al. 2021).

## Key concept

Given a query `q`, SPLADE computes:

```
w_t = log(1 + ReLU(h_t)) · IDF(t)   for each vocabulary term t
```

where `h_t` is the MLM head output for token position `t`, summed across all input positions.  The result is a sparse weight vector over the vocabulary.  The IDF weighting is optional; some variants omit it and rely entirely on learned weights.

For retrieval, the score between query `q` and document `d` is the dot product of their sparse weight vectors — identical in form to BM25, but with learned weights rather than term-frequency statistics.  Because most weights are zero (sparsity regularised via FLOPS loss during training), the inverted index representation remains efficient.

**Comparison with M5 dense retrieval:**

| Property | BM25 | Dense (M5) | SPLADE |
|---|---|---|---|
| Vocabulary | Exact terms only | None (latent space) | Vocabulary + expansion |
| Index type | Inverted | HNSW / ANN | Inverted |
| Retrieval cost | O(posting list) | O(n · dim) approx | O(sparse posting list) |
| Semantic expansion | No | Yes (implicit) | Yes (explicit, interpretable) |
| OOV handling | Fail | Partial | Partial |

## Design requirements

- Implement a `LearnedSparseProvider` Protocol analogous to `EmbeddingProvider`: takes text, returns a `dict[str, float]` of term → weight.
- Implement `LocalSparseProvider` using a HuggingFace SPLADE checkpoint (e.g., `opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill` for documents, and its companion query encoder).
- Create a new OpenSearch index mapping with a `rank_features` field type to store sparse vectors.
- Implement `sparse_search()` in `retrieval/sparse.py`: builds a `neural_sparse` query (OpenSearch ≥ 2.10) or a `rank_features` query with BM25 dot product scoring.
- Support two SPLADE modes:
  - **Bi-encoder mode**: encode both query and document at query time (more accurate, slower at index time).
  - **Doc-only mode**: encode documents offline at index time; use a lightweight query tokeniser at query time (faster, recommended for production).
- Add `GET /search/sparse?q=...` endpoint.
- Add a Makefile target `make sparse-search` and `make generate-sparse-embeddings`.
- AWS is optional: a `BedrockSparseProvider` may be added if Bedrock exposes a compatible model, but the full system must run locally.

## What to implement

- `retrieval/sparse.py` — `SparseSearchParams`, `SparseSearchResult`, `sparse_search()`
- `embeddings/sparse.py` — `LearnedSparseProvider` Protocol, `LocalSparseProvider`
- Scripts: `scripts/generate_sparse_embeddings.py`
- Updated index mapping with `rank_features` field (`sparse_embedding`)
- Unit tests: provider output shape (dict, non-negative weights), query construction, score aggregation
- Integration test: end-to-end sparse search against a populated index

## What to measure

| Metric | Description |
|---|---|
| NDCG@10 vs BM25, Vector, RRF | Primary quality comparison |
| Vocabulary expansion | Inspect top weighted terms for sample queries — are they semantically relevant? |
| Index size | Compare `rank_features` index size vs `knn_vector` field |
| Query latency | Compare sparse query p50/p95 vs BM25 and vector |
| Sparsity | Mean non-zero terms per query/document vector |

Run the existing evaluation CLI (`scripts/evaluate.py --strategy sparse`) and document results in `EXPERIMENTS.md`.

---

# ColBERT — late interaction retrieval

## Purpose

The retrieval methods implemented in Milestones 3–11 represent two extremes of the accuracy-efficiency spectrum:

- **Bi-encoder (M5)**: query and document each compressed to a single vector at index time.  Very fast at query time (ANN lookup), but information is lost in compression.
- **Cross-encoder (M8)**: query and document processed jointly by the model at query time.  Very accurate (no early compression), but O(candidates) model forward passes — too slow for first-stage retrieval.

**ColBERT** (Khattab & Zaharia, 2020) occupies the middle ground.  It encodes the query and document independently (like a bi-encoder) but retains all token-level embeddings rather than collapsing to a single vector.  Relevance is computed at query time via a **MaxSim** operation: for each query token embedding, find its maximum dot product across all document token embeddings.  The final score is the sum of these per-token maximums.

```
Score(q, d) = Σ_{i=1}^{|q|}  max_{j=1}^{|d|}  (q_i · d_j)
```

This is called **late interaction**: the query and document representations interact at the token level, but only after independent encoding — much cheaper than a full cross-encoder forward pass.

## Key concept

**Why MaxSim is more expressive than a single dot product:**

A single-vector bi-encoder compresses the full semantics of a document into one point in embedding space.  A 512-token hotel description must be represented by a 384-dimensional vector — a severe bottleneck.  MaxSim allows each query token to independently seek out the most relevant part of the document.  "Family" can match the paragraph about child facilities; "beach" can match the section about the waterfront — the model does not have to average these signals into a single representation.

**Storage and retrieval cost:**

Each document now stores `M × embedding_dim` floats (where M = sequence length, e.g. 128 tokens × 128 dims = 16,384 floats per document) rather than `embedding_dim` floats.  This is a significant storage increase.  At scale, PLAID (an efficient ColBERT serving system) uses centroid-based compression; for this project a simpler two-stage approach is sufficient:

1. **Candidate generation**: retrieve top-K candidates using the existing bi-encoder ANN index (fast).
2. **ColBERT re-scoring**: encode the query with the ColBERT model, load the stored token embeddings for each candidate, compute MaxSim scores, re-rank.

This two-stage design reuses the existing retrieval infrastructure and avoids full-corpus MaxSim computation.

## Design requirements

- Implement a `ColBERTReranker` that satisfies the existing `Reranker` Protocol (drop-in replacement for `LocalCrossEncoderReranker`).
- The reranker must:
  1. Encode the query into N token embeddings using a ColBERT checkpoint.
  2. Load pre-stored document token embeddings for each candidate hit.
  3. Compute MaxSim scores and return re-ranked results.
- Store per-document token embeddings offline (script: `scripts/generate_colbert_embeddings.py`).  Store as a serialised numpy array alongside the hotel ID, not in OpenSearch (OpenSearch does not natively support multi-vector per document in this form).
- New field in the OpenSearch document or a separate file-based store: `data/processed/colbert_embeddings/` — one `.npy` file per hotel or a single memory-mapped array.
- Add `colbert_reranker_enabled` feature flag in settings.
- Add Makefile target `make generate-colbert-embeddings`.
- The full system must run without ColBERT if the embeddings are not generated (graceful degradation to cross-encoder or no reranking).

## What to implement

- `reranking/colbert.py` — `ColBERTReranker` implementing the `Reranker` Protocol
- `scripts/generate_colbert_embeddings.py` — offline token-embedding generation
- Unit tests: MaxSim computation (known inputs → expected score), token embedding shape, graceful fallback when embeddings missing
- Integration test: ColBERT re-scores a candidate set correctly (scores differ from bi-encoder order)

## What to measure

Compare ColBERT re-scoring against cross-encoder re-scoring on the same candidate set:

| Metric | Description |
|---|---|
| NDCG@10 vs cross-encoder (M8) | Does MaxSim match or exceed the cross-encoder at the same candidate pool size? |
| Latency vs cross-encoder | MaxSim is cheaper than a full cross-encoder forward pass; quantify the saving |
| Query-class breakdown | Does ColBERT benefit specific query classes more than others? |
| Score agreement with cross-encoder | Spearman ρ between ColBERT and cross-encoder ranking of the same candidates |
| Storage overhead | Token embedding file size vs single-vector embedding size |

Add `--strategy colbert` to `scripts/evaluate.py` and document results in `EXPERIMENTS.md`.

---

# Two-tower model fine-tuning

## Purpose

The bi-encoder used in Milestone 5 (`all-MiniLM-L6-v2`) was trained on general-purpose text pairs (MS MARCO, NLI, STSb, etc.).  It has no exposure to travel vocabulary, hotel descriptions, or the specific relevance relationships in this dataset.  Milestone 19 asks a concrete empirical question: **does fine-tuning the bi-encoder on domain-specific signal improve retrieval quality, and by how much?**

This milestone also introduces the training side of two-tower models — the contrastive learning objectives, the data construction problem, and the hard negative mining strategies that are central to modern dense retrieval training.

## Key concept

**Two-tower architecture (training view):**

Both query and document encoders are trained jointly with a contrastive objective.  During training, a batch contains `B` (query, positive document) pairs.  The positive document for query `i` serves as a hard negative for query `j ≠ i` (in-batch negatives).

The loss function is **MultipleNegativesRankingLoss** (equivalent to InfoNCE / NT-Xent):

```
L = -log( exp(sim(q, d+) / τ) / Σ_k exp(sim(q, d_k) / τ) )
```

where `d+` is the positive hotel description, `d_k` ranges over all documents in the batch (positives of other queries become negatives), and `τ` is a temperature parameter.

**Why random negatives are insufficient:**

If the negatives are sampled randomly from the full corpus, they are trivially easy — a query "adults-only luxury spa" will not be confused with a random family budget hotel.  The model quickly learns to push apart very different documents and stops improving.  Hard negatives — documents that look similar to the positive but are not relevant — are needed for the model to learn fine-grained discriminations.

**Hard negative sources:**

1. **BM25 top-K non-relevant**: hotels ranked highly by BM25 for the query but with golden grade 0.
2. **Dense top-K non-relevant**: hotels ranked highly by the current bi-encoder but irrelevant.
3. **Cross-encoder ranked non-relevant from dense top-K**: the cross-encoder identifies the truly irrelevant ones among the dense candidates.

For this project, a mix of BM25 and dense hard negatives is sufficient.

## Design requirements

- Add `scripts/fine_tune_embeddings.py` — constructs training triplets from the golden dataset, runs contrastive fine-tuning using `sentence-transformers` training API, and saves the fine-tuned model checkpoint.
- Training data construction:
  - For each golden query with at least one relevant hotel (grade ≥ 2): the positive is the hotel description.
  - Hard negatives: top-K hotels from BM25 with golden grade 0.
  - Produce `data/evaluation/fine_tuning_pairs.jsonl` — one JSON object per training pair: `{"query": "...", "positive": "...", "negative": "..."}`.
- Fine-tuned model saved to `data/models/bi-encoder-travel/` (gitignored).
- Add `fine_tuned_embedding_model_path` setting in `settings.py`; when set, `LocalEmbeddingProvider` loads the fine-tuned model instead of `all-MiniLM-L6-v2`.
- Add Makefile targets: `make prepare-fine-tuning-data` and `make fine-tune-embeddings`.
- After fine-tuning: re-generate embeddings with the fine-tuned model (`make generate-embeddings`) and re-run the evaluation (`scripts/evaluate.py --strategy vector --model fine-tuned`).
- The base model must remain the default; fine-tuning is opt-in via configuration.

## What to implement

- `scripts/fine_tune_embeddings.py` — data preparation + training loop
- `scripts/prepare_fine_tuning_data.py` — hard negative mining from BM25 and dense rankings
- Updated `LocalEmbeddingProvider` to accept a model path override
- Unit tests: training pair construction (correct positive/negative structure), hard negative mining (negatives have golden grade 0)
- Document the training configuration (epochs, batch size, learning rate, loss function) in `EXPERIMENTS.md`

## What to measure

| Metric | Description |
|---|---|
| NDCG@10: fine-tuned vs base model | Primary measure of domain adaptation benefit |
| Per-query-class breakdown | Which classes benefit most from fine-tuning? |
| Training data size sensitivity | How does NDCG change as the number of training pairs increases? |
| Embedding space visualisation (optional) | t-SNE of query and document embeddings before/after fine-tuning — do clusters tighten? |
| Latency impact | Fine-tuned model may have different inference speed; measure p50/p95 |

Add `--strategy fine-tuned-vector` to `scripts/evaluate.py` and document results in `EXPERIMENTS.md`.  Report whether the improvement justifies the fine-tuning cost and data requirements.

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

Milestone 16
LLM-as-judge evaluation: `JudgeProvider` abstraction, `BedrockJudgeProvider` (independent model family from the generator), `LLMEvaluator`, human-annotated held-out query slice, generator-effect measurement, and `POST /evaluate/judge` endpoint.

Milestone 17
Learned sparse retrieval: `LearnedSparseProvider` Protocol, `LocalSparseProvider` (SPLADE checkpoint), `rank_features` index field, `sparse_search()`, offline sparse embedding generation, `GET /search/sparse` endpoint, evaluation against BM25 and dense vector baselines.

Milestone 18
ColBERT late interaction: `ColBERTReranker` implementing the `Reranker` Protocol, offline token-embedding generation (`generate_colbert_embeddings.py`), MaxSim re-scoring over bi-encoder candidates, evaluation against the cross-encoder reranker on the same candidate pool.

Milestone 19
Two-tower fine-tuning: training pair construction from the golden dataset, hard negative mining from BM25 and dense rankings, contrastive training with `MultipleNegativesRankingLoss`, fine-tuned model checkpoint, evaluation of base vs fine-tuned bi-encoder across query classes.

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
