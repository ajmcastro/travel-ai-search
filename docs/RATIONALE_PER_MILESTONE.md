# Rationale per Milestone

This document captures — for each completed milestone — the **IR concepts introduced**, the **design choices made**, and **why** those choices were made over the alternatives. Its purpose is to be a self-contained learning reference: you should be able to re-read any section independently and understand both what was built and the reasoning behind it.

---

## Milestone 0 — Project scaffold

### What we added

A clean, reproducible project skeleton: Python 3.12 managed by `uv`, OpenSearch running in Docker Compose, configuration via environment variables, Ruff for linting and formatting, mypy for type checking, and a pytest suite that verifies OpenSearch connectivity.

### Concepts

#### `uv` as the exclusive package manager

Traditional Python packaging (`pip` + `requirements.txt`, Poetry, Conda) solves the same problem differently. `uv` was chosen because:

- It produces a **lockfile** (`uv.lock`) that pins every transitive dependency to an exact version, making builds fully reproducible across machines and CI.
- Its `uv run` command executes commands inside the virtual environment without requiring the user to activate it — important for a project where every command is documented in a Makefile.
- It is significantly faster than pip at dependency resolution and installation.
- It has first-class support for the `src` layout and `pyproject.toml`, which is the modern standard.

#### The `src` layout

Placing the package under `src/travel_ai_search/` rather than at the repo root prevents a subtle import bug: if you run `python` or `pytest` from the repo root without the `src` layout, Python adds `.` to `sys.path` and your uninstalled local package shadows an installed one. With `src`, the package is only importable after `uv sync` installs it in development mode — so tests always test the installed package, not whatever happens to be in the working directory.

#### OpenSearch in Docker Compose

OpenSearch is the search backend for the entire project. Running it in Docker Compose means:

- **Reproducibility**: every developer, and CI, gets the exact same version (`2.15.0`) with the same configuration.
- **Isolation**: the search cluster does not interfere with any other software on the host.
- **Security plugin disabled** (`DISABLE_SECURITY_PLUGIN=true`): for local development, TLS and authentication add friction without educational value. In production these would be enabled.

#### Configuration via environment variables and `pydantic-settings`

`Settings` (in `src/travel_ai_search/config/settings.py`) inherits from `BaseSettings`. This means every field can be overridden by an environment variable of the same name (uppercased). The `.env` file is loaded automatically for local development; the real `.env` is gitignored and only `.env.example` is committed.

The `@lru_cache` on `get_settings()` means settings are parsed exactly once for the lifetime of the process. In tests, `get_settings.cache_clear()` resets this so `monkeypatch.setenv` takes effect.

This pattern separates configuration from code and is a core principle of 12-factor app design.

#### Infrastructure factory function

`create_client(settings: Settings) -> OpenSearch` in `infrastructure/opensearch.py` is a factory that builds an `OpenSearch` client from the `Settings` object. Isolating client construction here means:

- Tests that need to mock or skip the client only need to patch one function.
- The rest of the codebase never sees connection details — only the `Settings` object.
- Switching to a different connection strategy (e.g. adding authentication headers for production) requires changing exactly one place.

#### Ruff for linting and formatting

Ruff replaces `flake8`, `isort`, `pyupgrade`, and `black` in a single binary. It applies the same rule sets but runs an order of magnitude faster. Using one tool for both linting and formatting removes the friction of keeping multiple tools consistent.

#### Testing strategy at M0

M0 includes one integration test file: `tests/integration/test_opensearch_connection.py`. It verifies that:

1. The OpenSearch cluster is reachable.
2. The version is 2.x (so future version-specific features can be relied upon).
3. Cluster health is at least "yellow" (documents are accessible).
4. At least one data node is present.

If OpenSearch is not running, the fixture calls `pytest.skip(...)` — tests are skipped cleanly rather than failing, preserving the principle that unit tests never depend on infrastructure.

### Design decisions and alternatives considered

| Decision | Chosen | Alternative considered | Reason |
|---|---|---|---|
| Package manager | `uv` | Poetry, pip | Speed, lockfile, `uv run` ergonomics |
| Search backend | OpenSearch 2.15 | Elasticsearch | Open-source licence, AWS-compatible, same API |
| Configuration | `pydantic-settings` | `python-dotenv` + dataclasses | Validated types, auto-env mapping |
| Layout | `src/` layout | flat layout | Prevents accidental uninstalled imports |
| Linter/formatter | Ruff | black + flake8 + isort | Single fast tool covering all use cases |

---

## Milestone 1 — Synthetic travel dataset

### What we added

A deterministic synthetic dataset generator (`scripts/generate_dataset.py`) that produces ~5 000 travel hotel products as a JSONL file (`data/raw/hotels.jsonl`). The domain model (`TravelProduct`) was defined with Pydantic, including a separate `build_embedding_text()` function for future vector search.

### Concepts

#### Why synthetic data

Scraping real hotel data is legally problematic (terms of service), technically brittle, and risks including proprietary information. Generating synthetic data with a **fixed random seed** gives:

- **Reproducibility**: running the generator twice produces identical output.
- **Control**: we can deliberately include the variety (destinations, price ranges, family suitability, etc.) that makes evaluation meaningful.
- **Safety**: no legal, IP, or privacy concerns.

#### The domain model (`TravelProduct`)

Every field in `TravelProduct` was classified by its intended role in search. This classification directly drives the OpenSearch mapping in Milestone 2:

| Role | Fields | Why |
|---|---|---|
| **Full-text / BM25** | `hotel_name`, `hotel_description`, `activities`, `amenities` | Need tokenisation, stemming, stop-word removal for relevance scoring |
| **Exact filter / facet** | `destination`, `country`, `board_types`, `available_months`, `climate_zone` | Must match exactly; no tokenisation needed |
| **Numeric / range** | `star_rating`, `customer_rating`, `price_per_person_gbp`, `beach_distance_km` | Range queries like `price ≤ 1500` |
| **Boolean** | `family_friendly`, `adults_only` | Hard yes/no filters |
| **Geo** | `latitude`, `longitude` | Stored as floats for display; combined into a `geo_point` at ingest for distance queries |
| **Embedding input** | derived via `build_embedding_text()` | Only semantically meaningful fields; excludes noisy numbers |

This up-front classification is not cosmetic. If you accidentally index a price field as `text`, BM25 will try to analyse "1499" as a word and range queries will fail silently.

#### Pydantic validation and domain invariants

`TravelProduct` uses Pydantic v2 (`BaseModel`). Two important design choices:

1. **Field-level constraints** (`Field(ge=1, le=5)` for `star_rating`, `ge=-90, le=90` for `latitude`, etc.) enforce data quality at the boundary — the moment a product is constructed, not later when it reaches the search engine.

2. **Cross-field validation** (`@model_validator(mode="after")`): a hotel cannot be both `family_friendly=True` and `adults_only=True`. A single-field validator cannot catch this invariant; the model validator sees the fully-constructed object and raises a `ValueError` if the combination is logically impossible.

#### `build_embedding_text()` — a deliberate embedding contract

Vector search (Milestone 5) requires converting each hotel to a text string that the embedding model encodes as a dense vector. This function was defined in M1 deliberately — not at embedding time — because:

- The choice of **what to include** defines the semantic space. Include price and you get vectors that cluster by price, not by destination character.
- The choice of **what to exclude** is equally important: star rating, price, distances, and airport codes are structural filter fields that carry no semantic meaning for similarity ("a £499 Menorca hotel is not semantically similar to a £499 Mykonos hotel").
- Defining it early, with tests, means the embedding contract is stable and explicit before we ever run an embedding model.

The function includes: hotel name + location, description, board types (replaced underscores), activities, tags, climate zone, and a human-readable family/adults flag.

#### JSONL format

JSON Lines (one JSON object per line) was chosen over a single JSON array because:

- Individual lines can be streamed and parsed without loading the entire file into memory.
- Line-by-line format is trivially appendable — adding new products does not require re-serialising the whole file.
- It maps directly to the OpenSearch Bulk API format, which also works line by line.

#### Cluster-based generation for realistic variety

The generator creates destination clusters (e.g. Costa del Sol hotels share the same `country`, `region`, `climate_zone`) and then randomly samples product attributes within each cluster's plausible ranges. This avoids both the uniformity of pure random generation and the over-representation that would come from generating hotels independently.

### Design decisions and alternatives considered

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| Data format | JSONL | CSV, single JSON array | Streaming, appendable, bulk-API compatible |
| Validation | Pydantic v2 `BaseModel` | `dataclasses` + manual checks | Validated types, JSON round-trip, `.model_validate_json()` |
| Embedding text | Separate `build_embedding_text()` | Concatenate all fields | Explicit control over semantic space; testable contract |
| Dataset size | ~5 000 documents | More or fewer | Large enough for meaningful BM25/vector evaluation; fast to ingest and query locally |

---

## Milestone 2 — OpenSearch mappings and ingestion

### What we added

An OpenSearch index schema (`INDEX_BODY`) with carefully chosen field types, functions to create and delete the index (`ingestion/index.py`), and a bulk ingestion pipeline (`ingestion/ingestor.py`) that loads products from JSONL and pushes them to OpenSearch. Two CLI scripts (`scripts/create_index.py`, `scripts/ingest_data.py`) expose these as `make` targets.

### Concepts

#### The inverted index — how BM25 text search works internally

When OpenSearch indexes a `text` field, it runs the raw string through an **analysis chain** and stores the result in an **inverted index**. The inverted index is a data structure that maps from each unique token (after analysis) to the list of documents that contain it — the opposite direction of a normal document store. This is what makes full-text search fast: instead of scanning every document for the query term, the engine looks up the term in the index and gets the matching document list in O(1) time.

The analysis chain has three stages:

1. **Character filters** — applied to the raw text before tokenisation (e.g. stripping HTML tags, normalising smart quotes).
2. **Tokeniser** — splits the text into tokens. The `standard` tokeniser splits on whitespace and punctuation. The `english` tokeniser does the same but is tailored for English text.
3. **Token filters** — transformations applied to each token after splitting:
   - **Lowercasing**: `"Beach"` → `"beach"` so the query `"beach"` matches.
   - **Stop-word removal**: common words like `"the"`, `"a"`, `"and"` are discarded. They appear in nearly every document so their IDF is close to zero — they add noise without improving ranking.
   - **Porter stemming**: reduces words to their root form: `"swimming"` → `"swim"`, `"beaches"` → `"beach"`. This is why a query for `"swim"` matches documents that contain `"swimming"`.

We chose the **English analyser** (which includes all three token filters) for description, hotel name, amenities, and activities. For exact filter fields we used **no analyser at all** — `keyword` type stores the string as-is, so `country = "Spain"` matches only `"Spain"`, not `"spain"` or `"Spaniard"`.

#### Text vs keyword: the fundamental mapping choice

Every field in OpenSearch must be declared as one type or the other (or both). Getting this wrong is one of the most common mapping mistakes:

- **`text`** fields are analysed and indexed into the inverted index. They support BM25 full-text queries. They **cannot** be used for exact filters, aggregations, or sorting because the stored tokens are normalised (lowercased, stemmed), not the original strings.
- **`keyword`** fields are stored as-is (no analysis). They support exact `term` filters, `terms` aggregations (facets), and sorting. They **cannot** be meaningfully searched with BM25 because the full string is treated as one token.

#### Multi-field mappings: having both

For fields where you need both BM25 search and exact filtering, OpenSearch supports **multi-field** mappings: the same raw value is indexed twice under different names, each with a different type.

In our schema:

```
"hotel_name": {
    "type": "text",            # primary: BM25 search
    "analyzer": "english",
    "fields": {
        "keyword": {"type": "keyword"}  # sub-field: exact sorting / aggregation
    }
}
```

This means:
- `hotel_name` → full-text BM25 search (e.g. "beach family resort")
- `hotel_name.keyword` → exact sort, de-duplicate, or aggregate by hotel name

We applied the same pattern in reverse for `destination` and `tags` — primary type is `keyword` (for exact filters) with a `.text` sub-field (for BM25 discovery queries):

```
"destination": {
    "type": "keyword",         # primary: exact term filter
    "fields": {
        "text": {"type": "text", "analyzer": "standard"}  # sub-field: BM25 partial match
    }
}
```

#### `geo_point` — a derived field

OpenSearch's `geo_point` type stores a latitude/longitude pair as a single binary-encoded value. This encoding enables fast **geo-distance** and **bounding-box** queries that would be slow if implemented as two separate numeric range queries.

The `TravelProduct` model stores `latitude` and `longitude` as plain floats — useful for display. At ingest time, `to_document()` adds a derived `"location": {"lat": ..., "lon": ...}` field that OpenSearch stores as a `geo_point`. This is not redundant: the float fields and the `geo_point` field serve different purposes.

Geo queries are not used in M3 but the field is created now because **you cannot change a field's type without re-indexing all documents**. Adding the geo_point to the mapping from the start avoids a future migration.

#### Index settings: shards and replicas

```json
"settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
}
```

- **Shards**: OpenSearch distributes an index across shards, each of which is a self-contained Lucene index. One shard is appropriate for a development dataset of ~5 000 documents — splitting data across multiple shards only helps when data volume or query throughput exceeds what one shard can handle.
- **Replicas**: a replica is a copy of a shard on a different node, providing both redundancy and extra read throughput. With `replicas=0` and a single-node cluster, OpenSearch marks the cluster as **green** immediately. With `replicas=1` on a single node, the cluster stays **yellow** indefinitely because there is nowhere to place the replica — unit tests that check cluster health would always see degraded state.

In production both values would be higher.

#### The Bulk API

Indexing documents one at a time with individual HTTP requests is very inefficient: each request incurs network round-trip latency and OpenSearch has to commit a transaction and refresh internal state. The **Bulk API** batches multiple operations into one HTTP request, amortising this overhead.

`helpers.bulk()` from `opensearch-py` takes a Python generator that yields one action dict per document — this is memory-efficient because it never loads all documents into memory at once. We use `chunk_size=500` so each HTTP request contains at most 500 documents.

#### Idempotent ingestion via document `_id`

Each OpenSearch document has a `_id`. If you index a document with the same `_id` as an existing document, OpenSearch **replaces** it (upsert semantics). By setting `_id = product.id` (the natural key from our domain model), running the ingestion pipeline twice produces the same result as running it once — no duplicates, no need to delete and re-create the index before every ingest.

#### Near-real-time search and forced refresh

OpenSearch (like Lucene underneath) writes new documents to an in-memory buffer and periodically flushes them to disk segments that are then made searchable. This flush happens approximately every second by default — meaning a document indexed at t=0 might not be searchable until t=1.

In the ingestion pipeline, `client.indices.refresh(index=index)` forces an immediate flush after bulk loading. Without this, integration tests that index documents and then immediately query for them would intermittently fail. In production, you would typically not force-refresh (it is expensive) and instead accept the ~1 second delay.

### Design decisions and alternatives considered

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| Analyser for text fields | English (stemming + stop words) | Standard (no stemming) | Better recall: "swimming" matches "swim" |
| `destination` primary type | `keyword` with `.text` sub-field | `text` with `.keyword` sub-field | Exact destination filter is the more common use case |
| Shard count | 1 | More | No benefit at ~5 000 doc scale |
| Replica count | 0 | 1 | Green cluster on single-node dev; yellow would confuse health checks |
| Ingest format | `helpers.bulk()` generator | Single-doc indexing | Efficiency; memory-efficient with large datasets |
| Re-ingestion strategy | Use product id as `_id` (upsert) | Delete-and-recreate index | Idempotent; no data loss if ingest is re-run |

---

## Milestone 3 — BM25 lexical retrieval

### What we added

A full BM25 retrieval layer (`retrieval/lexical.py`), a FastAPI application (`api/app.py`) with a `GET /search/lexical` endpoint, Pydantic response schemas, and a comprehensive test suite: 28 pure unit tests for query building, 8 unit tests for the API (with mocked OpenSearch), and 25 integration tests against a curated 6-hotel index.

### Concepts

#### BM25 — Best Match 25

BM25 is the relevance scoring function that OpenSearch uses by default. It answers the question: "for a given query, how relevant is this document?" It combines two classic IR signals:

**IDF — Inverse Document Frequency**

A term is more informative when it appears in fewer documents. The word "the" appears everywhere and distinguishes nothing. The word "thalassotherapy" appears rarely and is highly discriminating. IDF captures this:

```
IDF(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )
```

where `N` is the total number of documents and `df` is the number of documents containing term `t`. Rare terms get high IDF; common terms get low IDF.

**TF — Term Frequency (saturated)**

A document that mentions "beach" five times is probably more about beaches than one that mentions it once. But this relationship is not linear — mentioning "beach" 100 times is not 20× more relevant than mentioning it five times. BM25 uses a **saturation function**:

```
TF_sat(t, d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
```

where `tf` is the raw term count, `k1` (default 1.2) controls saturation speed, `b` (default 0.75) controls length normalisation, `|d|` is the document length, and `avgdl` is the average document length. The length normalisation term `(1 - b + b * |d| / avgdl)` ensures that a short document that mentions "beach" twice is scored higher than a long document that mentions it twice — the short document is more focused on the topic.

The final BM25 score for a multi-term query is:

```
score(d, Q) = Σ IDF(t) × TF_sat(t, d)   for each term t in query Q
```

#### `multi_match` — searching across multiple fields

A simple `match` query searches one field. `multi_match` searches several fields simultaneously and combines their scores. We chose **`best_fields`** type:

```json
{
  "multi_match": {
    "query": "luxury spa adults",
    "fields": ["hotel_name^3", "destination^2", "hotel_description", ...],
    "type": "best_fields",
    "tie_breaker": 0.3
  }
}
```

**`best_fields`** takes the BM25 score from the **highest-scoring field** as the main score, then adds a fractional contribution from all other matching fields (`tie_breaker × score_of_field`). This reflects the intuition that a query like "luxury spa" matching primarily in `hotel_description` is stronger signal than it weakly matching across five fields simultaneously.

The alternatives:
- **`most_fields`**: sums all field scores. Tends to over-reward documents where the same term appears in many fields (can penalise focused, high-quality descriptions).
- **`cross_fields`**: treats all fields as if they were one big field. Useful when a query's terms are split across fields (e.g. "first name" / "last name"), but not appropriate here.
- **`phrase`**: requires terms to appear in order. Too strict for travel search.

#### Field boosting

Not all fields are equally important. A query for "Marbella" should prioritise a hotel whose destination is Marbella over one whose description merely mentions it in passing. We encode this with field-level boost multipliers:

```python
_SEARCH_FIELDS = [
    "hotel_name^3",  # exact hotel name searches must win
    "destination^2",  # primary travel dimension
    "region^2",  # secondary location
    "country^2",  # tertiary location
    "hotel_description",  # semantic context, base weight
    "activities",
    "tags.text",
    "amenities",
]
```

Boost values multiply the BM25 score for that field before the `best_fields` combination. The values (3, 2, 1) are initial heuristics; Milestone 4's evaluation framework will measure whether they are well-calibrated.

#### `bool` query: `must` vs `filter`

OpenSearch's `bool` query has four clauses: `must`, `filter`, `should`, and `must_not`. The distinction between `must` and `filter` is critical:

- **`must`**: the document must match, **and the match score contributes to the BM25 relevance score**. Use for text queries where relevance matters.
- **`filter`**: the document must match, **but the filter clause does not affect the BM25 score**. Use for hard constraints (exact values, ranges) where you only care about inclusion/exclusion, not ranking contribution.

Using `filter` for hard constraints has two additional benefits:

1. **Performance**: filter results are cached as **bitsets** in memory. A bitset is a compact binary array where each bit represents one document. On repeated queries with the same filter, OpenSearch can reuse the cached bitset and skip re-evaluating the filter entirely.
2. **Separation of concerns**: a price filter of "under £1500" should never inflate the BM25 score of a hotel — the filter is a binary gate, not a relevance signal.

Our query structure:

```json
{
  "query": {
    "bool": {
      "must": [{ "multi_match": { ... } }],
      "filter": [
        { "term": { "country": "Spain" } },
        { "range": { "price_per_person_gbp": { "lte": 1500 } } }
      ]
    }
  }
}
```

#### Filters: `term` vs `range`

- **`term`**: exact match on a `keyword` field. `{ "term": { "country": "Spain" } }` matches only documents where `country` is exactly `"Spain"`.
- **`range`**: numeric range on an `integer` or `float` field. `{ "range": { "star_rating": { "gte": 4 } } }` matches documents where `star_rating ≥ 4`.

Note that `term` queries on `text` fields would match against the **analysed tokens**, not the original string — this is a common mistake. That is why all filter fields in our schema are `keyword` type.

#### Aggregations / facets

Alongside the search results, we return aggregation buckets for four dimensions: `country`, `star_rating`, `board_types`, and `climate_zone`. These are **`terms` aggregations**: OpenSearch counts how many matching documents have each distinct value:

```json
"aggs": {
    "countries": { "terms": { "field": "country", "size": 30 } }
}
```

This returns something like `[{ "key": "Spain", "doc_count": 42 }, ...]`, which the UI can render as "Spain (42)" filter chips. Crucially, aggregations respect the query and filter clauses — the counts reflect the current result set, not the whole index. This is how "guided search" / "drill-down filtering" works on e-commerce sites.

#### `fuzziness: AUTO` — typo tolerance

BM25 requires an exact token match by default. A user who types `"Malorcca"` instead of `"Mallorca"` gets zero results. `fuzziness: AUTO` tells OpenSearch to allow **Levenshtein edit distance** (insertions, deletions, substitutions, transpositions) when matching tokens:

- Terms of 1–2 characters: no fuzziness (too short for edit distance to be meaningful).
- Terms of 3–5 characters: 1 edit allowed.
- Terms ≥ 6 characters: 2 edits allowed.

This handles the most common typos without a dedicated spell-correction pipeline. The tradeoff: fuzzy matching is more expensive than exact matching, and on very large corpora it can introduce false positives.

#### `_build_query()` as a pure function

The query builder is implemented as a pure function: it takes a `LexicalSearchParams` dataclass and returns a Python `dict` that is the OpenSearch query body — no network calls, no side effects. This design choice makes the query logic **fully unit-testable without OpenSearch running**:

```python
body = _build_query(LexicalSearchParams(query="beach", country="Spain"))
assert body["query"]["bool"]["filter"] == [{"term": {"country": "Spain"}}]
```

The thin `lexical_search()` wrapper takes the dict and makes the actual network call. This separation keeps the bulk of the logic in the pure layer and keeps integration tests focused on verifying OpenSearch behaviour (does filtering actually work?), not query construction correctness.

#### FastAPI lifespan — startup and shutdown

The FastAPI application needs an OpenSearch client. Creating a new client per request is wasteful (connection overhead). The `lifespan` context manager creates the client once at startup and stores it on `app.state`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.os_client = create_client(settings)
    yield  # application runs here
    app.state.os_client.close()
```

The `yield` statement separates startup code (before) from shutdown code (after). The client is closed cleanly when the application exits. This replaced the older `@app.on_event("startup")` / `@app.on_event("shutdown")` pattern, which is deprecated in FastAPI.

#### Dependency injection with `Depends()`

Route handlers receive the OpenSearch client and settings via FastAPI's dependency injection:

```python
@router.get("/lexical")
def search_lexical(
    q: str = "",
    os_client: OpenSearch = Depends(get_os_client),
    settings: Settings = Depends(get_settings),
) -> LexicalSearchResponse: ...
```

`get_os_client()` reads `request.app.state.os_client`. This pattern means:

1. The route handler has no knowledge of how the client is created.
2. In tests, `app.dependency_overrides[get_os_client] = lambda: mock_client` substitutes any dependency — making the handler fully testable without infrastructure.
3. Adding a second handler that needs the client requires zero changes to the client lifecycle.

#### API response schemas (Pydantic v2)

`LexicalSearchResponse`, `SearchHit`, and `FacetBucket` are Pydantic v2 models used as the API response shape. Key points:

- `ConfigDict(extra="ignore")` on `SearchHit` silently drops fields from `_source` that are not declared in the schema (e.g. raw `latitude`, `longitude`, `location`). This keeps the API response clean without requiring a manual field-by-field copy.
- `FacetBucket` renames the OpenSearch `"doc_count"` key to `"count"` — the API surface is stable regardless of how OpenSearch names its aggregation fields internally.
- The `from_result()` classmethod on `LexicalSearchResponse` converts from the internal `LexicalSearchResult` dataclass (used by the retrieval layer) to the API schema. The retrieval layer stays decoupled from the API layer.

### Design decisions and alternatives considered

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| `multi_match` type | `best_fields` | `most_fields`, `cross_fields` | Best models "one good field wins" travel search pattern |
| Field boosts | `name^3`, `location^2`, rest at 1.0 | Equal weights | Location and name are primary search dimensions |
| Filter placement | `bool.filter` | `bool.must` | No score contribution; bitset caching |
| Fuzziness | `AUTO` | None, or fixed edit distance | Balanced typo tolerance without hardcoding |
| Query building | Pure function | Mixed with HTTP call | Enables extensive unit testing without infrastructure |
| Empty query handling | `match_all` | Error / reject | Supports browsing / "show me everything" mode |
| OpenSearch client lifecycle | FastAPI `lifespan` | Per-request construction | Single connection pool; clean shutdown |
| Response field projection | Pydantic `extra="ignore"` | Manual field copy | Declarative, less code, safe by default |

---

---

## Milestone 4 — Evaluation framework and BM25 baseline

### What was added

- `src/travel_ai_search/evaluation/metrics.py` — seven pure-function IR metrics implemented from scratch (no external deps)
- `src/travel_ai_search/evaluation/dataset.py` — frozen domain model: `RelevanceJudgment`, `GoldenQuery`, `GoldenDataset`; `load_dataset()` reads JSONL
- `src/travel_ai_search/evaluation/evaluator.py` — strategy-agnostic `evaluate(search_fn, dataset)` function; `EvaluationReport`, `QueryMetrics`, `ClassSummary` dataclasses
- `scripts/build_golden_dataset.py` — one-time builder that scans the hotel corpus and generates `data/evaluation/golden_queries.jsonl`
- `scripts/evaluate.py` — CLI: loads dataset, wraps any strategy as a `SearchFn`, runs `evaluate()`, prints a table, saves JSON to `data/evaluation/results/`
- `data/evaluation/golden_queries.jsonl` — 62 queries, 10 classes, 48,675 graded judgments
- `data/evaluation/results/bm25_2026-08-13.json` — BM25 baseline results
- `tests/unit/test_metrics.py` (36 tests), `tests/unit/test_evaluation_dataset.py` (20 tests), `tests/integration/test_evaluation.py` (8 tests)

### IR concepts explained

#### Graded relevance and the TREC paradigm

The classical approach to IR evaluation, standardised by TREC (Text REtrieval Conference), separates two concerns: (1) the *search system* returns a ranked list of documents; (2) human *assessors* independently judge each document against each query on a relevance scale. Neither the assessors nor the system know about each other during judgment.

This project uses **4-point graded relevance (0–3)**:

| Grade | Meaning | Example |
|---|---|---|
| 3 | Highly relevant — ideal match | Family-friendly beach hotel in the exact queried destination |
| 2 | Relevant — clearly useful | Family-friendly beach hotel in the queried country but wrong island |
| 1 | Marginally relevant — partially useful | Family-friendly inland hotel in the right country |
| 0 | Irrelevant | Adults-only city hotel in a different country |

**Unjudged-is-irrelevant assumption:** TREC evaluation treats any document not in the judgment pool as grade 0. This is correct when the pool is large enough to cover all likely top-K results, but causes underestimation of recall when the pool is incomplete. For this project, the golden dataset covers grade-3 and grade-2 docs comprehensively for each query, and grade-1 docs are sampled (capped at 40 per query to limit pool size).

#### Precision@K

The fraction of the top-K retrieved documents that are relevant (grade ≥ 1):

```
P@K = |relevant ∩ top_K| / K
```

Interpretation: of the 10 results the user sees, how many are actually useful? P@10 = 0.6 means 6 of 10 results are relevant.

**Key property:** if `|retrieved| < K`, the denominator is still `K` (TREC convention). A system that returns fewer results is penalised.

#### Recall@K

The fraction of *all* relevant documents that appear in the top-K:

```
Recall@K = |relevant ∩ top_K| / |relevant|
```

**Why Recall@K is low in this project:** the golden dataset assigns relevance to hundreds or thousands of hotels per query (e.g., "adults-only hotel" has ~1,500 grade-3 hotels). Recall@10 ≈ |top-10 relevant| / 1500 ≈ 0.7%. This is expected and valid. Recall@K is most useful when the number of relevant documents is small (e.g., entity lookup). NDCG@K is the primary metric here.

#### HitRate@K (Success@K)

Binary: 1 if at least one relevant document is in the top-K, 0 otherwise:

```
HR@K = 1 if |relevant ∩ top_K| > 0 else 0
```

The mean across queries is the fraction of queries where the user found *something* useful. HitRate@10 = 0.82 means BM25 fails completely on 18% of queries.

#### Reciprocal Rank (RR) and MRR

RR measures how quickly the system surfaces the first relevant result:

```
RR = 1 / rank_of_first_relevant_document
```

RR = 1.0 if rank 1 is relevant; 0.5 if rank 2; 0.33 if rank 3; 0 if none found. The Mean Reciprocal Rank (MRR) is the average RR across all queries. MRR = 0.68 for BM25 means the first relevant result appears, on average, at roughly rank 1.5.

#### Average Precision (AP) and MAP

AP captures the *precision-recall curve* for a single query:

```
AP = (1 / |R|) × Σ P@i  for each position i where document is relevant
```

where |R| is the total number of relevant documents. AP rewards both finding many relevant documents *and* ranking them early. The Mean Average Precision (MAP) is the mean AP across all queries. MAP is low here (0.011) for the same reason Recall@K is low — many judged relevant documents per query.

#### DCG and NDCG@K

DCG (Discounted Cumulative Gain) is the workhorse metric for graded relevance:

```
DCG@K = Σ_{i=1}^{K}  (2^rel_i − 1) / log₂(i + 1)
```

Two design choices encoded in this formula:
1. **Exponential gain** `(2^rel − 1)`: grade 3 gives gain 7; grade 2 gives gain 3; grade 1 gives gain 1. This rewards placing the *most* relevant results first, not just any relevant result.
2. **Logarithmic position discount** `1/log₂(i+1)`: rank 1 has discount 1.0; rank 2 = 0.63; rank 3 = 0.50; rank 10 = 0.29. Lower positions matter less.

**IDCG** (Ideal DCG) is the DCG of the perfect ranking (all grade-3 docs first, then grade-2, then grade-1). **NDCG@K = DCG@K / IDCG@K** normalises the score to [0, 1], making it comparable across queries with different numbers of relevant documents.

**Why NDCG@K is the primary metric:** it correctly handles graded relevance, rewards ranking *highly* relevant results highest, is insensitive to the large number of judged docs (because it normalises by the ideal), and is the standard metric in industry search evaluation (Google, Bing, Amazon all use NDCG variants).

#### Strategy-agnostic evaluator design

The evaluator is defined around a type alias:

```python
SearchFn = Callable[[str, int, dict[str, Any]], list[str]]
#                    query  top_k  filters       ranked doc IDs
```

Any retrieval strategy — BM25, vector, hybrid, with or without reranking — can be wrapped as a `SearchFn` and passed to the same `evaluate()` function. This means:
- Metric computation code is written once and reused across all future milestones.
- Comparing strategies is simply a matter of calling `evaluate()` with different `SearchFn` wrappers and comparing `EvaluationReport` objects.
- The `filters` dict is passed through from `GoldenQuery.filters`, allowing structured constraints (country, family_friendly, etc.) to be applied at the retrieval level where the strategy supports it.

#### Golden dataset construction

The golden dataset was built programmatically (not by hand) using `scripts/build_golden_dataset.py`. For each of the 62 query specifications:
1. Every hotel in the corpus is scored against 3 lambda predicates (grade3, grade2, grade1).
2. Hotels satisfying grade3 are assigned grade 3; those satisfying grade2 (but not grade3) are assigned grade 2; those satisfying grade1 (but not grade2 or grade3) are assigned grade 1.
3. Grade-1 results are capped at 40 to keep pool size manageable.

This approach is reproducible (deterministic seed), scalable (scales to any corpus size), and eliminates human annotation effort. The trade-off is that the criteria must correctly encode the semantics of each query class — four queries had to be revised after discovering that the synthetic dataset's hotel archetypes have mutually exclusive tag combinations (e.g., `watersports` never co-occurs with `family_friendly`).

**Dataset archetype discovery** (important for M5+ design): The generator creates hotels in archetype clusters with non-obvious constraints. For example:
- Hotels tagged `watersports` are never `family_friendly`
- Hotels tagged `spa` are always `adults_only` + `luxury` + `romantic`
- Hotels tagged `peaceful` always pair with `boutique` and `adults_only`
- `yoga` and `meditation` activities never coexist in the same hotel

These constraints mean semantic queries must target what the data actually contains, not what a human would naturally assume.

### Design decisions and alternatives considered

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| Primary metric | NDCG@K | Recall@K, MRR | Handles graded relevance; robust to large judged pool; industry standard |
| Graded relevance scale | 0–3 (4-point) | Binary 0/1 | Allows partial credit; aligns with TREC 4-point scale |
| Unjudged assumption | Unjudged = grade 0 | Skip unjudged | Simpler; appropriate when pool is large enough |
| Dataset construction | Programmatic lambda predicates | Human annotation | Scalable, reproducible, no annotation budget |
| Grade-1 cap | 40 per query | Uncapped | Keeps JSONL size manageable; Recall@K is not the primary metric |
| SearchFn interface | `(str, int, dict) → list[str]` | `SearchStrategy` class | Simpler; any callable works; no inheritance hierarchy |
| Evaluator granularity | Per-query + by-class + overall | Overall only | Class breakdown reveals where BM25 fails; guides M5 decisions |
| Metrics implementation | From scratch | `ranx`, `ir-measures` | Educational purpose; full understanding; no dependency |
| Results persistence | JSONL (dataset) + JSON (results) | SQLite, Parquet | Human-readable; no external database needed; easy to diff |

---

## Milestone 5 — Embeddings and vector retrieval

### What we added

- **`EmbeddingProvider` Protocol** (`embeddings/base.py`) — structural interface with `dimension`, `embed()`, `embed_batch()`.
- **`LocalEmbeddingProvider`** (`embeddings/local.py`) — wraps `all-MiniLM-L6-v2` via sentence-transformers; L2-normalised outputs, batch encoding.
- **knn_vector index mapping** — `embedding_vector` field added to `ingestion/index.py` (HNSW, lucene engine, cosinesimil, m=16, ef_construction=128).
- **`scripts/generate_embeddings.py`** — offline batch pipeline: reads hotels.jsonl, embeds in batches of 64, updates OpenSearch via bulk Update API (~400 docs/s on CPU, ~14 s for 5,470 hotels).
- **`retrieval/vector.py`** — `VectorSearchParams`, `VectorSearchResult`, pure `_build_vector_query()`, `vector_search()`.
- **`/search/vector` API endpoint** — same filter parameters as `/search/lexical`; embedding provider injected via `app.state`.
- **`vector` strategy** in `scripts/evaluate.py` — enables `make evaluate --strategy vector`.
- **Tests** — 14 embedding unit tests (mocked model), 25 vector retrieval unit tests (pure function), 20 vector search integration tests (real model + real OpenSearch).

### Concepts

#### Dense embeddings

A language model maps a variable-length text string to a fixed-size real-valued vector (384 floats for MiniLM). The key property is that **semantically similar texts are geometrically close** in vector space — no keyword overlap is required. The query "romantic getaway near the sea" and the hotel description "adults-only sanctuary overlooking the ocean" have no words in common but are close neighbours in embedding space.

#### The BM25 destination gap — why it mattered

M4 measurement revealed that BM25 scored NDCG@10 = 0.18 for `exact_destination` queries — the lowest class by far. The root cause: `destination` is stored as a `keyword` field not included in the BM25 multi-match, and regional names like "Tenerife" or "Algarve" rarely appear verbatim in hotel descriptions. BM25 cannot see them. Embedding the hotel's full text (including its destination, region, country, and description) encodes these place names as semantic concepts. A query for "Tenerife" naturally lands near hotels in the Canary Islands. The fix is geometric, not lexical.

#### Cosine similarity and the L2-normalisation trick

Cosine similarity measures the angle between two vectors regardless of magnitude:

```
cos(θ) = (A · B) / (|A| × |B|)
```

When both vectors are **L2-normalised** (`|v| = 1.0`), the denominator is always 1, so:

```
cos(θ) = A · B
```

The dot product equals cosine similarity. OpenSearch's `cosinesimil` space type does the normalisation automatically, so passing already-L2-normalised vectors from MiniLM (which normalises by default with `normalize_embeddings=True`) is safe.

#### HNSW — Hierarchical Navigable Small Worlds

Exact nearest-neighbour search in 384 dimensions over 5,470 vectors is fast, but at millions of documents it becomes prohibitively slow (O(N) per query). HNSW is an **approximate nearest-neighbour (ANN)** algorithm that builds a multi-layer proximity graph during indexing:

- **Bottom layer** — every node, densely connected to its m=16 nearest neighbours.
- **Upper layers** — progressively sparser long-range connections, allowing the search to "zoom in" on the target region quickly.

At query time, the traversal starts at the top layer (long strides), descends, and greedily refines until it reaches the ef_search=100 candidate pool in the bottom layer. Recall is ~95% (95% of the time the true nearest neighbour is in the result) with query latency in single-digit milliseconds — a fundamentally different scaling profile than exact search.

Key parameters:

| Parameter | Value | Effect |
|---|---|---|
| m | 16 | Edges per node; higher → better recall, more memory |
| ef_construction | 128 | Build-time candidate pool; higher → better graph, slower build (one-time) |
| ef_search | 100 | Query-time candidate pool; higher → better recall, slower queries |

#### Offline embedding pipeline — why separate from ingestion

Embedding 5,470 hotels takes ~14 s on a MacBook CPU. Options:

1. **Inline at ingest time** — simpler, but ingestion becomes a slow CPU-bound operation and the BM25 index is unavailable until embedding finishes.
2. **Offline script** — ingestion (BM25 index) is immediately available; embedding runs separately and can be repeated with a different model without touching ingestion.

The offline approach also mirrors production architectures where embeddings are generated asynchronously (e.g., by a worker queue) and the main ingestion pipeline is not blocked.

#### The `EmbeddingProvider` Protocol — provider abstraction

```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

Python structural subtyping (not `ABC`/`isinstance` inheritance): any class that implements these three members satisfies the protocol. This means:
- Unit tests can pass a `MagicMock()` directly — no special mock class needed.
- AWS Bedrock embedding (M12) will be a drop-in without modifying any calling code.
- The API route receives `EmbeddingProvider` from `Depends(get_embedding_provider)` — the specific implementation is hidden.

#### Efficient filter mode (lucene engine)

OpenSearch supports two approaches to filtering k-NN results:

1. **Post-filtering** — run the full ANN search, then discard documents that fail the filter. Can return fewer than k results if many candidates are filtered out.
2. **Efficient filtering** (lucene engine, OpenSearch 2.9+) — the filter is applied **during HNSW graph traversal**, pruning branches that can never produce matching documents. Returns exactly k results (or all matching if fewer than k exist).

The `nmslib` engine does not support efficient filtering; `lucene` does. The index mapping was changed from `"engine": "nmslib"` to `"engine": "lucene"` to enable the filter-inside-knn query syntax:

```json
{
    "query": {
        "knn": {
            "embedding_vector": {
                "vector": [...],
                "k": 10,
                "filter": {"bool": {"filter": [{"term": {"country": "Spain"}}]}}
            }
        }
    }
}
```

#### API integration — embedding provider as shared state

The SentenceTransformer model (~80 MB) takes 1–2 s to load. Loading it per-request would be catastrophic for latency. The model is loaded **once at startup** in the FastAPI lifespan and stored on `app.state.embedding_provider`. Route handlers receive it via `Depends(get_embedding_provider)`.

The factory pattern (`_create_embedding_provider(settings)`) separates model construction from the lifespan context so unit tests can patch just the factory without mocking the entire lifespan:

```python
@patch("travel_ai_search.api.app._create_embedding_provider", return_value=MagicMock())
def test_vector_endpoint(mock_create): ...
```

### Design decisions

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | `bge-small-en-v1.5`, `text-embedding-3-small` | Industry benchmark quality; 384d (small HNSW index); runs on CPU; no API key required |
| Vector dimension | 384 | 768, 1536 | Small enough for single-node HNSW; MiniLM produces 384d by default; tradeoff: quality vs memory |
| Space type | `cosinesimil` | `l2`, `innerproduct` | Cosine is scale-invariant; equals dot product when L2-normalised; standard for sentence encoders |
| HNSW engine | `lucene` | `nmslib`, `faiss` | Only lucene supports efficient filter mode on OpenSearch 2.15; nmslib does not support filters |
| Embedding pipeline | Offline batch script | Real-time at ingest | Decouples slow CPU embedding from fast BM25 indexing; can be re-run with a new model |
| Batch size | 64 | 32, 128 | Fits comfortably in CPU RAM; maximises throughput without memory pressure |
| Provider abstraction | Protocol (structural) | ABC (nominal) | No inheritance required; MagicMock satisfies it directly; Bedrock provider is a drop-in |
| Model load location | `app.state` at lifespan | Module-level singleton, per-request | Avoids global state; respects FastAPI's dependency injection; testable via factory patch |
| Test isolation | `autouse` fixture patches factory | Patch per test | All unit tests are automatically insulated from model load; integration tests use real model |

### Results summary (M5)

| Metric | BM25 | Vector | Δ |
|---|---|---|---|
| NDCG@10 | 0.5007 | **0.6940** | +38.6% |
| MRR | 0.6842 | **0.8688** | +27.0% |
| HitRate@10 | 0.8226 | **1.0000** | +21.6% |
| exact_destination NDCG@10 | 0.1830 | **0.8392** | +358.8% |
| Latency p50 | 24 ms | 11 ms | faster |
| Latency p95 | 45 ms | 135 ms | more variable |

The most important result: the destination gap that cost BM25 NDCG=0.0 on 5 queries is fully resolved by vector retrieval. Hybrid retrieval (M6–7) will seek to combine the precision of BM25 for exact-match queries with the recall of vector search.
