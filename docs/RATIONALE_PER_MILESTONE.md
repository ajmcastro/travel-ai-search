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

---

## Milestone 6 — Hybrid retrieval

### What we added

- `src/travel_ai_search/retrieval/types.py` — shared `Hit` dataclass; extracted from `lexical.py` to break a circular import between `fusion.py` and `lexical.py`.
- `src/travel_ai_search/retrieval/fusion.py` — new module with three components:
  - `build_filter_clauses(**kwargs)` — the single authoritative source for OpenSearch filter clause construction; previously duplicated in `lexical.py` and `vector.py`.
  - `_normalize_scores(hits) → dict[id, (norm_score, source)]` — min-max normalisation, maps a ranked list to [0, 1].
  - `fuse_results(lex_hits, vec_hits, *, lexical_weight, vector_weight, top_k) → list[Hit]` — pure function (no I/O); performs union, weighted combination, sort, and truncation.
- `src/travel_ai_search/retrieval/hybrid.py` — orchestrates both retrievers and fusion; `HybridSearchParams`, `HybridSearchResult`, `hybrid_search()`.
- `GET /search/hybrid` endpoint with per-retriever timing in the response.
- `hybrid` strategy in `scripts/evaluate.py`.
- Unit tests for all pure functions in `fusion.py`; integration tests for filters, weights, and semantic quality.

### Concepts introduced

#### The score normalisation problem

BM25 and cosine similarity live in incompatible numeric ranges. BM25 scores are corpus- and query-dependent — a BM25 score of 12.4 means "more relevant than a score of 8.2 given this corpus and this query", but it carries no absolute meaning. Cosine similarity scores are in [0, 1] (for L2-normalised vectors). If you add them directly, BM25 dominates: its larger numeric range swamps the cosine signal.

**Min-max normalisation** maps each list independently to [0, 1]:

```
norm(score) = (score − min_score) / (max_score − min_score)
```

Edge cases: empty list → empty dict; single document → 1.0; all identical scores → 1.0 (denominator = 0 → clamp). After normalisation, both lists are on the same scale and can be combined linearly.

#### Weighted-sum fusion

```
combined_score = lexical_weight × norm_bm25 + vector_weight × norm_vector
```

where `norm_x = 0.0` for documents absent from retriever x's results. Documents found by both retrievers (and ranked highly by both) will score highest; those found by only one retriever are penalised — their maximum possible score is `max_weight × 1.0`.

The weights do not need to sum to 1.0 (they are applied to already-normalised scores), but conceptually keeping them normalised makes them interpretable as a convex combination.

#### Candidate pool (`candidate_k > top_k`)

Each retriever fetches `candidate_k=50` documents. Fusion selects the best `top_k=10` from up to 100 unique candidates. The larger pool is necessary because:

- A document ranked 11th by BM25 but 1st by vector should win fusion; it would be invisible with `candidate_k=top_k=10`.
- Fusion can promote documents that one retriever ranked just outside top-k if the other retriever strongly endorses them.
- Setting `candidate_k` too high increases latency; too low misses re-ranking opportunities.

#### Missing-retriever penalty (0.0 score for absent documents)

Documents found by only one retriever receive 0.0 for the missing side. This is a conservative choice that reflects lower confidence: if the embedding space says a document is semantically close but BM25 gives it near-zero, the document may be a false positive; if BM25 ranks it high but vector doesn't, the keyword match may be coincidental. Documents found by both retrievers — both signal agreement — score highest.

An alternative: use a floor score (e.g., 0.0 penalises less than −∞). Another: treat absence as a rank-penalty rather than a score penalty (this is what RRF does, Milestone 7).

#### Circular import and the shared types module

Putting `Hit` in `lexical.py` and `build_filter_clauses` in `fusion.py` creates a cycle: `fusion.py` imported `Hit` from `lexical.py`, which imported `build_filter_clauses` from `fusion.py`. Python's module initialisation fails on partially initialised modules.

The fix: `types.py` holds the shared `Hit` dataclass. Both `lexical.py` and `fusion.py` import from `types.py`. `lexical.py` still re-exports `Hit` in its namespace for backwards compatibility (existing imports like `from travel_ai_search.retrieval.lexical import Hit` continue to work).

### Design decisions

| Decision | Choice | Alternatives considered | Rationale |
|---|---|---|---|
| Fusion strategy | Min-max normalised weighted sum | RRF, WAND | Min-max is transparent and unit-testable; illustrates the score normalisation problem concretely. RRF deferred to M7 as a contrast. |
| Client-side vs server-side hybrid | Client-side | OpenSearch native hybrid pipeline | Server-side requires index-time pipeline setup and the hybrid plugin; client-side is self-contained and fully inspectable. |
| Candidate pool | `candidate_k=50`, final `top_k=10` | `candidate_k=top_k` | Larger pool allows fusion to surface documents ranked 11–50 by one retriever when the other retriever strongly endorses them. |
| Filter propagation | Both retrievers receive the same filter set | Filter only lexical or only vector | Filters are hard constraints; they must apply to all candidates regardless of retrieval path. |
| Timing metadata | `took_ms`, `lexical_took_ms`, `vector_took_ms` in response | Total only | Per-stage timing exposes where latency is spent; useful for deciding when to parallelise the two queries. |
| Sequential vs concurrent retrieval | Sequential | `asyncio.gather` / threads | Simpler code for an educational project. In production, running both queries concurrently would reduce p50 to `max(lex, vec) + fusion ≈ 25 ms`. |

### Results summary (M6)

| Metric | BM25 | Vector | Hybrid (50/50) | Notes |
|---|---|---|---|---|
| NDCG@10 | 0.5007 | **0.6940** | 0.6003 | Hybrid regresses from vector |
| MRR | 0.6842 | **0.8688** | 0.8542 | Near-vector quality |
| HitRate@10 | 0.8226 | **1.0000** | 0.9355 | Better than BM25, below vector |
| exact_destination NDCG@10 | 0.1830 | **0.8392** | 0.4799 | Bad BM25 signals dilute vector |
| Latency p50 | 24 ms | 11 ms | 57 ms | Two sequential queries |
| Latency p95 | 45 ms | 135 ms | 90 ms | Lower tail than vector alone |

**Core learning:** Naive 50/50 weighted-sum fusion does not beat the best individual retriever when one retriever produces meaningless scores for a query class. Adding a bad signal (BM25's near-random ordering for destination queries) at 50% weight degrades results. This motivates RRF (Milestone 7), which is rank-based and robust to score magnitude.

---

## Milestone 7 — RRF and alternative fusion

### What we added

- **`FusionMethod` enum** (`retrieval/fusion.py`) — `StrEnum` with values `weighted` and `rrf`; FastAPI auto-validates it from query string parameters.
- **`rrf_fuse(ranked_lists, *, k, top_k)`** (`retrieval/fusion.py`) — pure function implementing Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009). Accepts a variadic list of ranked lists, accumulates 1/(k + rank) per document per list, sorts, and truncates. No normalisation required.
- **`_RRF_K_DEFAULT = 60`** — the empirically robust constant from the original paper.
- **`HybridSearchParams.fusion`** and **`.rrf_k`** fields — allow the caller to select fusion strategy and tune the smoothing constant without changing code.
- **`/search/hybrid?fusion=rrf&rrf_k=60`** — API support: `fusion` and `rrf_k` query parameters added to the hybrid endpoint.
- **`hybrid_fusion` and `rrf_k` settings** in `config/settings.py` — enable environment-variable control over the default fusion strategy.
- **`--strategy rrf`** in `scripts/evaluate.py` — `make_rrf_fn()` wraps hybrid search with RRF fusion; the `"rrf"` key is added to `STRATEGIES`.
- **14 new unit tests** for `rrf_fuse()` covering: empty input, single-list score formula, multi-list accumulation, top-k truncation, source from first occurrence, k-value sensitivity, and three-list correctness.
- **8 new integration tests** for the RRF endpoint: result type, sorting, and all filter types (family_friendly, adults_only, month) plus a semantic quality assertion.

### IR concepts introduced

#### The score-scale problem — why normalisation is not enough

Min-max normalisation (M6) transforms each list to [0, 1] before combination. This solves the *magnitude* problem: BM25's score of 12.4 and cosine similarity of 0.76 are now both in [0, 1]. But it does not solve the *ordering* problem: after normalisation, the relative order within each list is preserved exactly. If BM25 ranks documents in a near-random order for destination queries, normalised BM25 scores are still in a near-random order — just mapped to [0, 1].

Giving 50% weight to a near-random ordering is almost always worse than ignoring that retriever entirely. This is the precise failure mode observed in M6: `exact_destination` NDCG dropped from 0.84 (vector alone) to 0.48 (50/50 hybrid), because BM25's random ordering moved relevant documents down in the fused ranking.

#### Reciprocal Rank Fusion — rank positions, not score values

RRF (Cormack et al., 2009) avoids score values entirely. For every document *d* in the union of all ranked lists *r*:

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```

where `rank_r(d)` is the 1-indexed position of *d* in list *r* (documents absent from a list contribute 0). The key properties:

1. **Score-scale invariance.** A document ranked 1st by BM25 contributes 1/(k+1) ≈ 0.016 regardless of whether its BM25 score is 12.4 or 0.001. The score value is discarded; only position matters.

2. **Bounded contribution.** Even the top-ranked document in a bad retriever contributes at most 1/(k+1). If BM25 ranks a document 1st by chance (destination query), that contributes 0.016 — not enough to promote it over a document ranked 1st by the vector retriever (also 0.016) unless BM25 also ranks the true answer highly.

3. **Natural upweighting of agreement.** A document ranked 1st by *both* retrievers scores 2/(k+1) ≈ 0.033, exactly double a document ranked 1st by only one. This reward for inter-retriever agreement is the core of RRF's success.

4. **No tuning of per-retriever weights.** Weighted-sum requires choosing `lexical_weight` and `vector_weight`; the optimal weights vary by query class. RRF has one shared parameter (`k`) that controls the contribution curve globally.

#### The role of k (smoothing constant)

The constant `k` controls the shape of the contribution curve:

```
rank 1:   1/(k+1)
rank 2:   1/(k+2)
rank 5:   1/(k+5)
rank 50:  1/(k+50)
```

With `k=60` (the Cormack et al. default):
- Rank 1 → 0.016
- Rank 10 → 0.014  (rank-1 is only 1.14× rank-10)
- Rank 50 → 0.009  (rank-1 is only 1.83× rank-50)

The curve is **flat**: top-ranked documents get a modest bonus over mid-ranked documents. This reflects the intuition that we don't fully trust either retriever's exact ordering — we only know that rank 1 is better than rank 50.

With a smaller `k` (e.g., `k=10`):
- Rank 1 → 0.091
- Rank 10 → 0.050  (rank-1 is 1.82× rank-10)
- Rank 50 → 0.017  (rank-1 is 5.4× rank-50)

The curve is **steeper**: top-ranked documents dominate. Smaller `k` is useful when you trust the retrievers' top rankings more and want to amplify their agreement.

`k=60` is a robust default that works across diverse retrieval systems without dataset-specific tuning. The `rrf_k` configuration field exposes this for experimentation.

#### Extensibility to N lists — multi-query preview

`rrf_fuse` accepts `ranked_lists: list[list[Hit]]` — a list of any number of ranked lists. Two lists (BM25 + vector) is the M7 use case. Three or more lists are the Milestone 11 (multi-query) use case: generate N reformulations of the original query, run each through BM25 and/or vector retrieval, and fuse all N result lists with a single `rrf_fuse` call. The interface requires no change:

```python
# M7 (two lists)
rrf_fuse([lex_hits, vec_hits], k=60, top_k=10)

# M11 (four lists — original + 3 query expansions, all through vector)
rrf_fuse([vec_hits_q0, vec_hits_q1, vec_hits_q2, vec_hits_q3], k=60, top_k=10)
```

This was a deliberate design choice: making `rrf_fuse` accept N lists now costs almost nothing and avoids a breaking interface change at M11.

#### `StrEnum` vs `(str, Enum)`

Python 3.11 introduced `enum.StrEnum` as a direct base class for string enumerations. Prior to 3.11, the idiom was `class MyEnum(str, Enum)`. `StrEnum` is preferred in Python 3.12 (this project's target) because:

- It conveys intent more clearly — the reader sees immediately that values are strings.
- Ruff's `UP042` rule flags `(str, Enum)` as a style issue.
- `StrEnum` members compare equal to their string values: `FusionMethod.rrf == "rrf"` is `True`, enabling direct use in FastAPI query parameter parsing without an explicit `FusionMethod(value)` conversion.

### Design decisions

| Decision | Choice | Alternatives considered | Rationale |
|---|---|---|---|
| Fusion dispatch | `FusionMethod` enum field on `HybridSearchParams` | Separate `rrf_search()` function | Single `hybrid_search()` entry point; no code duplication across filter handling, timing, or result construction. |
| Default `k` | 60 | 10, 20, 100 | Empirically robust across document types; matches the Cormack et al. paper; configurable for experiments. |
| Ignored weights | `lexical_weight` and `vector_weight` are stored but not used when `fusion=rrf` | Remove from params | Keeping them avoids a breaking change if callers switch between fusion methods; `hybrid.py` documents clearly that they are ignored. |
| `rrf_fuse` signature | `ranked_lists: list[list[Hit]]` (variadic) | Two separate `lex_hits`/`vec_hits` params | Naturally extensible to N lists (M11 multi-query); more general without extra complexity. |
| Source-ownership rule | First occurrence wins | Last occurrence, or merge | The first retriever to return a document is assumed to have the most authoritative source data; avoids arbitrary choice when both have it. |
| `StrEnum` | Yes (`from enum import StrEnum`) | `(str, Enum)` | Pythonic for 3.12; ruff UP042; direct string comparison. |
| Config field `hybrid_fusion` | `str` with `FusionMethod(settings.hybrid_fusion)` cast in route | `FusionMethod` directly | `pydantic-settings` parses env vars as strings; the cast validates and gives a clear error at startup if the value is invalid. |

### Results summary (M7)

| Metric | BM25 | Vector | Hybrid (50/50) | Hybrid (RRF) | Δ RRF vs weighted |
|---|---|---|---|---|---|
| NDCG@10 | 0.5007 | **0.6940** | 0.6003 | 0.6239 | +0.0236 (+3.9%) |
| MRR | 0.6842 | **0.8688** | **0.8542** | 0.8449 | −0.0093 (−1.1%) |
| HitRate@10 | 0.8226 | **1.0000** | 0.9355 | 0.9516 | +0.0161 (+1.7%) |
| exact_destination NDCG@10 | 0.1830 | **0.8392** | 0.4799 | 0.5347 | +0.0548 (+11.4%) |
| activities NDCG@10 | 0.2808 | 0.3837 | 0.3256 | **0.4002** | +0.0746 (+22.9%) |
| Latency p50 | **24 ms** | 11 ms | 57 ms | 56 ms | −1 ms |

**Core learning:** RRF is a robust upgrade over weighted-sum (+3.9% NDCG, +5.7% Precision) with essentially no latency cost. Its key advantage — score-scale invariance — substantially mitigates the M6 regression on `exact_destination` (0.48 → 0.53) and produces the only class where a fusion method beats both individual retrievers (`activities`, 0.40 vs vector 0.38). Its limitation: a retriever with a truly random ordering still contributes 1/(k+rank) per document — RRF reduces but does not eliminate the noise. Pure vector (NDCG=0.694) still leads overall. The architecture is ready for a cross-encoder reranker (M8) as a second pass over the top-50 RRF candidates, which should push quality past the retrieval ceiling.

---

## Milestone 8 — Cross-encoder reranking

### What we added

- **`reranking/base.py`** — `Reranker` Protocol (structural typing, `@runtime_checkable`).
- **`reranking/local.py`** — `LocalCrossEncoderReranker` wrapping `sentence_transformers.CrossEncoder`; `_build_reranking_text()` constructs the (query, document) input text from hotel fields.
- **`retrieval/hybrid.py`** — extended with `rerank_k` parameter in `HybridSearchParams` and optional `reranker` argument to `hybrid_search()`; graceful degradation if reranker raises.
- **`api/app.py`** — `_create_reranker()` factory; `reranker` stored on `app.state` at startup.
- **`api/deps.py`** — `get_reranker()` dependency.
- **`api/routes/search.py`** — `rerank=true` and `rerank_k` query parameters on `/search/hybrid`.
- **`api/schemas/search.py`** — `reranking_took_ms` field on `HybridSearchResponse`.
- **`config/settings.py`** — `reranking_enabled`, `reranker_model_name`, `rerank_k`.
- **`scripts/evaluate.py`** — `rerank` strategy with `make_rerank_fn()`.
- Tests: 20 unit tests (`tests/unit/test_reranking.py`), 10 integration tests (`tests/integration/test_reranking.py`).

### IR concepts

#### Bi-encoder vs cross-encoder

**Bi-encoder** (what BM25 and vector search use conceptually):
- Query and document are encoded **independently**. BM25 computes a term-frequency score per document at index time; a dense bi-encoder converts each document to a vector offline.
- At query time, matching is cheap: a BM25 score lookup or a dot-product between the query vector and each document vector.
- **Limitation:** the relevance model never sees the query and document *together*. It cannot capture interactions — "this word in the query is important because of that phrase in the document."

**Cross-encoder**:
- Query and document are concatenated and fed jointly into the model: `[CLS] query [SEP] document [SEP]`.
- Every attention head can compare any query token against any document token in the same forward pass — it can learn fine-grained interaction patterns.
- **Limitation:** no pre-computation. Scoring N documents requires N forward passes. For 5,470 hotels at ~20 ms/pair = 109 seconds — unusable for real-time search.

#### Two-stage retrieval (retrieve and rerank)

The standard production architecture solves the bi-encoder/cross-encoder tradeoff:

1. **Stage 1 — fast retrieval (bi-encoder):** Retrieve the top `rerank_k` candidates (e.g. 50) from the full corpus using BM25 + vector + RRF. This is fast (O(log N) for BM25, O(HNSW) for vector, O(rerank_k) for RRF).
2. **Stage 2 — precise reranking (cross-encoder):** Score only the top `rerank_k` candidates with the cross-encoder. Return the top `top_k` re-ordered results.

Latency: 50 pairs × ~20 ms each × batched = ~55 ms extra. Total: ~113 ms vs ~56 ms for RRF alone. Acceptable for interactive search.

The cross-encoder is only as good as the candidate pool it receives — if the relevant document is not in the top-50 from Stage 1, the cross-encoder cannot recover it. This is why improving Stage 1 recall (`candidate_k`, `rerank_k`) matters.

#### Why `cross-encoder/ms-marco-MiniLM-L-6-v2`

- **MS-MARCO training:** Trained on 8.8 M (query, passage) pairs from Bing web search. Strong generalisation to new domains.
- **MiniLM-L-6:** 6-layer BERT-like encoder, ~22 M parameters. 6 layers vs BERT-base's 12 → approximately 2× faster inference with modest quality loss (L-12 would score ~+2–3% NDCG).
- **Raw logit output:** The model outputs unnormalised logits (typically −10 to +10). Larger = more relevant. Scores are comparable within one query (for reranking) but not across queries.
- **Alternative — bigger model (`cross-encoder/ms-marco-MiniLM-L-12-v2`):** ~2× slower but higher quality. Worth evaluating if latency budget allows.

#### `_build_reranking_text()` design

The cross-encoder reads the hotel document as free text. The function constructs:
```
hotel_name | destination, country | hotel_description | Activities: a, b, c | Tags: t1, t2
```

Design decisions:
- **`|` separator:** clearly delimits fields for a model that reads the full sequence.
- **Activities and tags explicitly labeled:** "Activities: spa, golf" is more informative than "spa, golf" — the label signals these are structured attributes, not part of the description prose.
- **Omit numeric/geographic fields:** price, star rating, lat/lon are not text the model was trained to interpret for relevance. Including them adds noise.
- **Missing fields → empty string, skipped:** `_build_reranking_text({})` returns an empty string rather than raising. This prevents a bad document from crashing the entire reranking call.

#### `Reranker` Protocol

```python
@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], *, top_k: int) -> list[Hit]: ...
```

Structural typing (PEP 544): any object with a `rerank` method satisfying this signature is a `Reranker` — no inheritance required. This is the same pattern as `EmbeddingProvider`. Benefits:

- **Testability:** `MagicMock()` satisfies the Protocol for unit tests without loading the model.
- **Extensibility:** A `BedrockReranker` (M12) or a `DummyReranker` can be swapped in by returning any compliant object.
- **`@runtime_checkable`:** `isinstance(obj, Reranker)` works at runtime, enabling the API to check if a reranker was actually loaded.

#### Graceful degradation

If the model fails to load at startup: `_create_reranker()` catches the exception, logs a warning, and returns `None`. The API continues to serve without reranking. If the reranker raises at query time (OOM, GPU error): `hybrid_search()` catches the exception, logs a warning, and falls back to the truncated RRF results. No request fails because of a reranker error.

### Design decisions

| Decision | Chosen | Alternatives | Reason |
|---|---|---|---|
| Model | `ms-marco-MiniLM-L-6-v2` | L-2-v2 (faster), L-12-v2 (better) | Best latency-quality balance for CPU; ~86 MB download |
| `rerank_k=50` | 50 candidates passed to cross-encoder | 20 (faster), 100 (better recall) | Covers most NDCG@10 gains; latency stays < 150 ms p95 |
| Reranking text format | `field1 | field2 | …` | JSON, XML, raw description only | Pipe-delimited plaintext is the convention for cross-encoder inputs; keeps text short |
| Feature flag `reranking_enabled` | Boolean, defaults to `False` | Always-on | Model download is ~86 MB; opt-in prevents startup surprise |
| `reranking_took_ms` in response | Yes | No (keep response lean) | Observability: clients can see the reranking overhead and decide whether to enable it |
| `reranker` arg to `hybrid_search()` | Optional (`None`) | Separate `hybrid_rerank_search()` | Single function; `None` → skip reranking, keeps call sites clean |

### Results summary (M8)

| Metric | RRF | Rerank (RRF + CE) | Δ vs RRF |
|---|---|---|---|
| NDCG@10 | 0.6239 | **0.6830** | +0.0591 (+9.5%) |
| MRR | **0.8449** | 0.8191 | −0.0258 (−3.1%) |
| HitRate@10 | **0.9516** | **0.9516** | 0.0000 |
| Precision@10 | 0.7210 | **0.7935** | +0.0725 (+10.1%) |
| exact_destination NDCG@10 | 0.5347 | **0.7934** | +0.2587 (+48%) |
| activities NDCG@10 | 0.4002 | **0.5726** | +0.1724 (+43%) |
| nightlife NDCG@10 | 0.5053 | **0.6389** | +0.1336 (+26%) |
| family NDCG@10 | **0.7352** | 0.6507 | −0.0845 (−11%) |
| Latency p50 | **56 ms** | 113 ms | +57 ms |

**Core learning:** Cross-encoder reranking is the highest-NDCG strategy overall, surpassing pure vector (0.694) for the first time. The +9.5% NDCG improvement over RRF comes from the cross-encoder's ability to attend jointly to query tokens and document tokens — it can see that "Tenerife" in the query matches "destination: Tenerife" in the document, which neither BM25 nor vector can do independently. The cost is ~2× latency (113 ms vs 56 ms p50) and a small MRR regression on structured-constraint classes where the model's general-purpose relevance judgment has no advantage over the already-filtered pool. This two-stage architecture (fast retrieval → powerful reranking) is the standard production pattern for neural IR systems.

---

## Milestone 9 — Query understanding and structured constraints

### What was added

- **`query_understanding/` module**: domain model (`QueryUnderstanding`), `QueryUnderstandingEngine` Protocol, and `RuleBasedQueryUnderstandingEngine`.
- **`POST /search`**: full orchestrated pipeline endpoint — QU → hybrid RRF → optional reranking.
- **`POST /query/understand`**: standalone endpoint to inspect what the QU engine extracts from a query.
- **`understand` evaluation strategy**: treats query_text as the only input; ignores ground-truth filters; uses QU to recover constraints.
- Settings: `query_understanding_enabled: bool = True`.

### IR concepts introduced

#### Hard constraints vs soft preferences

Every travel query has two kinds of requirements:
- **Hard constraints** must be satisfied. A hotel that departs from the wrong airport or is outside the price budget is not acceptable regardless of how good the description sounds. These become OpenSearch `filter` clauses.
- **Soft preferences** should influence ranking but are not dealbreakers. "Beach", "spa", "pool" remain in the semantic query so that BM25 and vector search can score them naturally.

The distinction is crucial: treating a soft preference as a hard filter (e.g., filtering only to hotels with "beach" as a keyword) would incorrectly exclude hotels that describe themselves as "beachfront" or "seafront". The right place to enforce soft preferences is in the ranking step, not the filter step.

#### Semantic query distillation

After extracting hard constraints, the residual text (semantic_query) is used for BM25/vector retrieval. For "family beach hotel Greece July Manchester":
- Extracted: `family_friendly=True`, `country=Greece`, `month=July`, `departure_airport=MAN`
- Semantic query: "beach hotel"

Without distillation, BM25 would try to score "family beach hotel Greece July Manchester" against every hotel description. The terms "Greece", "July", and "Manchester" would produce BM25 term-frequency signal in documents that mention those words, conflating the constraint signal with the semantic signal. With distillation, only the semantically meaningful residue reaches BM25 and vector search.

#### Named Entity Recognition (NER) for structured search

NER is the task of identifying and classifying named entities (persons, locations, organizations, dates, quantities) in text. This extractor performs domain-specific NER:
- **Geographic entities**: "Manchester" → `departure_airport=MAN`; "Spain" → `country=Spain`; "Tenerife" (island) → `country=Spain` via region-to-country mapping
- **Temporal entities**: "October" → `month=October`
- **Numeric entities**: "under £2000" → `max_price=2000`; "5 star" → `min_star_rating=5`

The key design challenge for geography: the dataset stores destinations at city level ("Playa de las Américas"), but users refer to the island ("Tenerife"). Mapping region→country (not region→destination) avoids false filter precision: "hotels in Tenerife" correctly restricts to Spain without requiring an exact destination match.

#### Rule-based vs LLM-based extraction

| Approach | Speed | Cost | Generalisation | Brittleness |
|---|---|---|---|---|
| Rule-based (M9) | Sub-ms | Free | Poor for novel phrasings | High (regex edge cases) |
| LLM rewriting (M10) | 200–500 ms | Per-call cost | Excellent | Low (understands intent) |
| Bedrock LLM (M12) | 200–500 ms | AWS cost | Excellent | Low |

Rule-based extraction is the right starting point: it is free, deterministic, and fast. It works well for structured queries ("family beach hotel Greece July Manchester") and fails on ambiguous phrasing ("4-star value for money" → should NOT extract min_star_rating=4). LLM-based engines address this by understanding intent, not just pattern matching.

#### Separation of understanding from retrieval

`QueryUnderstandingEngine` is a pure function (query → `QueryUnderstanding`) with no knowledge of the retrieval layer. This separation enables:
- Testing the understanding engine in isolation (pure Python unit tests, no OpenSearch needed)
- Swapping the engine (rule-based → LLM → Bedrock) without changing routes or retrieval code
- Exposing the understanding for observability (the `POST /search` response includes the full `QueryUnderstanding` object, so callers can see exactly what was extracted)
- Evaluating understanding quality independently (by comparing extracted constraints to golden annotations)

### Key design decisions

| Decision | Chosen | Alternatives | Reason |
|---|---|---|---|
| Region → country (not destination) | Map island/region to country only | Map to destination city | Dataset destinations are city-level; users don't use city names for islands |
| No destination extraction | Destination always None | Maintain a city→destination lookup | City names overlap with many common words; false-positive risk high |
| Soft preferences NOT removed from semantic | Kept in semantic_query | Remove all detected features | BM25/vector naturally boost hotels matching "spa", "pool", etc. |
| Semantic query fallback | `semantic or original_query` | Raise error on empty | Graceful: if all tokens consumed, original query is better than empty string |
| `query_understanding_enabled: bool` | True by default | Opt-in | Rule-based engine is free; no reason to disable by default |
| POST /search uses only QU for filters | No manual override in request body | Allow filter override | Forces the NL interface; GET /search/hybrid provides explicit filters |

### Results summary (M9)

| Metric | RRF | Understand | Δ vs RRF |
|---|---|---|---|
| NDCG@10 | 0.6239 | **0.6312** | +0.0073 (+1.2%) |
| MRR | 0.8449 | **0.8620** | +0.0171 (+2.0%) |
| HitRate@10 | **0.9516** | 0.9355 | −0.0161 (−1.7%) |
| Precision@10 | 0.7210 | **0.7290** | +0.0080 (+1.1%) |
| adults_couples NDCG@10 | 0.8928 | **0.9721** | +0.0793 (+8.9%) |
| multi_constraint NDCG@10 | 0.7473 | **0.7627** | +0.0154 (+2.1%) |
| budget NDCG@10 | **0.4221** | 0.3445 | −0.0776 (−18.4%) |
| family NDCG@10 | **0.7352** | 0.6900 | −0.0452 (−6.1%) |
| Latency p50 | 56 ms | **45 ms** | −11 ms (−20%) |

**Core learning:** Rule-based QU improves NDCG and MRR over raw RRF while also being 20% faster (semantic distillation shortens the BM25 query). The main failure mode is false-positive constraint extraction — applying a hard filter when the user expressed a preference. The `adults_couples` class shows the best-case scenario: a clear boolean intent ("adults only") maps cleanly to a structured filter. The `budget` class shows the worst case: ambiguous phrasing ("4-star value") causes an incorrect hard filter. The protocol abstraction (`QueryUnderstandingEngine`) ensures M10's LLM-based engine can be substituted without changing any routes or retrieval code.
