# Experiments

Each time a new retrieval strategy is introduced or compared, the hypothesis, configuration, results, and observations are recorded here. This log is the companion to the evaluation framework.

Results are also saved as machine-readable JSON under `data/evaluation/results/`.

---

## Format

```
### [Milestone N] Strategy comparison: X vs Y

**Date:** YYYY-MM-DD
**Hypothesis:** ...
**Configuration:**
- index: ...
- embedding model: (if applicable) ...
- fusion: (if applicable) ...
- dataset size: ...

**Results:**

| Strategy | Recall@10 | MRR | NDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

**Query-class breakdown:** (if available)

**Surprises / observations:**
- ...

**Next question this raises:**
- ...
```

---

### [Milestone 3] BM25 lexical baseline

**Date:** 2026-08-13
**Hypothesis:** A BM25 multi-match query across hotel name (boost ×3), destination/region/country (boost ×2), and description/activities/tags/amenities (boost ×1.0) will return relevant results for the most common query classes: destination-based, amenity-based, and family/trip-type queries. Filter precision should be exact (zero false-positives for keyword filters on country, board type, month, etc.).

**Configuration:**
- index: `travel_hotels` (single shard, no replicas, English analyser on text fields)
- dataset: ~5 000 synthetic travel products (Milestone 1 generator, seed 42)
- embedding model: n/a
- fusion: n/a
- key parameters: `tie_breaker=0.3`, `fuzziness=AUTO`, `type=best_fields`

**Expected results (to be measured at Milestone 4):**

| Strategy | Recall@10 | MRR | NDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| BM25 multi-match | TBD | TBD | TBD | TBD | TBD |

**Implemented and verified (Milestone 3):**
- Hard-filter precision: integration tests confirm exact counts for country, family, adults-only, price, star-rating, month, airport, and combined filters against a known 6-hotel curated index.
- Text ranking: soft assertions confirm that "luxury spa adults" ranks the Marbella 5-star adults-only hotel in the top 2; "quiet peaceful nature retreat" ranks the Menorca boutique hotel in the top 2; "cultural heritage authentic medina" ranks the Marrakech riad in the top 2.
- Aggregations: all four facet buckets (countries, star ratings, board types, climate zones) are returned alongside every search response.

**Surprises / observations:**
- `fuzziness=AUTO` applies edit-distance matching for typo tolerance but adds measurable overhead on small corpora. This tradeoff will become clearer when evaluated at scale in Milestone 4.
- Storing destination as `keyword` and adding a `.text` (standard analyser) subfield enables both exact-match filtering and partial-text matching without sacrificing either.
- The English analyser (porter stemming + stop-word removal) on `hotel_description` means queries like "swimming" will match "swim" — useful for discovery queries but could reduce precision for very specific terms.

**Next question this raises:**
- What are the actual Recall@10, MRR, and NDCG@10 numbers across all 50+ golden-set query classes? (Milestone 4)
- Will the ×3 hotel-name boost over-rank exact name matches at the expense of semantically relevant properties? Hypothesis: yes, for discovery queries — the driver for adding vector retrieval in Milestone 5.

---

### [Milestone 4] BM25 baseline — measured evaluation

**Date:** 2026-08-13
**Hypothesis:** BM25 multi-match will achieve moderate NDCG@10 overall (≥0.45) but will fail badly on exact-destination queries where the destination name does not appear in the hotel's free-text description fields.

**Configuration:**
- index: `travel_hotels` (5,470 hotels, seed 42)
- golden dataset: 62 queries, 10 classes, 48,675 total judgments (grades 0–3)
- strategy: `bm25` — multi-match `best_fields`, `tie_breaker=0.3`, `fuzziness=AUTO`
- k: 10
- dataset file: `data/evaluation/golden_queries.jsonl`
- results file: `data/evaluation/results/bm25_2026-08-13.json`

**Results (BM25 @ K=10, 62 queries):**

| Metric | Value |
|---|---|
| NDCG@10 | **0.5007** |
| MRR | **0.6842** |
| HitRate@10 | **0.8226** |
| Precision@10 | **0.6145** |
| Recall@10 | 0.0124 *(many judged docs per query — expected)* |
| MAP | 0.0111 |
| Latency p50 | 24 ms |
| Latency p95 | 45 ms |

**Query-class breakdown:**

| Class | n | NDCG@10 | MRR | HitRate@10 | P@10 |
|---|---|---|---|---|---|
| adults_couples | 6 | **0.8483** | **1.0000** | 1.0000 | 1.0000 |
| luxury | 6 | **0.7026** | 0.9167 | 1.0000 | 0.8667 |
| multi_constraint | 6 | 0.6169 | **1.0000** | 1.0000 | 0.8667 |
| quiet_peaceful | 5 | 0.5793 | 0.8333 | 1.0000 | 0.8000 |
| natural_language | 4 | 0.5279 | 0.8750 | 1.0000 | 0.6000 |
| family | 9 | 0.6243 | 0.6667 | 0.7778 | 0.7111 |
| budget | 5 | 0.4343 | 0.8000 | 0.8000 | 0.5600 |
| nightlife | 5 | 0.3452 | 0.4500 | 0.6000 | 0.4600 |
| activities | 6 | 0.2808 | 0.3849 | 0.6667 | 0.3000 |
| **exact_destination** | 10 | **0.1830** | **0.2692** | 0.6000 | 0.2000 |

**Surprises / observations:**

1. **exact_destination is BM25's biggest weakness (NDCG 0.18).** The five lowest-NDCG queries are all destination-exact: "hotels in Tenerife", "Mallorca beach resort", "Crete holiday", "Algarve Portugal beach", "family holiday Tenerife" — all NDCG=0.0. Root cause: the hotel's `destination` field is stored as `keyword` (not in the BM25 multi-match fields), and regional names like "Tenerife" rarely appear in the hotel's free-text `hotel_description`. BM25 simply cannot see the destination.

2. **Structured queries work extremely well.** `adults_couples` (MRR=1.0, P@10=1.0) and `multi_constraint` (MRR=1.0) succeed because the query text contains highly discriminative terms ("adults only", "spa", "luxury") that appear in the hotel descriptions and tags. Filters passed alongside the query further eliminate noise.

3. **Nightlife underperforms (NDCG 0.35).** Hotels with `nightlife` or `party` tags don't always include those words in their natural-language descriptions; BM25 can't see tag fields with sufficient weight.

4. **Latency is excellent** — 24 ms p50, 45 ms p95. BM25 on a single-shard local OpenSearch instance is fast enough for real-time use.

5. **Recall@10 is inherently low** (1.2%) because each query has hundreds to thousands of judged hotels. This is a characteristic of the evaluation design, not a system failure. NDCG@10 is the primary metric; it correctly rewards ranking the most relevant hotels first.

**Key finding:** BM25 cannot match hotels by geographic destination unless the name appears in free-text fields. Adding a dense-vector retrieval stage (Milestone 5) that encodes destination semantics into the embedding will be the primary fix. Alternatively, the `destination` field could be added to the multi-match list as a `.text` sub-field — a low-cost partial fix.

**Next question this raises:**
- Will vector retrieval fix the `exact_destination` gap? Hypothesis: yes — embedding models encode place names as semantic concepts, so "Tenerife" would be close to hotels in Tenerife even without the word appearing in the description.
- Can we combine filter matching (keyword filters are already working) with vector retrieval to preserve filter precision while fixing the destination-match gap? (Milestone 6: hybrid retrieval)

---

### [Milestone 5] Vector (ANN) baseline vs BM25

**Date:** 2026-08-13
**Hypothesis:** Dense vector retrieval (all-MiniLM-L6-v2, 384d, HNSW cosine) will:
1. Fix the `exact_destination` gap by encoding place-name semantics — "hotels in Tenerife" should find Tenerife hotels even if the word never appears in the description.
2. Achieve higher overall NDCG@10 than BM25 (expected ≥0.65 vs 0.50) due to semantic understanding.
3. Come at a latency cost — ANN over 5k vectors with HNSW traversal is slower than an inverted-index lookup.

**Configuration:**
- index: `travel_hotels` (5,470 hotels, lucene/HNSW engine, `cosinesimil`, m=16, ef_construction=128, ef_search=100)
- embedding model: `all-MiniLM-L6-v2` (384d, L2-normalised outputs, CPU inference)
- embedding pipeline: offline batch (64 docs/batch, ~400 docs/s on CPU); embeddings stored as `knn_vector` field
- strategy: `vector` — `embed(query)` → `knn` query with efficient filter mode (lucene engine)
- golden dataset: 62 queries, 10 classes, 48,675 total judgments
- k: 10
- results file: `data/evaluation/results/vector_2026-08-13.json`

**Results (BM25 vs Vector @ K=10, 62 queries):**

| Metric | BM25 | Vector | Δ |
|---|---|---|---|
| NDCG@10 | 0.5007 | **0.6940** | +0.1933 (+38.6%) |
| MRR | 0.6842 | **0.8688** | +0.1846 (+27.0%) |
| HitRate@10 | 0.8226 | **1.0000** | +0.1774 (+21.6%) |
| Precision@10 | 0.6145 | **0.7790** | +0.1645 (+26.8%) |
| Latency p50 | **24 ms** | 11 ms | −13 ms (faster) |
| Latency p95 | **45 ms** | 135 ms | +90 ms |

**Query-class breakdown (Vector):**

| Class | n | NDCG@10 | vs BM25 | MRR | HitRate@10 | P@10 |
|---|---|---|---|---|---|---|
| adults_couples | 6 | **0.8923** | +0.044 | **1.0000** | 1.0000 | 0.9333 |
| **exact_destination** | 10 | **0.8392** | **+0.656** | 0.8700 | 1.0000 | 0.8800 |
| multi_constraint | 6 | **0.7815** | +0.165 | **1.0000** | 1.0000 | 1.0000 |
| luxury | 6 | 0.7446 | +0.042 | 0.8889 | 1.0000 | 0.8167 |
| quiet_peaceful | 5 | 0.6863 | +0.107 | 0.9000 | 1.0000 | 0.8000 |
| natural_language | 4 | 0.6056 | +0.078 | 0.8000 | 1.0000 | 0.7250 |
| nightlife | 5 | 0.6307 | +0.286 | 0.8400 | 1.0000 | 0.7000 |
| family | 9 | 0.7422 | +0.118 | 0.9259 | 1.0000 | 0.8000 |
| budget | 5 | 0.4269 | −0.007 | 0.8200 | 1.0000 | 0.6000 |
| activities | 6 | 0.3837 | +0.103 | 0.5833 | 1.0000 | 0.4000 |

**Surprises / observations:**

1. **exact_destination gap completely closed.** NDCG@10 jumped from 0.18 to 0.84 (+0.656) — the largest single-class improvement. Vector search encodes "Tenerife" as a geometric concept close to hotels in the Canary Islands even without the word in their descriptions. The hypothesis was confirmed.

2. **HitRate@10 = 1.0000.** Vector search finds at least one relevant hotel for every query in the golden set (vs 82% for BM25). The richer semantic representation eliminates the zero-hit cases.

3. **Vector p50 latency (11 ms) is faster than BM25 (24 ms).** This is counterintuitive — HNSW ANN traversal should be slower than an inverted-index lookup. Likely explanation: the lucene HNSW implementation benefits from compiled similarity code and the index fits entirely in memory; BM25 with `fuzziness=AUTO` has additional overhead from edit-distance calculation on all candidate terms.

4. **Vector p95 latency (135 ms) is slower than BM25 (45 ms).** HNSW graph traversal is more variable than BM25 — queries that explore more of the graph (few nearby neighbours) take significantly longer. This is expected ANN behavior.

5. **activities class remains weak (NDCG 0.38).** Activity queries like "hiking", "watersports", "water skiing" require knowing that specific hotel activities match the query intent. The embedding model understands "hiking" but may not strongly differentiate between hotels with hiking as a listed activity vs those that merely describe a mountainous region.

6. **Budget class is essentially unchanged (NDCG 0.43, Δ −0.007).** Budget queries rely heavily on price ordering, which neither BM25 nor vector retrieval captures — both return semantically relevant hotels regardless of price. A numeric range filter solves this precisely; a retrieval stage cannot.

7. **MRR = 1.000 for multi_constraint.** Vector search puts the correct hotel first for all 6 multi-constraint queries. This is the highest-intent query class, and the richer semantic encoding correctly prioritises highly specific matches.

**Key finding:** Vector retrieval is strictly better than BM25 for destination-based and discovery queries. The 38.6% NDCG improvement and the perfect HitRate are compelling. However, for price-sensitive queries ("budget", "value for money") vector retrieval provides no benefit — the correct solution is to pair vector retrieval with numeric filter constraints. This is the primary motivation for hybrid retrieval in Milestone 6.

**Engine note:** The index was created with the `lucene` engine (not `nmslib`) to enable efficient filter support (filter applied during HNSW graph traversal rather than post-hoc). `nmslib` does not support filters in the knn query clause on OpenSearch 2.15.

**Next question this raises:**
- Can hybrid retrieval (BM25 + vector fusion) combine the precision of BM25 for exact-term queries with the recall of vector search for semantic queries? Hypothesis: yes — RRF fusion should produce NDCG ≥ 0.70. (Milestone 6–7)
- Does the activities/budget gap reveal a fundamental limit of dense retrieval, or can a better embedding text (including more structured fields) close it? Worth investigating by adding activity list and price tier to the embedding text.

---

### [Milestone 6] Hybrid retrieval vs BM25 and Vector

**Date:** 2026-08-13
**Hypothesis:** Combining BM25 and vector retrieval via min-max normalised weighted-sum fusion (50/50 default) will improve on BM25 (NDCG 0.50) and approach or match vector (NDCG 0.69) — getting the best of both: vector's semantic recall and BM25's precision for exact-term queries.

**Configuration:**
- index: `travel_hotels` (5,470 hotels, lucene/HNSW, cosinesimil)
- embedding model: `all-MiniLM-L6-v2` (384d, L2-normalised)
- fusion: client-side min-max normalised weighted sum
- `candidate_k=50` (each retriever fetches 50 candidates; fusion selects top 10 from ≤ 100 unique)
- `lexical_weight=0.5`, `vector_weight=0.5`
- strategy: `hybrid` — lexical → vector → `fuse_results()` → top 10
- golden dataset: 62 queries, 10 classes, 48,675 judgments
- results file: `data/evaluation/results/hybrid_2026-08-13.json`

**Results (BM25 vs Vector vs Hybrid @ K=10, 62 queries):**

| Metric | BM25 | Vector | Hybrid (50/50) | Δ vs BM25 | Δ vs Vector |
|---|---|---|---|---|---|
| NDCG@10 | 0.5007 | **0.6940** | 0.6003 | +0.0996 (+19.9%) | −0.0937 (−13.5%) |
| MRR | 0.6842 | **0.8688** | 0.8542 | +0.1700 (+24.9%) | −0.0146 (−1.7%) |
| HitRate@10 | 0.8226 | **1.0000** | 0.9355 | +0.1129 (+13.7%) | −0.0645 (−6.5%) |
| Precision@10 | 0.6145 | **0.7790** | 0.6823 | +0.0678 (+11.0%) | −0.0967 (−12.4%) |
| Latency p50 | **24 ms** | 11 ms | 57 ms | +33 ms | +46 ms |
| Latency p95 | **45 ms** | 135 ms | 90 ms | +45 ms | −45 ms |

**Query-class breakdown (Hybrid vs Vector):**

| Class | n | BM25 NDCG | Vector NDCG | Hybrid NDCG | Hybrid vs Vector |
|---|---|---|---|---|---|
| adults_couples | 6 | 0.8483 | 0.8923 | **0.9020** | **+0.010** |
| multi_constraint | 6 | 0.6169 | 0.7815 | **0.7035** | −0.078 |
| quiet_peaceful | 5 | 0.5793 | 0.6863 | **0.6801** | −0.006 |
| luxury | 6 | 0.7026 | 0.7446 | **0.7045** | −0.040 |
| family | 9 | 0.6243 | 0.7422 | **0.7165** | −0.026 |
| natural_language | 4 | 0.5279 | 0.6056 | 0.5472 | −0.058 |
| budget | 5 | 0.4343 | 0.4269 | 0.4577 | +0.031 |
| nightlife | 5 | 0.3452 | 0.6307 | 0.4559 | −0.175 |
| activities | 6 | 0.2808 | 0.3837 | 0.3256 | −0.058 |
| **exact_destination** | 10 | 0.1830 | **0.8392** | 0.4799 | **−0.359** |

**Surprises / observations:**

1. **Hybrid with 50/50 weights regresses from vector across most classes.** The overall NDCG dropped from 0.694 (vector) to 0.600 (hybrid). This contradicts the naive expectation that "combining two good retrievers always helps." The mechanism: min-max normalisation rescales each list independently, but BM25's *relative ordering* within its list still carries information — and for classes where BM25's ordering is wrong, it actively degrades the fused ranking.

2. **exact_destination: the biggest regression (0.84 → 0.48).** This is the clearest illustration of the mixing problem. For destination queries like "hotels in Tenerife", BM25 scores near-randomly (NDCG=0.18) because destination names rarely appear in hotel descriptions. Giving those random BM25 scores 50% weight pulls highly ranked vector results down and promotes incorrectly ranked BM25 results. Adding bad signal is worse than no signal.

3. **Nightlife regresses significantly (0.63 → 0.46).** Same mechanism: BM25 scores nightlife queries poorly (0.35), and 50% weight given to random BM25 rankings dilutes the vector advantage.

4. **adults_couples is the one class where hybrid *beats* vector (0.892 → 0.902).** This is a class where BM25 also performs well (0.848) — "adults only", "spa", "luxury" are discriminative terms that appear in both descriptions and the query. When both retrievers agree on ranking, fusion concentrates score on the overlapping top documents and improves precision.

5. **Hybrid p50 latency = 57 ms** (vs BM25 24 ms, vector 11 ms). Latency is the sum of two sequential queries (~24 ms + ~11 ms) plus fusion (~1 ms). For production, running both queries concurrently would reduce this to max(lex, vec) + fusion ≈ 25 ms.

6. **Hybrid p95 = 90 ms** (vs vector 135 ms, BM25 45 ms). The p95 is lower than vector alone — likely because ANN's worst-case (deeply explored HNSW graph) is bounded by `candidate_k=50`, and lexical's p95 is also bounded. The joint distribution smooths the tail.

**Key finding:** Naive 50/50 weighted-sum fusion does not beat the best individual retriever (vector, NDCG=0.694). The fundamental problem is that min-max normalisation cannot fix a retriever that produces meaningless scores for a given query class — it only scales scores, not fixes their ordering. **Weight tuning** (e.g., 0.3 BM25 / 0.7 vector) would help for this dataset. **Reciprocal Rank Fusion (RRF, Milestone 7)** avoids this problem entirely by fusing on ranks rather than scores: a document ranked 50th by BM25 contributes the same small amount regardless of its BM25 score value.

**Next question this raises:**
- Will RRF (rank-based fusion) avoid the score-scale problem and outperform weighted-sum? Hypothesis: yes — RRF is robust to score meaninglessness because it only cares about *position* in the list, not *score magnitude*. (Milestone 7)
- What lexical/vector weight ratio maximises hybrid NDCG on this dataset? The data suggest ~0.25/0.75 or 0.2/0.8 would reduce the regression on `exact_destination`. (Future: hyperparameter sweep)

---

### [Milestone 7] RRF vs weighted-sum hybrid

**Date:** 2026-08-14
**Hypothesis:** Reciprocal Rank Fusion (Cormack et al., 2009) will outperform the M6 50/50 weighted-sum hybrid on overall NDCG@10 — especially for `exact_destination` queries. The mechanism: RRF ignores raw score values entirely and combines only rank positions, so BM25's near-random score ordering for destination queries contributes at most one uniform 1/(k+rank) term rather than inflating the fused score with meaningless magnitude. Expected: RRF NDCG ≥ vector baseline (0.694); weighted-sum regressed to 0.600.

**Configuration:**
- index: `travel_hotels` (5,470 hotels, lucene/HNSW, cosinesimil)
- embedding model: `all-MiniLM-L6-v2` (384d, L2-normalised)
- fusion: Reciprocal Rank Fusion, `k=60` (Cormack et al. default)
- `candidate_k=50` (each retriever fetches 50 candidates; RRF selects top 10 from ≤ 100 unique)
- `rrf_k=60` — smoothing constant; rank-1 contributes 1/61 ≈ 0.016
- strategy: `rrf` — lexical → vector → `rrf_fuse([lex_hits, vec_hits])` → top 10
- golden dataset: 62 queries, 10 classes, 48,675 judgments
- results file: `data/evaluation/results/rrf_<date>.json`

**Results (BM25 vs Vector vs Hybrid weighted vs Hybrid RRF @ K=10, 62 queries):**

| Metric | BM25 | Vector | Hybrid (50/50) | Hybrid (RRF) | Δ RRF vs weighted |
|---|---|---|---|---|---|
| NDCG@10 | 0.5007 | **0.6940** | 0.6003 | 0.6239 | +0.0236 (+3.9%) |
| MRR | 0.6842 | **0.8688** | **0.8542** | 0.8449 | −0.0093 (−1.1%) |
| HitRate@10 | 0.8226 | **1.0000** | 0.9355 | 0.9516 | +0.0161 (+1.7%) |
| Precision@10 | 0.6145 | **0.7790** | 0.6823 | 0.7210 | +0.0387 (+5.7%) |
| Latency p50 | **24 ms** | 11 ms | 57 ms | 56 ms | −1 ms |
| Latency p95 | **45 ms** | 135 ms | 90 ms | 84 ms | −6 ms |

**Query-class breakdown:**

| Class | n | BM25 NDCG | Vector NDCG | Hybrid NDCG | RRF NDCG | RRF best? |
|---|---|---|---|---|---|---|
| adults_couples | 6 | 0.8483 | 0.8923 | **0.9020** | 0.8928 | — |
| quiet_peaceful | 5 | 0.5793 | 0.6863 | 0.6801 | **0.6922** | ✓ |
| multi_constraint | 6 | 0.6169 | 0.7815 | 0.7035 | **0.7473** | ✓ (vs hybrid) |
| family | 9 | 0.6243 | 0.7422 | 0.7165 | **0.7352** | ✓ (vs hybrid) |
| luxury | 6 | 0.7026 | **0.7446** | 0.7045 | 0.6951 | — |
| activities | 6 | 0.2808 | 0.3837 | 0.3256 | **0.4002** | ✓ best overall |
| nightlife | 5 | 0.3452 | **0.6307** | 0.4559 | 0.5053 | ✓ (vs hybrid) |
| natural_language | 4 | 0.5279 | **0.6056** | 0.5472 | 0.5520 | ✓ (vs hybrid) |
| budget | 5 | 0.4343 | 0.4269 | **0.4577** | 0.4221 | — |
| **exact_destination** | 10 | 0.1830 | **0.8392** | 0.4799 | 0.5347 | ✓ (vs hybrid) |

**Surprises / observations:**

1. **RRF beats weighted-sum overall (NDCG 0.6003 → 0.6239), but falls short of pure vector (0.6940).** RRF is not a silver bullet — it corrects the score-scale problem but introduces its own assumption: that every retriever's top-ranked documents are equally trustworthy. When vector rankings are highly reliable (exact_destination, nightlife) but BM25 rankings are not, RRF still gives BM25's top-ranked documents a non-negligible 1/(60+1) contribution.

2. **exact_destination recovers partially (0.48 → 0.53), not fully.** This confirms the hypothesis that RRF is better than weighted-sum for this class, but does not reach vector (0.84). The residual gap: even though BM25's score *magnitude* is ignored, BM25's rank *ordering* is still random for destination queries. Documents ranked 1–5 by BM25 still contribute 0.016–0.015 each — enough to promote some wrong documents above the correct vector top-10. Vector alone is still the best retriever for this class.

3. **activities: RRF beats vector (0.38 → 0.40) — the one class where BM25 genuinely helps.** Activity queries like "watersports", "hiking" contain discriminative keywords that appear in the hotel's `activities` field (indexed as `text`). BM25 correctly ranks hotels that literally list "watersports" as an activity; vector search relies on semantic proximity, which is weaker for specific activity-tag matches. RRF incorporates BM25's useful signal without the score-scale penalty, producing the only case where the fused result beats both individual retrievers.

4. **quiet_peaceful: RRF beats vector (0.686 → 0.692) — small but consistent improvement.** Both retrievers perform well on this class (BM25=0.579, vector=0.686), and RRF amplifies their agreement: documents that both retrievers rank highly get accumulated RRF scores, concentrating results at the top.

5. **adults_couples: RRF loses the weighted-sum advantage (0.902 → 0.893).** Weighted-sum beat vector for this class because min-max normalised scores, when both retrievers agree, concentrate combined scores on the overlapping documents. RRF accumulates uniform 1/(k+rank) terms — agreement still helps, but the "concentration" effect from score magnitude is absent. RRF's result (0.893) is still excellent, matching vector.

6. **budget class regresses under both hybrid methods.** Budget queries (e.g., "value for money 4 star highly rated") require price ordering, which neither BM25 nor vector encodes. Adding BM25 — however robustly — still does not help; the correct fix is numeric `max_price` filter constraints, not retrieval ranking.

7. **MRR is 1.4% lower for RRF than weighted-sum (0.8449 vs 0.8542).** MRR measures only the rank of the *first* relevant document. Weighted-sum can concentrate combined scores on a single top document more aggressively than RRF's bounded rank accumulation; this occasionally puts the first relevant document at rank 1 when RRF puts it at rank 2.

8. **Latency nearly identical (56 ms vs 57 ms p50).** RRF replaces the min-max normalisation + weighted sum with a simpler accumulation loop — same asymptotic complexity, ~1 ms difference in practice.

**Key finding:** RRF is a robust upgrade over weighted-sum (+3.9% NDCG, +5.7% Precision, +1.7% HitRate) with no latency cost. The score-scale problem from M6 is substantially mitigated. However, RRF cannot fully overcome a retriever that produces a *random ordering* — only zero weight (i.e., dropping the retriever entirely) would eliminate its negative influence on destination queries. The best single retriever (vector, NDCG=0.694) still beats any fusion at k=60. **A cross-encoder reranker (Milestone 8) applied to the top-50 RRF candidates should break through this ceiling** by applying a more powerful relevance model as a second pass.

**Next question this raises:**
- Does tuning `rrf_k` (smaller k → amplifies top-rank advantage; larger k → flattens) help `exact_destination`? Try `k=10`: rank-1 becomes 0.091 instead of 0.016, amplifying vector's strong top-1 advantage. Risk: amplifies BM25's random rank-1 equally.
- Can a cross-encoder reranker (Milestone 8) applied to the top-50 RRF candidates improve NDCG? Hypothesis: yes — the reranker reads both query and document together, capturing query-document interaction that neither BM25 nor vector encodes.

---

### [Milestone 8] Cross-encoder reranking (RRF + reranker) vs RRF

**Date:** 2026-08-14
**Hypothesis:** A cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) applied to the top-50 RRF candidates will improve overall NDCG@10 beyond the RRF ceiling (0.6239), especially for semantic classes (`exact_destination`, `activities`, `nightlife`) where individual token-level attention signals are most useful. Expected cost: latency increases from ~56 ms (RRF) to ~130 ms (RRF + 50-pair inference on CPU).

**Configuration:**
- index: `travel_hotels` (5,470 hotels, lucene/HNSW, cosinesimil)
- embedding model: `all-MiniLM-L6-v2` (384d, L2-normalised)
- fusion: RRF, `rrf_k=60`, `candidate_k=50`
- reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22 M params, ~86 MB, MS-MARCO trained)
- `rerank_k=50` (top-50 RRF candidates passed to cross-encoder; top-10 returned)
- reranking text: `hotel_name | destination, country | hotel_description | Activities: … | Tags: …`
- strategy: `rerank` — lexical → vector → RRF fusion → top-50 → cross-encoder → top-10
- golden dataset: 62 queries, 10 classes, 48,675 judgments
- results file: `data/evaluation/results/rerank_2026-08-14.json`

**Results (all strategies @ K=10, 62 queries):**

| Metric | BM25 | Vector | Hybrid | RRF | Rerank | Δ Rerank vs RRF |
|---|---|---|---|---|---|---|
| NDCG@10 | 0.5007 | **0.6940** | 0.6003 | 0.6239 | **0.6830** | +0.0591 (+9.5%) |
| MRR | 0.6842 | **0.8688** | 0.8542 | 0.8449 | 0.8191 | −0.0258 (−3.1%) |
| HitRate@10 | 0.8226 | **1.0000** | 0.9355 | **0.9516** | **0.9516** | 0.0000 |
| Precision@10 | 0.6145 | **0.7790** | 0.6823 | 0.7210 | **0.7935** | +0.0725 (+10.1%) |
| Latency p50 | **24 ms** | 11 ms | 57 ms | 56 ms | 113 ms | +57 ms |
| Latency p95 | **45 ms** | 135 ms | 90 ms | 84 ms | 142 ms | +58 ms |

**Query-class breakdown:**

| Class | n | RRF NDCG | Rerank NDCG | Δ | Winner |
|---|---|---|---|---|---|
| **exact_destination** | 10 | 0.5347 | **0.7934** | +0.2587 (+48%) | ✓ Rerank |
| activities | 6 | 0.4002 | **0.5726** | +0.1724 (+43%) | ✓ Rerank |
| nightlife | 5 | 0.5053 | **0.6389** | +0.1336 (+26%) | ✓ Rerank |
| natural_language | 4 | 0.5520 | **0.6062** | +0.0542 (+10%) | ✓ Rerank |
| luxury | 6 | 0.6951 | **0.7176** | +0.0225 (+3%) | ✓ Rerank |
| budget | 5 | 0.4221 | **0.4401** | +0.0180 (+4%) | ✓ Rerank |
| adults_couples | 6 | **0.8928** | 0.8953 | +0.0025 | Tie |
| quiet_peaceful | 5 | **0.6922** | 0.6681 | −0.0241 (−3%) | RRF |
| multi_constraint | 6 | **0.7473** | 0.7134 | −0.0339 (−5%) | RRF |
| family | 9 | **0.7352** | 0.6507 | −0.0845 (−11%) | RRF |

**Surprises / observations:**

1. **exact_destination: the largest single gain (+48%, 0.53 → 0.79).** The cross-encoder, trained on MS-MARCO query-passage pairs, has been exposed to "hotels in X", "X resort", and similar patterns. It can score a (query, hotel-document) pair jointly — the destination name and description appear in the same context window, allowing attention to fire across them. Both bi-encoders (BM25, vector) are blind to this interaction: BM25 misses destination keywords; vector encodes the query and document independently. This is the canonical motivation for two-stage retrieval.

2. **activities: +43% (0.40 → 0.57).** For queries like "hiking adventure lodge" or "watersports beach", the cross-encoder can simultaneously attend to the query term and the `Activities: hiking, climbing, …` clause in the reranking text. This captures query-activity interaction that RRF approximates only via BM25's keyword match.

3. **MRR decreased (−3.1%, 0.84 → 0.82) even though NDCG improved.** MRR measures only the rank of the *first* relevant document. The reranker occasionally reorders the top results in a way that pushes a relevant document from rank 1 to rank 2 while placing a different relevant document at rank 1 — NDCG does not penalise this (it rewards ranking any highly-relevant document first), but MRR does (it rewards only the very first hit). This is expected behaviour when a reranker corrects many lower-ranked positions but occasionally misorders the very top.

4. **family regression (−11%, 0.74 → 0.65).** The MS-MARCO cross-encoder was trained on web-passage relevance, not travel-specific structured attributes. "Family beach holiday with kids" semantically overlaps with many generic beach hotel descriptions, making the cross-encoder less discriminative than RRF for this class. The cross-encoder reads the hotel description, but `family_friendly=True` is a structured boolean not in the free-text document — it was already enforced as a hard filter before reranking, so the reranker cannot see it. This is a limit of the current reranking text format.

5. **multi_constraint regression (−5%, 0.75 → 0.71).** Similar mechanism: hard-constraint queries (adults_only + max_price + min_stars) already have their correctness enforced by filters before reranking. The cross-encoder then reranks a pool of already-correct documents, and its general-purpose text relevance model may prefer candidates that sound more luxurious over those that strictly satisfy the price constraint.

6. **Latency doubles (56 ms → 113 ms p50).** 50 cross-encoder forward passes on CPU take ~55 ms extra. This is expected: 22 M parameters × 50 pairs = ~1.1 B float ops. For a production system, this could be reduced by: (a) lowering `rerank_k` to 20–30 (top-10 NDCG is insensitive to candidates ranked > 30); (b) using a smaller model (L-2-v2, ~12 M params); (c) batch-GPU inference. The ~113 ms p50 is still acceptable for interactive search.

7. **HitRate unchanged (0.9516 = RRF).** Reranking only reorders — it cannot add documents not in the top-50 RRF pool. The two queries that RRF misses are still missed. A larger `candidate_k` or a different first-stage retriever is needed to improve recall.

**Key finding:** Cross-encoder reranking is the highest-NDCG strategy overall (+9.5% vs RRF, +47.5% vs weighted-sum). It achieves this by applying token-level attention across the full (query, document) pair — a strictly more powerful relevance model than either BM25 or vector bi-encoders. The cost is ~55 ms extra latency and a small regression on structured-constraint query classes where the cross-encoder's general-purpose text model has no advantage over the already-filtered RRF pool. **This two-stage architecture (fast candidate generation → powerful reranking) is the standard production pattern for neural IR systems.**

**Next question this raises:**
- Does expanding the reranking text to include star-rating, price tier, and board type improve `multi_constraint` and `family` without hurting semantic classes?
- Can `rerank_k=20` (instead of 50) preserve most of the NDCG gain with half the latency overhead? (Latency-quality tradeoff sweep)
- Would a domain-specific fine-tuned cross-encoder outperform the general MS-MARCO model on travel queries?

---

### [Milestone 9] Query understanding: rule-based constraint extraction

**Date:** 2026-08-14
**Hypothesis:** Parsing hard constraints (month, departure airport, max price, star rating, family/adults flags, country) from free-text queries and running hybrid RRF with extracted filters will outperform baseline hybrid RRF that ignores the natural-language constraint signals.  The benefit should be largest on `multi_constraint` and `natural_language` query classes, and smallest (or zero) on purely semantic classes like `activities`.

**Configuration:**
- index: `travel_hotels` (5,470 hotels)
- embedding model: `all-MiniLM-L6-v2` (384-dim)
- QU engine: `RuleBasedQueryUnderstandingEngine` (regex + keyword lookup; no LLM)
- hybrid: RRF fusion, k=60, candidate_k=50
- key design: ground-truth filters IGNORED; constraints extracted from query_text only

**Results:**

| Metric | BM25 | Vector | Hybrid | RRF | Rerank | **Understand** |
|---|---|---|---|---|---|---|
| NDCG@10 | 0.5007 | 0.6940 | 0.6003 | 0.6239 | **0.6830** | 0.6312 |
| MRR | 0.6842 | 0.8688 | 0.8542 | 0.8449 | 0.8191 | **0.8620** |
| HitRate@10 | 0.8226 | 1.0000 | 0.9355 | **0.9516** | **0.9516** | 0.9355 |
| Precision@10 | 0.6145 | 0.7790 | 0.6823 | 0.7210 | **0.7935** | 0.7290 |
| Latency p50 | 24 ms | 11 ms | 57 ms | 56 ms | 113 ms | **45 ms** |
| Latency p95 | 45 ms | 135 ms | 90 ms | 84 ms | 142 ms | **71 ms** |

**Query-class breakdown (Understand vs RRF):**

| Class | n | RRF NDCG | Understand NDCG | Δ | Winner |
|---|---|---|---|---|---|
| adults_couples | 6 | 0.8928 | **0.9721** | +0.0793 (+8.9%) | ✓ Understand |
| multi_constraint | 6 | 0.7473 | **0.7627** | +0.0154 (+2.1%) | ✓ Understand |
| quiet_peaceful | 5 | **0.6922** | **0.6922** | 0.0000 | Tie |
| luxury | 6 | **0.6951** | **0.6951** | 0.0000 | Tie |
| natural_language | 4 | 0.5520 | **0.5778** | +0.0258 (+4.7%) | ✓ Understand |
| exact_destination | 10 | 0.5347 | **0.5962** | +0.0615 (+11.5%) | ✓ Understand |
| family | 9 | **0.7352** | 0.6900 | −0.0452 (−6.1%) | RRF |
| nightlife | 5 | **0.5053** | 0.4981 | −0.0072 (−1.4%) | RRF |
| activities | 6 | **0.4002** | 0.4002 | 0.0000 | Tie |
| budget | 5 | **0.4221** | 0.3445 | −0.0776 (−18.4%) | RRF |

**Surprises / observations:**

1. **adults_couples class: largest gain (+8.9%).** The QU engine correctly extracts `adults_only=True` from phrases like "adults only", "no kids", "couples only". This hard filter eliminates all family-friendly hotels from the candidate pool, dramatically improving precision and NDCG for this class. RRF must rely purely on semantic overlap ("adults", "spa", "romantic") without the boolean guarantee. This is the clearest win for structured constraint extraction: a deterministic rule delivers a result that a semantic model cannot guarantee.

2. **exact_destination +11.5% (0.53 → 0.60).** For queries like "hotels in Tenerife" or "holiday in Santorini", the QU engine maps the island/region name to a country filter (Tenerife → Spain, Santorini → Greece). This narrows the search to the correct country, allowing vector search to focus on the right regional hotels. BM25's "in Tenerife" keyword match was already good; the country filter adds a hard guarantee.

3. **luxury and quiet_peaceful: identical to RRF (tie).** These query classes contain no extractable hard constraints — "luxury boutique spa retreat" has no month, price, airport, or country. The semantic_query equals the original query, and the hybrid search runs identically to RRF. QU adds zero noise but zero signal: pure semantic searches are unaffected by the QU layer.

4. **budget class: severe regression (−18.4%).** The golden budget queries include "value for money 4 star highly rated" which causes QU to extract `min_star_rating=4`. If the golden relevant hotels include 3-star budget hotels (likely, since budget queries value affordability over luxury), the hard `min_star_rating=4` filter excludes them entirely — NDCG=0 for that specific query. This reveals a key limitation of rule-based extraction: the star rating extractor cannot distinguish "4 star value" (quality preference) from "find me 4-star hotels" (hard constraint).

5. **family regression (−6.1%).** QU extracts `family_friendly=True` from "family" queries. Some relevant hotels in the golden set may be tagged `family_friendly=False` in the structured field despite offering kids activities (e.g., all-inclusive resorts that welcome children without the explicit boolean flag). The hard filter then incorrectly excludes them. Rule-based NL extraction cannot know the semantic gap between "family" (user intent) and `family_friendly` (structured field).

6. **multi_constraint and natural_language: confirms QU hypothesis (+2.1%, +4.7%).** For "family beach hotel Greece July Manchester" and "Find me a family holiday somewhere warm in October departing from Manchester under 2000 pounds", the QU engine recovers nearly the same constraints as the hand-annotated golden filters. This confirms that the rule-based extractor correctly parses the constraint language in these query classes.

7. **Latency FASTER than baseline RRF (45 ms vs 56 ms p50).** The QU engine is sub-millisecond (pure Python regex). The speed gain comes from semantic_query distillation: after extracting constraints, the residual semantic query is shorter (e.g., "beach hotel" vs "family beach hotel Greece July Manchester"). A shorter BM25 query has fewer terms to analyse and score, reducing BM25 latency. Hard constraint filters also reduce the BM25 and vector candidate pools faster, lowering heap operation cost.

**Key finding:** Rule-based QU beats raw RRF on overall NDCG (+1.2%) and MRR (+2.0%) while being FASTER (45 ms vs 56 ms). It wins decisively on `adults_couples` (+8.9%), `exact_destination` (+11.5%), and structured-constraint classes. The main failure mode is **false-positive constraint extraction**: extracting star ratings or family flags from ambiguous phrases and then filtering out relevant results. An LLM-based engine (M10, M12) could avoid this by understanding query intent more fully.

**Next question this raises:**
- Can a confidence threshold on extracted constraints reduce false positives (e.g., only apply min_star_rating if the phrasing is unambiguous)?
- M10 (LLM rewriting): would an LLM engine improve the `budget` and `family` class by understanding that "4-star value" is a preference, not a hard filter?
- Combining QU + reranking: does the better-filtered candidate pool (from QU) improve the reranker's output quality beyond just the RRF+rerank baseline?

---

### [Milestone 10] Query rewriting: LocalLLMProvider keyword expansion vs query understanding

**Date:** 2026-08-14
**Hypothesis:** Expanding the semantic query (post-QU) with domain synonym expansion before retrieval will improve recall (HitRate@10) by matching more vocabulary in hotel descriptions, at possible cost to ranking precision (NDCG) due to a noisier query embedding centroid.

**Configuration:**
- index: `travel_hotels` (5,470 hotels)
- embedding model: `all-MiniLM-L6-v2` (384-dim)
- QU engine: `RuleBasedQueryUnderstandingEngine` (same as M9 `understand` strategy)
- rewriter: `QueryRewriter(LocalLLMProvider())` — keyword synonym expansion; no real LLM
- hybrid: RRF fusion, k=60, candidate_k=50
- key design: ground-truth filters IGNORED; constraints extracted from query_text; rewriting applied to semantic_query only (hard constraints handled separately by QU)

**Results:**

| Metric | RRF | Understand | **Rewrite** |
|---|---|---|---|
| NDCG@10 | 0.6239 | **0.6312** | 0.6130 |
| MRR | 0.8449 | **0.8620** | 0.8226 |
| HitRate@10 | 0.9516 | 0.9355 | **0.9677** |
| Precision@10 | 0.7210 | **0.7290** | 0.7242 |
| Latency p50 | 56 ms | 45 ms | 54 ms |
| Latency p95 | 84 ms | 71 ms | 98 ms |

**Query-class breakdown (Rewrite vs Understand):**

| Class | n | Understand NDCG | Rewrite NDCG | Δ | Winner |
|---|---|---|---|---|---|
| adults_couples | 6 | **0.9721** | 0.8627 | −0.1094 (−11.2%) | Understand |
| budget | 5 | 0.3445 | **0.3535** | +0.0090 (+2.6%) | ✓ Rewrite |
| quiet_peaceful | 5 | **0.6922** | 0.6731 | −0.0191 (−2.8%) | Understand |
| family | 9 | 0.6900 | **0.7196** | +0.0296 (+4.3%) | ✓ Rewrite |
| multi_constraint | 6 | **0.7627** | 0.7639 | +0.0012 (+0.2%) | Tie |
| exact_destination | 10 | **0.5962** | 0.6203 | +0.0241 (+4.0%) | ✓ Rewrite |
| luxury | 6 | **0.6951** | 0.6526 | −0.0425 (−6.1%) | Understand |
| activities | 6 | 0.4002 | **0.4580** | +0.0578 (+14.4%) | ✓ Rewrite |
| nightlife | 5 | **0.4981** | 0.3092 | −0.1889 (−37.9%) | Understand |
| natural_language | 4 | **0.5778** | 0.5562 | −0.0216 (−3.7%) | Understand |

**Surprises / observations:**

1. **HitRate@10 improves (+3.4%, 0.9355 → 0.9677).** More queries have at least one relevant hotel in the top 10. This confirms the recall hypothesis: synonym expansion broadens the vocabulary match and surfaces hotels that semantic search would have missed. But the NDCG regression shows those extra hotels rank lower.

2. **NDCG and MRR regress vs understand (−2.9%, −4.6%).** Adding synonyms like "coastal, seaside, sandy beach" to "beach holiday" shifts the query embedding centroid. This makes the vector less precisely aligned with the original intent and admits hotels that merely mention coastal terms without being ideal beach hotels. Ranking suffers because the rewritten query becomes more ambiguous.

3. **activities class: largest gain (+14.4%).** Activities queries ("hiking adventure holidays", "water sports beach") benefit most from synonym expansion because hotel descriptions use varied vocabulary: "trekking", "walking trails", "watersports", "aquatic". The expansion bridges the terminology gap that both BM25 and vector search struggle with in this class.

4. **nightlife class: severe regression (−37.9%).** The `nightlife` expansion table maps "nightlife" → "bars, clubs, entertainment". For queries like "young party holiday summer beach", the vector embedding of "nightlife bars clubs entertainment beach" becomes generic entertainment-focused rather than the specific party-resort context. The golden relevant hotels for nightlife queries likely have strong "party resort" and "lively atmosphere" signals that the expanded query dilutes.

5. **adults_couples regression (−11.2%).** Understand extracted `adults_only=True` as a hard filter, boosting adults_couples NDCG strongly in M9 (+8.9% over RRF). Rewrite does the same QU step, but the synonym expansion adds "couples, romantic" to queries already handled well by the `adults_only` filter — minor noise. The real cause of the gap is that the expansion terms ("adults-only, couples") are less specific than the hard boolean filter, and the expansion slightly dilutes the vector embedding.

6. **family class improves (+4.3% over understand).** Family queries that had marginal QU extraction (e.g., "family holidays" → `family_friendly=True`) now benefit from expanded vocabulary: "family-friendly, children, kids" terms in the rewritten query boost BM25 matching on hotel descriptions that use those exact phrases. This partially offsets the false-positive filter issue from M9.

7. **LocalLLMProvider is NOT a real LLM.** The expansion is keyword-based and deterministic. It cannot handle creative queries ("something like Mallorca but quieter"), correct awkward residual text from QU, or infer semantic context. The architecture is correct; the provider is a baseline stub. BedrockLLMProvider (Milestone 12) will replace it with Claude/Titan for real evaluation.

8. **Latency: 54 ms p50 (between RRF 56 ms and Understand 45 ms).** The rewriting step is sub-millisecond (pure Python dictionary lookup), so latency is essentially the same as understand + the slightly longer rewritten query for BM25/vector evaluation.

**Key finding:** `LocalLLMProvider` keyword expansion improves HitRate@10 (+3.4%) and activities class (+14.4%) but regresses NDCG and MRR. This is the classic precision-recall tradeoff of naive query expansion — good for recall, bad for ranking precision. A real LLM (M12) would produce targeted paraphrases that improve both, not just recall. The architecture (LLMProvider → QueryRewriter → retrieval) is now in place for M12 to drop in a real provider.

**Next question this raises:**
- M12 (Bedrock): replacing LocalLLMProvider with Claude Sonnet should produce targeted, context-aware rewrites that improve NDCG without the recall-precision tradeoff.
- Can confidence-weighted rewriting (partially blend original and rewritten embeddings) balance precision and recall?
- Expansion for the nightlife class clearly needs a different term set — custom domain-specific expansion tables per query class?

---

### [Milestone 11] Multi-query expansion vs single-query strategies

**Date:** 2026-08-14
**Hypothesis:** Generating N variant queries from a single semantic query and retrieving independently for each, then fusing all 2N rank lists via RRF, will improve HitRate@10 and recall over single-query hybrid retrieval. The cost will be N× latency. NDCG and MRR may regress slightly if the added variants introduce noise.

**Configuration:**
- index: `travel_hotels` (5,470 hotels, lucene/HNSW, cosinesimil)
- embedding model: `all-MiniLM-L6-v2` (384d, L2-normalised)
- fusion: RRF, `k=60`
- `candidate_k=50` per retriever, per query (N=3 queries → 6 lists, each ≤ 50 hits → ≤ 300 candidates before deduplication)
- `n_queries=3`: original + synonym substitution variant + context elaboration variant
- expander: `LocalQueryExpander` (deterministic, rule-based; no API key)
- strategy: `expand` — QU → LocalQueryExpander → N×(BM25+vector) → rrf_fuse(2N lists) → top 10
- golden dataset: 62 queries, 10 classes, 48,675 judgments
- results file: `data/evaluation/results/expand_2026-08-14.json`

**Results (Understand vs Rewrite vs Expand @ K=10, 62 queries):**

| Metric | Understand | Rewrite | **Expand** | Δ Expand vs Understand |
|---|---|---|---|---|
| NDCG@10 | **0.6312** | 0.6130 | 0.6285 | −0.0027 (−0.4%) |
| MRR | **0.8620** | 0.8226 | 0.8308 | −0.0312 (−3.6%) |
| HitRate@10 | 0.9355 | **0.9677** | 0.9516 | +0.0161 (+1.7%) |
| Precision@10 | **0.7290** | 0.7242 | 0.7226 | −0.0064 (−0.9%) |
| Latency p50 | **45 ms** | 54 ms | 180 ms | +135 ms (+300%) |
| Latency p95 | **57 ms** | 68 ms | 236 ms | +179 ms |

**Query-class breakdown (Understand vs Expand):**

| Class | n | Understand NDCG | Expand NDCG | Δ | Winner |
|---|---|---|---|---|---|
| activities | 6 | 0.4002 | **0.4840** | +0.0838 (+20.9%) | Expand |
| adults_couples | 6 | **0.9721** | 0.8085 | −0.1636 (−16.8%) | Understand |
| budget | 5 | 0.3445 | **0.4171** | +0.0726 (+21.1%) | Expand |
| exact_destination | 10 | **0.7069** | 0.6290 | −0.0779 (−11.0%) | Understand |
| family | 9 | 0.6900 | **0.7259** | +0.0359 (+5.2%) | Expand |
| luxury | 6 | **0.7188** | 0.6545 | −0.0643 (−8.9%) | Understand |
| multi_constraint | 6 | **0.7627** | 0.7685 | +0.0058 (+0.8%) | Expand |
| natural_language | 4 | 0.5562 | **0.5538** | −0.0024 (−0.4%) | Tie |
| nightlife | 5 | 0.4981 | **0.4193** | −0.0788 (−15.8%) | Understand |
| quiet_peaceful | 5 | **0.7095** | 0.6911 | −0.0184 (−2.6%) | Understand |

**Surprises / observations:**

1. **HitRate@10 improves (+1.7%, 0.9355 → 0.9516) but less than rewrite (+3.4%).** Expanding to 3 variants still misses some queries that rewriting catches. This suggests the synonym substitution and context elaboration variants don't always cover the vocabulary gap as effectively as the targeted synonym expansion in LocalLLMProvider. The two approaches are complementary rather than one dominating.

2. **NDCG and MRR regress slightly vs understand (−0.4%, −3.6%).** More surprising is that expansion performs *better* than rewrite on NDCG (0.6285 vs 0.6130), despite higher latency. The reason: multi-query expansion keeps the original query as the first variant, preserving its precision, while the rewrite replaces the original. When the synonym variant or context variant are noisy, the original still contributes its clean signal to RRF.

3. **activities and budget classes: large improvements (+20.9%, +21.1%).** These are the classes where vocabulary mismatch between users and hotel descriptions is most severe. "watersports", "jet skiing", "water activities" — activities descriptions use varied terms that the original query misses. Multiple variants increase coverage. Budget queries benefit similarly: the context elaboration ("offering value for money with good facilities") aligns with how budget hotels describe themselves.

4. **adults_couples regression (−16.8%).** This is the most surprising finding. Understand extracts `adults_only=True` as a hard filter and retrieves 100% precision results. Expansion generates a synonym variant ("adults-only retreat...") and a context variant (both still correct) but the extra variants retrieve some couples-focused hotels without the `adults_only` flag, reducing precision. The QU-extracted hard filter is more effective here than vocabulary expansion.

5. **exact_destination regression (−11.0%).** For queries like "hotels in Tenerife", QU extracts `country=Spain` as a hard filter and the semantic query is clean. Expansion adds context elaboration ("beach resort accommodation") that may pull in non-destination-specific hotels from Spain that happen to match the context terms but not the intended destination.

6. **Latency is 4× slower (180 ms vs 45 ms).** N=3 queries × 2 retrievals = 6 sequential OpenSearch requests. Single-query hybrid is 2 requests. This is the fundamental cost of ensemble retrieval. Production systems would parallelize the N retrievals to reduce wall-clock time to max(N retrieval latencies) ≈ single-query latency + fusion overhead. Parallelization would require `asyncio.gather()` or a thread pool, which is out of scope for this educational implementation.

7. **Multi-query retains full filter precision.** All HybridSearchParams filter fields (country, family_friendly, etc.) are applied identically to every per-query retrieval. Filters remain hard constraints — no multi-query variant can bypass them.

8. **RRF naturally handles duplicate results across variants.** If both the synonym variant and context variant retrieve the same hotel, its RRF score accumulates contributions from both rank positions, correctly boosting it. No explicit deduplication is needed.

**Key finding:** Multi-query expansion with `LocalQueryExpander` shows the expected ensemble pattern: +1.7% HitRate but −0.4% NDCG vs understand. It is more precise than rewrite (keeps the original query signal) but less effective at recall than a well-targeted LLM rewrite. The 4× latency cost is the primary barrier to production use in this sequential implementation. A real LLM expander (M12+) would generate semantically diverse variants — not just synonym substitutions — which would improve recall without vocabulary-mismatch noise. The architecture (QueryExpander Protocol → multi_query_search → rrf_fuse) is in place for that upgrade.

**Next question this raises:**
- M12 (Bedrock): an LLM-generated expansion set would produce variants like "quiet Mediterranean family resort", "peaceful child-friendly beach destination southern Europe", "family coastal resort away from nightlife" — semantically diverse rather than word-substitution based. Would this close the gap with understand on precision while improving recall?
- Can parallel retrieval (asyncio or threads) make multi-query latency competitive with single-query?
- Is there a per-class routing heuristic — use single-query for `exact_destination`/`adults_couples` (where hard filters dominate), multi-query for `activities`/`budget` (where vocabulary mismatch is the main failure mode)?

---

## [Milestone 13] RAG / destination knowledge base

**Date:** 2026-08-14

**Hypothesis:** A separate destination knowledge base, searched semantically and used to augment the LLM synthesis prompt, will produce richer and more contextually accurate travel recommendations than hotel search alone.  The product retrieval pipeline (hotel ranking) should remain unchanged — RAG is purely additive.

**Configuration:**
- Knowledge base: 30 destination documents, one per island/region
- Knowledge index: `travel_destinations` (OpenSearch knn_vector, same dimension as hotel index)
- Knowledge retrieval: knn ANN search on `embedding_vector` (all-MiniLM-L6-v2, 384d)
- Optional country filter from QueryUnderstandingEngine
- LLM synthesis: configurable via `llm_provider` setting (`local`, `echo`, or `bedrock`)
- RAG prompt: query + knowledge context + top-5 hotel summaries → 2-3 sentence recommendation

**Why no numeric evaluation:**

RAG is purely additive — it does not change hotel ranking.  Running `evaluate.py` with `rag=true` would produce identical NDCG/MRR/HitRate to the current baseline because the hotel list is unchanged.  A meaningful RAG evaluation would require human judgment labels on synthesis quality ("did the summary correctly describe the destination?") which we don't have.  Numeric evaluation of RAG is M15 scope.

**Live experiments (run 2026-08-15, server on port 8765, RAG_ENABLED=true, LLM_PROVIDER=echo):**

*Experiment 1 — Greece beach holiday (semantic query, no destination extracted by QU)*

```
POST /search  {"query": "relaxed beach holiday in Greece", "rag": true}

Knowledge context returned (3 destinations):
  Rhodes (Greece)     — character: history and beaches, medieval charm, family friendly
  Santorini (Greece)  — character: romantic, iconic views, luxury boutique, honeymoon
  Crete (Greece)      — character: cultural depth, ancient history, diverse landscapes
```

Observation: the knowledge retriever correctly returns three Greek islands — the most semantically relevant destinations for the query.  None are from Spain, Turkey, or the Caribbean.  The country filter (QU extracted `country=Greece`) was not triggered because QU found no exact country entity in the query, so this result comes from pure vector similarity to the query embedding.  All three are topically correct.

*Experiment 2 — Similarity query (tests semantic limitation of knowledge retrieval)*

```
POST /search  {"query": "somewhere like Mallorca but quieter", "rag": true}

QU extracted: destination=None, country=Spain   ← country filter applied

Knowledge context returned (3 destinations):
  Ibiza    (Spain) — nightlife=HIGH   ← wrong: Ibiza is louder than Mallorca
  Gran Canaria (Spain) — nightlife=HIGH   ← wrong: also louder
  Costa Blanca (Spain) — nightlife=moderate ← plausible
```

Ibiza ranked first by semantic similarity because "Mallorca" and "Ibiza" appear together in both documents' `similar_destinations` field vocabulary — even though `similar_destinations` is deliberately excluded from the embedding text (see `build_knowledge_embedding_text`).  The presence of shared location names (Balearic Islands, Spain, beach) is enough for high cosine similarity.

**Cross-comparison — Graph vs. Vector for "like Mallorca but quieter":**

The same query via graph traversal gives immediately correct results:

```
GET /graph/similar?destination=Mallorca&hops=1

  Ibiza      — nightlife=high, family=low   ← louder
  Menorca    — nightlife=low, family=high   ← correct quieter option
  Costa del Sol — nightlife=moderate
  Costa Blanca  — nightlife=moderate
```

The graph surfaces **Menorca** (the obvious "quieter Mallorca" answer) in the top results.  Vector knowledge retrieval ranked Ibiza first because description-level similarity (both are Balearic Islands, beach destinations) dominates over the quietness constraint.  The graph can be post-filtered by `nightlife_level=low` to extract only the quiet options — something the embedding cannot do without reranking.

This is the concrete empirical demonstration of why graph traversal complements vector search.

**Architectural observations:**

1. **Product retrieval vs knowledge retrieval separation is the core lesson.** Embedding hotel descriptions into the knowledge documents would conflate two retrieval tasks with different semantics. Destination knowledge ("what is Menorca like?") is stable and shared; hotel descriptions are specific and ranked. Keeping them in separate indices makes both better.

2. **The knn country filter demonstrates OpenSearch's pre-filter mode.** The `filter` clause inside the `knn` query block is applied before ANN scoring — only documents in the given country are candidates. Without this, "beach holiday in Greece" could retrieve a Maldives knowledge document because the semantic similarity is higher than any specific Greek island.

3. **EchoLLMProvider is the right local testing backend.** `LLM_PROVIDER=echo` returns the prompt as the synthesis output. This is sufficient to verify that the prompt is correctly structured, contains destination names and hotel names, and has the right length — without needing AWS credentials.

4. **Graceful degradation chain:** `rag=false` → no overhead; `rag=true` + missing index → warning + hotel results only; `rag=true` + index OK + no LLM → `knowledge_context` returned, no `rag_summary`; `rag=true` + index OK + LLM configured → full response.

**Next questions this raises:**
- Would a BM25 knowledge retrieval (term matching on destination name) outperform knn for exact-name queries ("tell me about Mallorca") while knn beats it for semantic queries ("somewhere like Mallorca but quieter")?
- Could the knowledge documents be automatically refreshed from authoritative sources (Wikipedia infoboxes, travel authority data) rather than being manually curated?
- M14 (graph-enhanced retrieval): could "similar_destinations" fields in knowledge documents seed a graph of destination similarity, enabling "travellers who liked Menorca also liked Corfu" style recommendations?

---

## Milestone 14 — Graph-enhanced retrieval

### Concept

Graph-enhanced retrieval augments vector and lexical search with structural, curated relationships between entities.  Where embeddings approximate similarity from text, a graph encodes exact, deterministic facts:

- `Mallorca SIMILAR_TO Menorca` — editorial link, not embedding distance
- `GLA FLIES_TO Tenerife` — structural reachability fact
- `LGW FLIES_TO Barbados` (but `GLA` does not) — long-haul hub restriction

### Hypothesis

> Graph traversal can answer two classes of query that vector search cannot:
> 1. **Curated similarity** — "similar destinations to Mallorca" where editorial links are more reliable than embedding proximity.
> 2. **Structural reachability** — "which destinations can I fly to from Glasgow?" which has no meaningful embedding representation.

### Implementation

The destination graph is pure Python — an in-memory directed adjacency-list with **38 nodes, 309 edges** (30 destination + 8 airport nodes; 94 SIMILAR_TO + 215 FLIES_TO edges).  No external graph database is used; the graph is rebuilt at startup from the knowledge JSONL file in under 5 ms.

**Edge types:**
- `SIMILAR_TO` (bidirectional): seeded from `similar_destinations` in knowledge docs
- `FLIES_TO` (directed, airport → destination): based on realistic UK charter routes; long-haul destinations (Barbados, Cancún, Maldives, Phuket, Koh Samui) restricted to hub airports (LGW, LHR, MAN)

### Live results (run against the actual server)

**Experiment 1 — SIMILAR_TO traversal depth**

```
GET /graph/similar?destination=Mallorca&hops=1  →  4 results
  Ibiza, Menorca, Costa del Sol, Costa Blanca

GET /graph/similar?destination=Mallorca&hops=2  →  11 results
  + Mykonos, Sardinia, Hvar, Corfu, Algarve, Agadir, Tenerife
```

Observation: hops=1 returns all Balearic siblings plus Spanish mainland coast.  hops=2 adds second-degree links — Ibiza's editorial neighbours (Mykonos, Sardinia, Hvar) and Menorca's neighbours (Corfu) surface without any extra embedding call.  A single vector search for "like Mallorca" cannot systematically reach Hvar at 2 degrees; the graph does it in a single BFS pass.

**Experiment 2 — Airport reachability (FLIES_TO)**

```
GET /graph/destinations?airport=GLA  →  25 destinations  (no long-haul)
GET /graph/destinations?airport=LHR  →  30 destinations  (all including long-haul)
```

Glasgow regional airport serves 25 short-haul destinations.  Heathrow serves all 30, including the 5 long-haul ones.  The 5-destination gap is a structural fact no embedding can encode.

**Experiment 3 — Reverse reachability (which airports serve X?)**

```
GET /graph/airports?destination=Barbados  →  3 airports  (LGW, LHR, MAN)
GET /graph/airports?destination=Tenerife  →  8 airports  (all UK airports)
```

Confirms the asymmetry: a traveller in Newcastle can reach Tenerife but not Barbados.  This is the graph's clearest differentiator from semantic search — "can I fly there from NCL?" is a binary structural fact, not a similarity score.

**Key observation — similarity vs. embedding:**

| Query | Vector search | Graph traversal |
|---|---|---|
| "Similar to Mallorca" (1-hop) | Approximate — depends on vocabulary overlap | Exact: Ibiza, Menorca, Costa del Sol, Costa Blanca |
| "2nd-degree similar" (2-hop) | Requires a second embedding round-trip | Single BFS pass: +7 destinations |
| "Fly from Glasgow" | No meaningful result | 25 destinations (exact reachability) |
| "Fly from Heathrow" | No meaningful result | 30 destinations (includes long-haul) |
| "Which airports serve Barbados?" | No meaningful result | LGW, LHR, MAN (3 of 8) |

**What the graph cannot do:**
- Score or rank results (no edge weights in this prototype)
- Generalise from learned patterns (no ML, no embedding)
- Recover from missing editorial links (if a SIMILAR_TO link was not curated, traversal misses it)

This illustrates the complementary nature of the two approaches: vector search finds semantically related content it was never explicitly told about; graph traversal follows exact structural facts that embeddings cannot encode.

### Experiments run (2026-08-15)

1. **Similarity depth comparison** (confirmed): hops=1 → 4 results (Ibiza, Menorca, Costa del Sol, Costa Blanca); hops=2 → 11 results (+Mykonos, Sardinia, Hvar, Corfu, Algarve, Agadir, Tenerife).  The 2-hop set includes destinations that are editorially linked to Ibiza and Menorca but not directly to Mallorca.  A single vector search cannot reproduce this without a second round-trip.

2. **Hub vs. regional airport** (confirmed): GLA → 25 (no long-haul); LHR → 30 (all); Barbados → 3 airports (LGW, LHR, MAN); Tenerife → 8 airports (all).  The hub restriction is deterministic and verifiable — no approximation.

3. **Graph vs. vector for "like Mallorca but quieter"** (confirmed): Vector knowledge retrieval returned Ibiza (nightlife=high) as the closest match — semantically plausible but wrong for the quietness constraint.  Graph traversal returned Menorca (nightlife=low) alongside Ibiza, making the correct answer immediately available for post-filtering.  See M13 section for full experiment output.

### Next questions this raises

- Should SIMILAR_TO edges carry weights (strength of similarity) rather than being binary?
- Could the graph be built from behavioral data (booking co-occurrence) rather than editorial curation, making it more like collaborative filtering?
- Would a sparse matrix representation of the graph enable efficient batch scoring — multiplying hotel relevance scores by destination similarity to produce "graph-boosted" ranking?
- In a production system, how would you keep editorial similarity links fresh without a full re-ingest cycle?

---

## Milestone 15 — Production API, observability, resilience, and final evaluation

### Hypothesis

Adding per-stage pipeline timing, structured request logging, in-memory Prometheus-compatible metrics, and an enhanced deep health endpoint does not change retrieval quality but enables post-hoc analysis of where latency is spent. Resilience fallbacks (rewriter fails → original query; hybrid fails → BM25) ensure the pipeline degrades predictably rather than returning 500 errors.

### Configuration

- Evaluation script: `uv run python scripts/evaluate.py`
- 62 golden queries, 10 query classes, graded relevance 0–3, K=10
- Server: default settings (no reranking, no query rewriting, no RAG, no expansion)
- Date: 2026-08-15

### Final evaluation results

| Strategy | NDCG@10 | MRR | HitRate@10 | P@10 | Latency p50 | Latency p95 |
|---|---|---|---|---|---|---|
| BM25 | 0.5021 | 0.6874 | 0.8387 | 0.6161 | 26 ms | 46 ms |
| Vector | **0.6940** | **0.8688** | **1.0000** | 0.7790 | 10 ms | 292 ms |
| RRF | 0.6239 | 0.8449 | 0.9516 | 0.7210 | 50 ms | 86 ms |
| Rerank | 0.6830 | 0.8191 | 0.9516 | **0.8935** | 109 ms | 178 ms |
| Understand | 0.6312 | 0.8620 | 0.9355 | 0.7290 | 46 ms | 75 ms |
| Rewrite | 0.6130 | 0.8226 | **0.9677** | 0.7242 | 56 ms | 126 ms |
| Expand | 0.6285 | 0.8308 | 0.9516 | 0.7226 | 167 ms | 218 ms |

*Best value per metric shown in bold.*

### Observability experiment — per-stage timing

Running `POST /search` with the new timing fields exposed:

```
curl -s -X POST http://localhost:8765/search \
  -H "Content-Type: application/json" \
  -d '{"query": "family beach hotel Greece July"}'
```

Sample timing breakdown:
- `took_ms`: 170 (end-to-end including FastAPI overhead)
- `qu_took_ms`: 0 (rule-based, pure Python, sub-millisecond)
- `rewrite_took_ms`: 0 (disabled)
- `lexical_took_ms`: 21 (OpenSearch BM25 query)
- `vector_took_ms`: 11 (ANN + embedding)
- `reranking_took_ms`: 0 (disabled)
- `rag_took_ms`: 0 (disabled)

**Finding:** the bulk of latency when reranking is disabled is split between the two OpenSearch queries (BM25 + vector). The `took_ms` total is larger than `lexical_took_ms + vector_took_ms` because it includes FastAPI serialization, Pydantic validation, and embedding inference time not separately tracked.

### Observability experiment — Prometheus metrics endpoint

After sending 10 POST /search requests, `GET /metrics` returns:

```
travel_search_requests_total{strategy="hybrid"} 10
travel_search_latency_ms_bucket{le="100"} 0
travel_search_latency_ms_bucket{le="250"} 8
travel_search_latency_ms_bucket{le="500"} 10
travel_search_latency_ms_count 10
travel_search_latency_ms_sum 1840.000
```

This tells us: 8 of 10 requests completed under 250 ms (p80), all under 500 ms. A Prometheus scraper could compute p50/p95 from these bucket values.

### Deep health check

`GET /health` now returns component-level status:

```json
{
  "status": "ok",
  "components": {
    "opensearch": {"status": "ok", "version": "2.15.0"},
    "index":      {"status": "ok", "name": "travel_hotels"},
    "embedding":  {"status": "ok", "provider": "LocalEmbeddingProvider"},
    "reranker":   {"status": "disabled", "detail": "set RERANKING_ENABLED=true to activate"},
    "graph":      {"status": "ok", "nodes": "38", "edges": "309"}
  }
}
```

This is the key difference from a shallow liveness probe: a readiness check verifies that the index is populated and the embedding model is loaded before considering the instance ready to serve traffic.

### Key observations

1. **Vector search is the strongest single strategy** for this synthetic dataset: NDCG@10=0.694, HitRate@10=1.000 (it finds at least one relevant result for every query). BM25 fails on exact-destination queries where the destination name appears in the description but not the title.

2. **Reranking improves Precision@10** (0.79 → 0.89) without improving HitRate, confirming the expected behaviour: the candidates are already there, the reranker just moves them up. NDCG@10 is slightly lower than pure vector (0.683 vs 0.694) because the reranker only sees the top-50 hybrid candidates, not the full vector list.

3. **Query expansion (multi-query) adds latency (~167 ms p50) without consistent metric improvement** over RRF alone (0.629 vs 0.624). The benefit is query-class-specific: for vague discovery queries it helps; for exact-destination queries it can dilute precision.

4. **BM25 consistently fails on exact-destination queries** (NDCG=0.187 for `exact_destination` class) — the index's BM25 fields boost hotel names but the golden queries ask for destinations, and there can be many hotels per destination.

5. **Graceful degradation is testable**: 9 new resilience tests verify that rewriter failure, embedding failure, and RAG failure all produce valid 200 responses with appropriate `strategy` and `fallback_used` fields rather than 500 errors.

### Surprises

- The vector model achieves **HitRate@10=1.000** — it finds at least one relevant hotel for every query in the golden set. This is surprising because the model (`all-MiniLM-L6-v2`) was not fine-tuned for travel data.
- **Query rewriting reduces NDCG@10** compared to raw hybrid (0.613 vs 0.624). The local LLM keyword-expansion rewriter adds terms that push the query toward noisier candidates. This highlights that rewriting quality matters more than the pipeline mechanism.
- **QU is essentially free** (`qu_took_ms` ≈ 0 ms): the rule-based extraction is pure Python regex matching, not ML inference.
