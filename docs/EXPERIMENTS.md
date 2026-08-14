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
