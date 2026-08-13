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
