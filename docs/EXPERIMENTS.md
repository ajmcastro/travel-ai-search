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

*First experiment will be recorded at Milestone 3 (BM25 baseline).*
