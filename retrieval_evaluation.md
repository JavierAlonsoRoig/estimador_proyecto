## Retrieval evaluation

Golden set: 5 queries | top_k=5 | recall_k=50 | rrf_k=60 | reranker=cross-encoder/ms-marco-MiniLM-L-6-v2 | runs per query=5

Latency = end-to-end as seen by the client (HTTP + server). Embed / Search / Rerank = server-side stage medians.

| Config | Search | Rerank | Precision@5 (mean) | Latency p50 (ms) | Latency p95 (ms) | Embed p50 | Search p50 | Rerank p50 |
|---|---|---|---|---|---|---|---|---|
| A | vector | no | 0.84 | 232 | 276 | 181 | 6.3 | - |
| B | hybrid | no | 0.88 | 232 | 264 | 175 | 9.3 | - |
| C | vector | yes | 0.84 | 3740 | 3892 | 176 | 6.8 | 3505 |
| D | hybrid | yes | 0.84 | 3976 | 4528 | 188 | 10.3 | 3723 |

### Precision@5 per query

| Query | A | B | C | D |
|---|---|---|---|---|
| q1-fashion-store | 0.8 | 1.0 | 0.8 | 0.8 |
| q2-marketplace | 0.8 | 0.8 | 0.8 | 0.8 |
| q3-telehealth | 1.0 | 1.0 | 1.0 | 1.0 |
| q4-predictive-maintenance | 0.6 | 0.6 | 0.6 | 0.6 |
| q5-warehouse-logistics | 1.0 | 1.0 | 1.0 | 1.0 |

### Top-5 budgets per query (relevant ones in bold)

**q1-fashion-store** (relevant: BUD-2024-005, BUD-2024-017)
- A: **005** **005** **017** 008 **005**
- B: **005** **005** **005** **017** **017**
- C: **005** **005** **005** **017** 008
- D: **005** **005** **005** **017** 008

**q2-marketplace** (relevant: BUD-2024-005, BUD-2024-006)
- A: **006** **006** **006** **006** 008
- B: **006** **006** **006** **006** 008
- C: **006** **006** **006** **006** 017
- D: **006** **006** **006** **006** 017

**q3-telehealth** (relevant: BUD-2024-009, BUD-2024-010)
- A: **010** **010** **009** **009** **010**
- B: **010** **010** **009** **009** **010**
- C: **009** **010** **010** **009** **010**
- D: **009** **010** **010** **009** **010**

**q4-predictive-maintenance** (relevant: BUD-2024-013, BUD-2024-015)
- A: **013** **013** **013** 010 010
- B: **013** **013** **013** 010 010
- C: **013** **013** **013** 003 003
- D: **013** **013** **013** 003 003

**q5-warehouse-logistics** (relevant: BUD-2024-007, BUD-2024-014)
- A: **014** **014** **014** **014** **007**
- B: **007** **014** **007** **014** **007**
- C: **014** **014** **014** **007** **014**
- D: **014** **014** **014** **007** **014**
