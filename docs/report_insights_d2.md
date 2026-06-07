# D2 Results — report_insights_d2.md
**Single source of truth for all D2 measurements. Update after every cluster run.**

---

## §1 — Strong Scaling Results

> Fill after Phase 3 runs.

### Template row format (one per run):
`matrix | P | rank | local_rows | local_nnz | kernel_time_ms | bandwidth_GBs | gflops`

| matrix | P | rank | local_rows | local_nnz | kernel_time_ms | bandwidth_GBs | gflops | correctness |
|--------|---|------|-----------|-----------|----------------|---------------|--------|-------------|
| — | — | — | — | — | — | — | — | — |

### Speedup table (T(1) / T(P), averaged across ranks):

| matrix | T(1) ms | T(2) ms | T(4) ms | Speedup(2) | Speedup(4) | Efficiency(2) | Efficiency(4) |
|--------|---------|---------|---------|-----------|-----------|--------------|--------------|
| bone010 | — | — | — | — | — | — | — |
| eu-2005 | — | — | — | — | — | — | — |
| rajat31 | — | — | — | — | — | — | — |
| hollywood-2009 | — | — | — | — | — | — | — |

---

## §2 — Weak Scaling Results

> Fill after Phase 4 runs.

| synthetic matrix | P | N (rows) | NNZ | kernel_time_ms | efficiency |
|-----------------|---|----------|-----|----------------|-----------|
| random_500k_k10 | 1 | 500K | 5M | — | — |
| random_1m_k10 | 2 | 1M | 10M | — | — |
| random_2m_k10 | 4 | 2M | 20M | — | — |

---

## §3 — NNZ Balance per Rank (Cyclic Partition)

> Cyclic partition distributes rows {r, r+P, r+2P, ...} to rank r.
> For regular matrices (uniform NNZ/row), balance is perfect.
> For irregular matrices, balance depends on row-length distribution.

| matrix | P | rank 0 NNZ | rank 1 NNZ | rank 2 NNZ | rank 3 NNZ | imbalance % |
|--------|---|-----------|-----------|-----------|-----------|------------|
| — | — | — | — | — | — | — |

---

## §4 — Correctness Validation Log

| matrix | P | result | max_abs_error | tol | notes |
|--------|---|--------|--------------|-----|-------|
| — | — | — | — | — | — |

---

## §5 — Observations and Report Narrative

> Write analysis notes here as you collect data. These become the Discussion section.

### When does 1D cyclic work well?
_To fill._

### When is it bound by the interconnect?
_To fill._

### Comparison vs professor's baseline:
_To fill._

---

*Last updated: —*
