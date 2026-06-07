# GPU Computing 2026 — Deliverable 2

**Distributed Sparse Matrix-Vector Multiplication (SpMV) with MPI + CUDA-aware**  
Prof. Flavio Vella · A.Y. 2025–2026 · DISI Cluster (NVIDIA A30, sm_80)

---

## Build

```bash
module load CUDA/12.1.1
module load OpenMPI       # or whichever MPI module is available
make mpi                  # builds bin/MPI/spmv_mpi_baseline
make cpu                  # builds bin/CPU/spmv_cpu_ref
```

## Run (single node, correctness check)

```bash
mpirun -np 2 ./bin/MPI/spmv_mpi_baseline Data/bone010.mtx
```

## SLURM (scaling experiments)

```bash
sbatch scripts/run_strong_scaling.sh
sbatch scripts/run_weak_scaling.sh
```

## Directory layout

```
MPI/          ← main distributed SpMV programs
GPU/          ← local GPU kernels (TPV, reused from D1)
CPU/          ← sequential reference (correctness validation only)
include/      ← mtx_io.h (MTX reader), csr_utils.h (COO→CSR, local index)
TIMER_LIB/    ← gettimeofday timing library
scripts/      ← SLURM launchers, parse_results.py, plot scripts
synthetic/    ← random sparse matrix generator (weak-scaling experiments)
Data/         ← SuiteSparse matrices (download with: make data)
outputs/      ← SLURM .out/.err files
results_tables/ ← parsed CSVs
assets/       ← generated plots
report/       ← LaTeX source
docs/
└── report_insights_d2.md  ← accumulated results (update after every run)
```

## Algorithm (1D cyclic partition)

1. `MPI_Init` → each rank binds to GPU `rank % ngpus`
2. Rank 0 reads full MTX file (via `mtx_io.h`), distributes rows: `owner(i) = i mod P`
3. Each rank converts local COO → local CSR (`csr_utils.h`)
4. Each rank generates full x with fixed seed `srand(42)` (same values, no Allgather needed for random x)
5. H2D transfer of local CSR + full x
6. TPV kernel on local sub-matrix → local y
7. D2H → `MPI_Gather` to rank 0 for correctness validation

## Datasets

Same 10 SuiteSparse matrices as D1. Download:
```bash
make data
```

## Deadline

June 29, 2026 at 23:59 (first exam session).
Submit PDF report via Google Form with UNITN email.
