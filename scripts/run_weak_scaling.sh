#!/bin/bash
# run_weak_scaling.sh — submit 3 weak-scaling jobs to SLURM
#
# Problem size scales with P: each rank always processes N_BASE rows (K nnz/row).
#   P=1 : N_BASE   rows total  → 5M NNZ/rank
#   P=2 : 2*N_BASE rows total  → 5M NNZ/rank
#   P=4 : 4*N_BASE rows total  → 5M NNZ/rank
#
# Per-rank compute is constant; Allgatherv cost grows O(N_BASE*P*sizeof(float)),
# exposing the 1D partition communication bottleneck under weak scaling.
#
# Usage (from repo root): bash scripts/run_weak_scaling.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(realpath "$SCRIPT_DIR/..")"
JOB_SCRIPT="$SCRIPT_DIR/run_weak_scaling_job.sh"
BIN="$ROOT/bin/MPI/spmv_mpi_baseline"
GEN="$ROOT/bin/synth/gen_random_csr"

N_BASE=500000   # rows per rank
K=10            # nnz per row (→ 5M NNZ per rank)
SEED=42

if [ ! -f "$BIN" ]; then
    echo "ERROR: binary not found — run 'make mpi' first" >&2; exit 1
fi
if [ ! -f "$GEN" ]; then
    echo "ERROR: generator not found — run 'make synth' first" >&2; exit 1
fi

mkdir -p "$ROOT/outputs"

submitted=0
for NP in 1 2 4; do
    NROWS=$(( NP * N_BASE ))
    NNZ=$(( NROWS * K ))

    NP="$NP" N_BASE="$N_BASE" K="$K" SEED="$SEED" sbatch \
        --ntasks="$NP" \
        --gres="gpu:$NP" \
        --job-name="spmv_weak_P${NP}" \
        --output="$ROOT/outputs/weak_P${NP}_%j.out" \
        --error="$ROOT/outputs/weak_P${NP}_%j.err" \
        "$JOB_SCRIPT"

    echo "[submit] P=${NP}  ${NROWS} rows  ${NNZ} NNZ total  (~${NNZ} NNZ / ${NP} ranks = $((NNZ / NP)) NNZ/rank)"
    (( submitted++ )) || true
done

echo ""
echo "Submitted ${submitted} weak-scaling jobs."
echo "Monitor : squeue -u \$USER"
echo "Outputs : $ROOT/outputs/weak_P*.out"
