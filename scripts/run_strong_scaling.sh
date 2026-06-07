#!/bin/bash
# run_strong_scaling.sh — submit strong-scaling jobs to SLURM
#
# Submits 30 jobs: 10 SuiteSparse matrices × P ∈ {1, 2, 4}
# Each job calls run_mpi_baseline.sh with command-line overrides for
# --ntasks and --gres (which take precedence over the #SBATCH defaults
# inside that script).
#
# Usage (from repo root):
#   bash scripts/run_strong_scaling.sh
#
# Prerequisites:
#   make mpi          — binary must exist
#   make data         — matrices must be in Data/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(realpath "$SCRIPT_DIR/..")"
BIN="$ROOT/bin/MPI/spmv_mpi_baseline"
BASELINE_SCRIPT="$SCRIPT_DIR/run_mpi_baseline.sh"

if [ ! -f "$BIN" ]; then
    echo "ERROR: binary not found — run 'make mpi' first" >&2
    exit 1
fi

mkdir -p "$ROOT/outputs"

# All 10 SuiteSparse matrices (same set as D1)
MATRICES=(
    bone010
    ldoor
    Rucci1
    nlpkkt80
    ASIC_680ks
    rajat31
    boyd2
    eu-2005
    webbase-1M
    hollywood-2009
)

P_VALUES=(1 2 4)

submitted=0
skipped=0

for name in "${MATRICES[@]}"; do
    mtx="$ROOT/Data/${name}/${name}.mtx"
    if [ ! -f "$mtx" ]; then
        echo "[SKIP] $name — matrix not found (run 'make data' first)"
        (( skipped++ )) || true
        continue
    fi
    for NP in "${P_VALUES[@]}"; do
        MATRIX="$mtx" NP="$NP" sbatch \
            --ntasks="$NP" \
            --gres="gpu:$NP" \
            --mem=32G \
            --job-name="ss_${name}_P${NP}" \
            --output="$ROOT/outputs/strong_${name}_P${NP}_%j.out" \
            --error="$ROOT/outputs/strong_${name}_P${NP}_%j.err" \
            "$BASELINE_SCRIPT"

        echo "[submit] P=${NP}  ${name}"
        (( submitted++ )) || true
    done
done

echo ""
echo "Submitted ${submitted} jobs, skipped ${skipped} matrices."
echo "Monitor : squeue -u \$USER"
echo "Outputs : $ROOT/outputs/strong_*.out"
