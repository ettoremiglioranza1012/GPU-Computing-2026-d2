#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --job-name=spmv_weak
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:05:00
# --ntasks and --gres are overridden by run_weak_scaling.sh command-line args

# ── Modules ──────────────────────────────────────────────────────────────────
module load CUDA/12.1.1
module load OpenMPI

# ── Config (set by run_weak_scaling.sh via environment) ──────────────────────
NP="${NP:-1}"
N_BASE="${N_BASE:-500000}"   # rows per rank
K="${K:-10}"                 # nnz per row
SEED="${SEED:-42}"

NROWS=$(( NP * N_BASE ))
NNZ_TOTAL=$(( NROWS * K ))

BIN="bin/MPI/spmv_mpi_baseline"
GEN="bin/synth/gen_random_csr"
MTX="Data/synthetic/weak_N${NROWS}_K${K}_P${NP}.mtx"

mkdir -p Data/synthetic

echo "========================================"
echo "SLURM job   : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "P (ranks)   : $NP"
echo "Rows/rank   : $N_BASE"
echo "Total rows  : $NROWS"
echo "NNZ/row     : $K"
echo "Total NNZ   : $NNZ_TOTAL"
echo "GPUs        : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -"$NP")"
echo "========================================"

if [ ! -f "$GEN" ]; then
    echo "ERROR: $GEN not found — run 'make synth' first" >&2; exit 1
fi
if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN not found — run 'make mpi' first" >&2; exit 1
fi

echo "[gen] generating matrix → $MTX"
"$GEN" --rows "$NROWS" --nnz-per-row "$K" --seed "$SEED" > "$MTX"
echo "[gen] done ($(wc -l < "$MTX") lines)"

mpirun -np "$NP" "$BIN" "$MTX"

rm -f "$MTX"
echo "[done] matrix removed"
