#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --job-name=spmv_mpi
#SBATCH --output=outputs/spmv_mpi_%j.out
#SBATCH --error=outputs/spmv_mpi_%j.err
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=32G

# ── Modules ──────────────────────────────────────────────────────────────────
module load CUDA/12.1.1
module load OpenMPI

# ── Config ────────────────────────────────────────────────────────────────────
# Override from command line: MATRIX=Data/bone010/bone010.mtx NP=2 sbatch ...
MATRIX="${MATRIX:-Data/rajat31/rajat31.mtx}"
NP="${NP:-4}"
BIN="bin/MPI/spmv_mpi_baseline"

echo "========================================"
echo "SLURM job  : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Binary     : $BIN"
echo "Matrix     : $MATRIX"
echo "MPI ranks  : $NP"
echo "GPUs       : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -4)"
echo "========================================"

if [ ! -f "$BIN" ]; then
    echo "ERROR: binary not found — run 'make mpi' first" >&2
    exit 1
fi

if [ ! -f "$MATRIX" ]; then
    echo "ERROR: matrix not found: $MATRIX — run 'make data' first" >&2
    exit 1
fi

mpirun -np "$NP" "$BIN" "$MATRIX"
