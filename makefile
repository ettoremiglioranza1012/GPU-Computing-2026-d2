# =============================================================================
# GPU Computing 2026 — SpMV Deliverable 2
# Distributed SpMV with MPI + CUDA (NVIDIA A30, sm_80)
# =============================================================================

CC       = gcc
NVCC     = nvcc
OPT      = -O3
GPU_ARCH = --gpu-architecture=sm_80
GPU_FLAGS = -m64

# MPI flags — nvcc uses -Xlinker instead of GCC's -Wl, syntax
MPI_CFLAGS := $(shell mpicc --showme:compile 2>/dev/null)
MPI_LFLAGS := $(shell mpicc --showme:link 2>/dev/null | sed 's/-Wl,/-Xlinker /g')

# Directory layout
INC_DIR   := include
MPI_DIR   := MPI
GPU_DIR   := GPU
SYN_DIR   := synthetic
BIN_MPI   := bin/MPI
BIN_GPU   := bin/GPU
BIN_SYN   := bin/synth
SCRIPTS   := scripts
DATA_DIR  := Data

# =============================================================================
# Default target
# =============================================================================
.PHONY: all mpi gpu synth data run clean help

all: mpi gpu synth

# =============================================================================
# MPI + CUDA binary  (requires: module load CUDA/12.1.1 + module load OpenMPI)
# =============================================================================
mpi: $(BIN_MPI)/spmv_mpi_baseline

$(BIN_MPI)/spmv_mpi_baseline: $(MPI_DIR)/spmv_mpi_baseline.cu $(INC_DIR)/mtx_io.h
	@mkdir -p $(BIN_MPI)
	$(NVCC) $(GPU_ARCH) $(GPU_FLAGS) -I$(INC_DIR) -o $@ $< $(MPI_CFLAGS) $(MPI_LFLAGS)

# =============================================================================
# Standalone GPU binary  (single-rank correctness check, no MPI)
# =============================================================================
gpu: $(BIN_GPU)/spmv_tpv_local

$(BIN_GPU)/spmv_tpv_local: $(GPU_DIR)/spmv_tpv_local.cu $(INC_DIR)/mtx_io.h
	@mkdir -p $(BIN_GPU)
	$(NVCC) $(GPU_ARCH) $(GPU_FLAGS) -I$(INC_DIR) -o $@ $<

# =============================================================================
# Synthetic matrix generator  (weak scaling experiments)
# =============================================================================
synth: $(BIN_SYN)/gen_random_csr

$(BIN_SYN)/gen_random_csr: $(SYN_DIR)/gen_random_csr.c
	@mkdir -p $(BIN_SYN)
	$(CC) $< -o $@ $(OPT) -lm

# =============================================================================
# Submit MPI baseline job to SLURM
# Override matrix and rank count: make run MATRIX=Data/bone010/bone010.mtx NP=2
# =============================================================================
MATRIX ?= Data/rajat31/rajat31.mtx
NP     ?= 4

run: mpi
	MATRIX=$(MATRIX) NP=$(NP) sbatch scripts/run_mpi_baseline.sh

# =============================================================================
# Data download
# =============================================================================
data:
	@mkdir -p $(DATA_DIR) outputs results_tables assets
	bash $(SCRIPTS)/download_data.sh

# =============================================================================
# Clean
# =============================================================================
clean:
	rm -rf bin/

clean_outputs:
	rm -f outputs/*.out outputs/*.err outputs/*.txt

clean_results:
	rm -f results_tables/*.csv assets/*.png

# =============================================================================
# Help
# =============================================================================
help:
	@echo ""
	@echo "GPU Computing 2026 — SpMV Deliverable 2"
	@echo ""
	@echo "Build targets:"
	@echo "  all    build mpi + gpu + synth (default)"
	@echo "  mpi    build bin/MPI/spmv_mpi_baseline  [needs CUDA + OpenMPI modules]"
	@echo "  gpu    build bin/GPU/spmv_tpv_local      [needs CUDA module]"
	@echo "  synth  build bin/synth/gen_random_csr"
	@echo ""
	@echo "Run (cluster):"
	@echo "  run               sbatch scripts/run_mpi_baseline.sh (P=4, rajat31)"
	@echo "  run MATRIX=Data/bone010/bone010.mtx NP=2   override matrix/ranks"
	@echo ""
	@echo "Data:"
	@echo "  data   download all SuiteSparse matrices into Data/"
	@echo ""
	@echo "Clean:"
	@echo "  clean         remove all binaries and object files"
	@echo "  clean_outputs remove SLURM output files"
	@echo "  clean_results remove CSVs and plots"
	@echo ""
