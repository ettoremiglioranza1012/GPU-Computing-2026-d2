#!/bin/bash
# run_tests.sh — build and run all locally executable tests
#
# What this covers:
#   test_index_math     pure-C unit tests for all index math formulas
#   test_mpi_spmv_cpu   MPI integration test (full distribute/allgather/compute/
#                       reconstruct cycle, CPU-only — exercises every MPI call)
#   test_mtx_reader     mtx_io.h parser (general/symmetric/pattern MTX variants)
#   test_gen_random.sh  gen_random_csr output format validation
#
# What is NOT covered (requires GPU on the cluster):
#   - CUDA TPV kernel correctness
#   - CUDA-aware MPI (device pointers in Allgatherv)
#   - make mpi / make gpu compilation
#
# Usage (from repo root): bash tests/run_tests.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PASS=0; FAIL=0

run() {
    local name="$1"; shift
    printf "%-55s " "$name"
    if "$@" >/tmp/_test_out.txt 2>&1; then
        printf "\033[32mPASS\033[0m\n"
        ((PASS++)) || true
    else
        printf "\033[31mFAIL\033[0m\n"
        sed 's/^/    /' /tmp/_test_out.txt
        ((FAIL++)) || true
    fi
}

# ── Build ────────────────────────────────────────────────────────────────────
echo "=== Build ==="
printf "%-55s " "make synth"
if make synth >/tmp/_build_synth.txt 2>&1; then
    printf "\033[32mOK\033[0m\n"
else
    printf "\033[31mFAILED\033[0m\n"; cat /tmp/_build_synth.txt; exit 1
fi

printf "%-55s " "make -C tests"
if make -C tests >/tmp/_build_tests.txt 2>&1; then
    printf "\033[32mOK\033[0m\n"
else
    printf "\033[31mFAILED\033[0m\n"; cat /tmp/_build_tests.txt; exit 1
fi

# ── Run tests ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Tests ==="

run "test_index_math (pure C)"              ./tests/test_index_math
run "test_mtx_reader (mtx_io.h)"            ./tests/test_mtx_reader
run "test_mpi_spmv_cpu P=1"                 mpirun -np 1 ./tests/test_mpi_spmv_cpu
run "test_mpi_spmv_cpu P=2"                 mpirun -np 2 ./tests/test_mpi_spmv_cpu
run "test_mpi_spmv_cpu P=4"                 mpirun -np 4 ./tests/test_mpi_spmv_cpu
run "test_gen_random (format validation)"   bash tests/test_gen_random.sh

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
printf "  Results: \033[32m%d passed\033[0m  \033[31m%d failed\033[0m\n" "$PASS" "$FAIL"
echo "══════════════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Note: GPU kernel and CUDA-aware MPI can only be tested on the cluster."
    exit 1
fi
