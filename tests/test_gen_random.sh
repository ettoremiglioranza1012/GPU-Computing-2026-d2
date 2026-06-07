#!/bin/bash
# test_gen_random.sh — validate gen_random_csr output format
#
# Checks:
#   - binary exists (or builds it)
#   - header line is valid MatrixMarket coordinate real general
#   - declared NNZ matches actual data line count
#   - all data lines have exactly 3 fields (row col val)
#   - rows and cols are in bounds [1..N]
#   - no duplicate (row,col) entries within a row
#
# Usage (from repo root): bash tests/test_gen_random.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/bin/synth/gen_random_csr"

PASS=0; FAIL=0
ok()   { printf "  \033[32mPASS\033[0m %s\n" "$1"; ((PASS++)) || true; }
fail() { printf "  \033[31mFAIL\033[0m %s\n" "$1"; ((FAIL++)) || true; }

# ── Build if needed ──────────────────────────────────────────────────────────
if [ ! -f "$BIN" ]; then
    echo "[build] make synth ..."
    make -C "$ROOT" synth >/dev/null 2>&1 || { echo "Build failed"; exit 1; }
fi
[ -f "$BIN" ] && ok "binary exists: $BIN" || { fail "binary not found"; exit 1; }

# ── Helper: run the generator and validate its output ────────────────────────
# Write to a temp file to avoid SIGPIPE with "echo $large_output | head -1"
run_test() {
    local desc="$1"; shift
    local tmp
    tmp=$(mktemp /tmp/gen_random_test.XXXXXX)
    "$BIN" "$@" 2>/dev/null > "$tmp"

    # Line 1: %% banner (read only the first byte range — awk exits immediately)
    local line1
    line1=$(awk 'NR==1 {print; exit}' "$tmp")

    # Header line (first non-comment line): "N N NNZ"
    # Use awk with 'exit' to avoid writing large output through grep|head pipes,
    # which causes SIGPIPE in non-interactive scripts with pipefail.
    local header_line
    header_line=$(awk '!/^%/ {print; exit}' "$tmp")
    local declared_N declared_NNZ
    declared_N=$(awk '{print $1}' <<< "$header_line")
    declared_NNZ=$(awk '{print $3}' <<< "$header_line")

    # Data lines (non-comment, skip first non-comment header line)
    local data_tmp
    data_tmp=$(mktemp /tmp/gen_random_data.XXXXXX)
    grep -v '^%' "$tmp" | tail -n +2 > "$data_tmp"

    local actual_NNZ
    actual_NNZ=$(wc -l < "$data_tmp" | tr -d ' ')

    # Check 1: MTX banner present
    [[ "$line1" == %%MatrixMarket* ]] \
        && ok "$desc: MTX banner present" \
        || fail "$desc: MTX banner missing"

    # Check 2: declared NNZ matches actual data lines
    [ "$declared_NNZ" -eq "$actual_NNZ" ] \
        && ok "$desc: declared NNZ ($declared_NNZ) == actual data lines" \
        || fail "$desc: NNZ mismatch (declared $declared_NNZ, actual $actual_NNZ)"

    # Check 3: all data lines have exactly 3 fields
    local bad3
    bad3=$(awk 'NF != 3 {c++} END {print c+0}' "$data_tmp")
    [ "$bad3" -eq 0 ] \
        && ok "$desc: all data lines have 3 fields" \
        || fail "$desc: $bad3 data lines with wrong field count"

    # Check 4: row and col indices in [1..N]
    local bad_bounds
    bad_bounds=$(awk -v N="$declared_N" \
        '$1 < 1 || $1 > N || $2 < 1 || $2 > N {c++} END {print c+0}' "$data_tmp")
    [ "$bad_bounds" -eq 0 ] \
        && ok "$desc: all indices in [1..$declared_N]" \
        || fail "$desc: $bad_bounds out-of-bound indices"

    # Check 5: no duplicate (row,col) pairs
    local dups
    dups=$(awk '{print $1, $2}' "$data_tmp" | sort | uniq -d | wc -l | tr -d ' ')
    [ "$dups" -eq 0 ] \
        && ok "$desc: no duplicate (row,col) pairs" \
        || fail "$desc: $dups duplicate pairs found"

    rm -f "$tmp" "$data_tmp"
}

echo ""
echo "=== gen_random_csr format tests ==="

run_test "small  (N=100  nnz/row=5  seed=1)" \
    --rows 100   --nnz-per-row 5  --seed 1

run_test "medium (N=1000 nnz/row=10 seed=42)" \
    --rows 1000  --nnz-per-row 10 --seed 42

run_test "sparse (N=500  nnz/row=1  seed=7)" \
    --rows 500   --nnz-per-row 1  --seed 7

run_test "dense  (N=50   nnz/row=50 seed=99)" \
    --rows 50    --nnz-per-row 50 --seed 99

echo ""
printf "Results: %d passed, %d failed\n" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
