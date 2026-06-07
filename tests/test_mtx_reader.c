/* test_mtx_reader.c — tests mtx_io.h (mtx_read_coo) with three MTX variants
 *
 * Creates known .mtx files in /tmp/, reads them, verifies COO arrays.
 *
 * Tested variants:
 *   1. general real      — no expansion, values present
 *   2. symmetric real    — off-diagonal entries mirrored, diagonal kept once
 *   3. pattern (general) — values absent, filled with 1.0
 *
 * Compile: gcc -O0 -g -I../include -o test_mtx_reader test_mtx_reader.c -lm
 * Run:     ./test_mtx_reader
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mtx_io.h"

#ifdef _WIN32
#define COL_PASS ""
#define COL_FAIL ""
#define COL_RST  ""
#else
#define COL_PASS "\033[32m"
#define COL_FAIL "\033[31m"
#define COL_RST  "\033[0m"
#endif

static int nerr = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("  " COL_FAIL "FAIL" COL_RST " %s  (line %d)\n",           \
                   msg, __LINE__);                                             \
            nerr++;                                                            \
        } else {                                                               \
            printf("  " COL_PASS "PASS" COL_RST " %s\n", msg);                \
        }                                                                      \
    } while (0)

/* Write a string to a temp file and return the path */
static const char *write_mtx(const char *filename, const char *content)
{
    static char path[256];
    snprintf(path, sizeof(path), "/tmp/%s", filename);
    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen"); exit(1); }
    fputs(content, f);
    fclose(f);
    return path;
}

/* Sort two parallel int arrays by key (row), for comparison */
static void sort_pairs(int *a, int *b, float *v, int n)
{
    /* bubble sort (small n) */
    for (int i = 0; i < n-1; i++)
        for (int j = i+1; j < n; j++)
            if (a[j] < a[i] || (a[j] == a[i] && b[j] < b[i])) {
                int   ta = a[i]; a[i] = a[j]; a[j] = ta;
                int   tb = b[i]; b[i] = b[j]; b[j] = tb;
                float tv = v[i]; v[i] = v[j]; v[j] = tv;
            }
}

/* ── Test 1: general real ──────────────────────────────────────────────────── */
static void test_general_real(void)
{
    printf("\n=== MTX general real ===\n");

    /* 3×3 matrix, 4 nonzeros (1-indexed in file, 0-indexed after read) */
    const char *mtx =
        "%%MatrixMarket matrix coordinate real general\n"
        "% comment line\n"
        "3 3 4\n"
        "1 1 1.0\n"
        "1 3 2.0\n"
        "2 2 3.0\n"
        "3 2 4.0\n";

    const char *path = write_mtx("test_general.mtx", mtx);

    int nrows, ncols, nnz;
    int   *rows = NULL; int   *cols = NULL; float *vals = NULL;
    mtx_read_coo(path, &nrows, &ncols, &nnz, &rows, &cols, &vals);

    CHECK(nrows == 3, "nrows == 3");
    CHECK(ncols == 3, "ncols == 3");
    CHECK(nnz   == 4, "nnz == 4");

    sort_pairs(rows, cols, vals, nnz);

    /* Expected (0-indexed): (0,0,1), (0,2,2), (1,1,3), (2,1,4) */
    CHECK(rows[0]==0 && cols[0]==0 && fabsf(vals[0]-1.f)<1e-6f, "entry (0,0)=1.0");
    CHECK(rows[1]==0 && cols[1]==2 && fabsf(vals[1]-2.f)<1e-6f, "entry (0,2)=2.0");
    CHECK(rows[2]==1 && cols[2]==1 && fabsf(vals[2]-3.f)<1e-6f, "entry (1,1)=3.0");
    CHECK(rows[3]==2 && cols[3]==1 && fabsf(vals[3]-4.f)<1e-6f, "entry (2,1)=4.0");

    free(rows); free(cols); free(vals);
}

/* ── Test 2: symmetric real ────────────────────────────────────────────────── */
static void test_symmetric_real(void)
{
    printf("\n=== MTX symmetric real (off-diagonal mirroring) ===\n");

    /* 4×4 symmetric, 3 stored entries:
     *   (1,1,5)  — diagonal, kept once
     *   (2,1,1)  — off-diagonal, mirrored to (1,2,1)
     *   (3,2,2)  — off-diagonal, mirrored to (2,3,2)
     * Expected 5 COO entries after expansion.
     */
    const char *mtx =
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "4 4 3\n"
        "1 1 5.0\n"
        "2 1 1.0\n"
        "3 2 2.0\n";

    const char *path = write_mtx("test_symmetric.mtx", mtx);

    int nrows, ncols, nnz;
    int   *rows = NULL; int *cols = NULL; float *vals = NULL;
    mtx_read_coo(path, &nrows, &ncols, &nnz, &rows, &cols, &vals);

    CHECK(nrows == 4, "nrows == 4");
    CHECK(ncols == 4, "ncols == 4");
    CHECK(nnz   == 5, "nnz == 5 (1 diagonal + 2 off-diag pairs)");

    /* Count how many (r,c) vs (c,r) pairs exist */
    int found_00 = 0, found_10 = 0, found_01 = 0, found_21 = 0, found_12 = 0;
    for (int i = 0; i < nnz; i++) {
        if (rows[i]==0 && cols[i]==0 && fabsf(vals[i]-5.f)<1e-6f) found_00=1;
        if (rows[i]==1 && cols[i]==0 && fabsf(vals[i]-1.f)<1e-6f) found_10=1;
        if (rows[i]==0 && cols[i]==1 && fabsf(vals[i]-1.f)<1e-6f) found_01=1;
        if (rows[i]==2 && cols[i]==1 && fabsf(vals[i]-2.f)<1e-6f) found_21=1;
        if (rows[i]==1 && cols[i]==2 && fabsf(vals[i]-2.f)<1e-6f) found_12=1;
    }
    CHECK(found_00, "diagonal (0,0)=5.0 present");
    CHECK(found_10, "original off-diag (1,0)=1.0 present");
    CHECK(found_01, "mirrored off-diag (0,1)=1.0 present");
    CHECK(found_21, "original off-diag (2,1)=2.0 present");
    CHECK(found_12, "mirrored off-diag (1,2)=2.0 present");

    free(rows); free(cols); free(vals);
}

/* ── Test 3: pattern general ───────────────────────────────────────────────── */
static void test_pattern_general(void)
{
    printf("\n=== MTX pattern general (values default to 1.0) ===\n");

    /* 3×3 pattern matrix, 3 nonzeros — no value column */
    const char *mtx =
        "%%MatrixMarket matrix coordinate pattern general\n"
        "3 3 3\n"
        "1 2\n"
        "2 3\n"
        "3 1\n";

    const char *path = write_mtx("test_pattern.mtx", mtx);

    int nrows, ncols, nnz;
    int   *rows = NULL; int *cols = NULL; float *vals = NULL;
    mtx_read_coo(path, &nrows, &ncols, &nnz, &rows, &cols, &vals);

    CHECK(nrows == 3, "nrows == 3");
    CHECK(ncols == 3, "ncols == 3");
    CHECK(nnz   == 3, "nnz == 3");

    sort_pairs(rows, cols, vals, nnz);

    /* Expected (0-indexed): (0,1,1), (1,2,1), (2,0,1) */
    CHECK(rows[0]==0 && cols[0]==1 && fabsf(vals[0]-1.f)<1e-6f, "entry (0,1)=1.0 (pattern)");
    CHECK(rows[1]==1 && cols[1]==2 && fabsf(vals[1]-1.f)<1e-6f, "entry (1,2)=1.0 (pattern)");
    CHECK(rows[2]==2 && cols[2]==0 && fabsf(vals[2]-1.f)<1e-6f, "entry (2,0)=1.0 (pattern)");

    free(rows); free(cols); free(vals);
}

/* ── Driver ───────────────────────────────────────────────────────────────── */
int main(void)
{
    printf("=== MTX Reader Tests (mtx_io.h) ===\n");
    test_general_real();
    test_symmetric_real();
    test_pattern_general();

    printf("\n%s — %d test(s) failed\n",
           nerr == 0 ? COL_PASS "ALL PASSED" COL_RST
                     : COL_FAIL "FAILURES"   COL_RST,
           nerr);
    return nerr ? 1 : 0;
}
