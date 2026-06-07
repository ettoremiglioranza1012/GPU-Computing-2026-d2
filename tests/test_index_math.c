/* test_index_math.c — unit tests for all pure-C index formulas in spmv_mpi_baseline
 *
 * Tests three things that are easy to get wrong:
 *   1. local_row_count(nrows, rank, P)          — how many local rows each rank owns
 *   2. x block-distribution math               — x_blk, x_start, x_count
 *   3. y reconstruction formula                 — full_y[r + j*P] = gathered_y[displs[r]+j]
 *   4. global→local row remapping               — local_row = global_row / P
 *
 * Compile: gcc -O0 -g -o test_index_math test_index_math.c
 * Run:     ./test_index_math
 */

#include <stdio.h>
#include <string.h>

/* ANSI colours only when stdout is a terminal */
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

/* Exact copy of helper from spmv_mpi_baseline.cu */
static int local_row_count(int nrows, int rank, int P)
{
    if (rank >= nrows) return 0;
    return (nrows - rank - 1) / P + 1;
}

/* ── Test 1: local_row_count ──────────────────────────────────────────────── */
static void test_local_row_count(void)
{
    printf("\n=== local_row_count(nrows, rank, P) ===\n");

    /* 6 rows, P=2: rows {0,2,4}→rank0, {1,3,5}→rank1 */
    CHECK(local_row_count(6, 0, 2) == 3, "nrows=6 rank=0 P=2 → 3");
    CHECK(local_row_count(6, 1, 2) == 3, "nrows=6 rank=1 P=2 → 3");

    /* 7 rows, P=2: {0,2,4,6}→rank0 (4), {1,3,5}→rank1 (3) */
    CHECK(local_row_count(7, 0, 2) == 4, "nrows=7 rank=0 P=2 → 4 (uneven split)");
    CHECK(local_row_count(7, 1, 2) == 3, "nrows=7 rank=1 P=2 → 3 (uneven split)");

    /* 6 rows, P=4: {0,4}→rank0 (2), {1,5}→rank1 (2), {2}→rank2 (1), {3}→rank3 (1) */
    CHECK(local_row_count(6, 0, 4) == 2, "nrows=6 rank=0 P=4 → 2");
    CHECK(local_row_count(6, 1, 4) == 2, "nrows=6 rank=1 P=4 → 2");
    CHECK(local_row_count(6, 2, 4) == 1, "nrows=6 rank=2 P=4 → 1");
    CHECK(local_row_count(6, 3, 4) == 1, "nrows=6 rank=3 P=4 → 1");

    /* P=1: rank 0 owns everything */
    CHECK(local_row_count(100, 0, 1) == 100, "P=1: rank 0 owns all 100 rows");

    /* rank index >= nrows → 0 (matrix smaller than P) */
    CHECK(local_row_count(2, 5, 4) == 0, "rank >= nrows → 0");

    /* Evenly divisible: 8 rows, P=4 → 2 per rank */
    CHECK(local_row_count(8, 0, 4) == 2, "nrows=8 rank=0 P=4 → 2");
    CHECK(local_row_count(8, 1, 4) == 2, "nrows=8 rank=1 P=4 → 2");
    CHECK(local_row_count(8, 2, 4) == 2, "nrows=8 rank=2 P=4 → 2");
    CHECK(local_row_count(8, 3, 4) == 2, "nrows=8 rank=3 P=4 → 2");

    /* Total always sums to nrows */
    int total = 0;
    for (int r = 0; r < 4; r++) total += local_row_count(10, r, 4);
    CHECK(total == 10, "sum of local_row_count across P=4 ranks == 10");
}

/* ── Test 2: x block-distribution ─────────────────────────────────────────── */
static void test_x_block_distribution(void)
{
    printf("\n=== x block-distribution (x_blk, x_start, x_count) ===\n");

    /* ncols=10, P=4: blk=ceil(10/4)=3 */
    {
        int ncols = 10, P = 4;
        int x_blk = (ncols + P - 1) / P;
        CHECK(x_blk == 3, "ncols=10 P=4: blk=3");

        /* Helper that mirrors the code exactly */
        #define XCOUNT(rank) \
            ((rank)*x_blk < ncols \
             ? ((rank)*x_blk + x_blk <= ncols ? x_blk : ncols - (rank)*x_blk) \
             : 0)

        CHECK((0)*x_blk == 0 && XCOUNT(0) == 3, "rank=0: start=0 count=3");
        CHECK((1)*x_blk == 3 && XCOUNT(1) == 3, "rank=1: start=3 count=3");
        CHECK((2)*x_blk == 6 && XCOUNT(2) == 3, "rank=2: start=6 count=3");
        CHECK((3)*x_blk == 9 && XCOUNT(3) == 1, "rank=3: start=9 count=1 (tail)");

        int total = XCOUNT(0)+XCOUNT(1)+XCOUNT(2)+XCOUNT(3);
        CHECK(total == ncols, "sum of x_count across all ranks == ncols");
        #undef XCOUNT
    }

    /* ncols=4, P=4: each rank gets exactly 1 element */
    {
        int ncols = 4, P = 4;
        int x_blk = (ncols + P - 1) / P;
        CHECK(x_blk == 1, "ncols=4 P=4: blk=1");
        int total = 0;
        for (int r = 0; r < P; r++) {
            int s = r * x_blk;
            int c = s < ncols ? (s + x_blk <= ncols ? x_blk : ncols - s) : 0;
            total += c;
        }
        CHECK(total == ncols, "ncols=4 P=4: sum == 4");
    }

    /* ncols=7, P=1: rank 0 gets all 7 */
    {
        int ncols = 7, P = 1;
        int x_blk = (ncols + P - 1) / P;
        int s = 0;
        int c = s < ncols ? (s + x_blk <= ncols ? x_blk : ncols - s) : 0;
        CHECK(c == 7, "ncols=7 P=1: rank 0 gets all 7");
    }

    /* ncols=5, P=4: blk=2 → counts [2,2,1,0] */
    {
        int ncols = 5, P = 4;
        int x_blk = (ncols + P - 1) / P;
        CHECK(x_blk == 2, "ncols=5 P=4: blk=2");
        int counts[4];
        for (int r = 0; r < P; r++) {
            int s = r * x_blk;
            counts[r] = s < ncols ? (s + x_blk <= ncols ? x_blk : ncols - s) : 0;
        }
        CHECK(counts[0]==2 && counts[1]==2 && counts[2]==1 && counts[3]==0,
              "ncols=5 P=4: counts=[2,2,1,0]");
    }
}

/* ── Test 3: y reconstruction ─────────────────────────────────────────────── */
static void test_y_reconstruction(void)
{
    printf("\n=== y reconstruction  full_y[r + j*P] = gathered_y[displs[r]+j] ===\n");

    /* 6 rows, P=2.
     * rank 0 local: rows 0,1,2 map to global rows 0,2,4
     * rank 1 local: rows 0,1,2 map to global rows 1,3,5
     * gathered_y = [10, 20, 30,   11, 21, 31]
     *               ↑rank0 block   ↑rank1 block         */
    {
        int P = 2;
        int all_lr[2]   = {3, 3};
        int displs_y[2] = {0, 3};
        float gathered_y[6] = {10.f, 20.f, 30.f, 11.f, 21.f, 31.f};
        float full_y[6] = {0};

        for (int r = 0; r < P; r++)
            for (int j = 0; j < all_lr[r]; j++)
                full_y[r + j * P] = gathered_y[displs_y[r] + j];

        CHECK(full_y[0] == 10.f, "P=2 full_y[0]=10 (rank0 local0 → global0)");
        CHECK(full_y[1] == 11.f, "P=2 full_y[1]=11 (rank1 local0 → global1)");
        CHECK(full_y[2] == 20.f, "P=2 full_y[2]=20 (rank0 local1 → global2)");
        CHECK(full_y[3] == 21.f, "P=2 full_y[3]=21 (rank1 local1 → global3)");
        CHECK(full_y[4] == 30.f, "P=2 full_y[4]=30 (rank0 local2 → global4)");
        CHECK(full_y[5] == 31.f, "P=2 full_y[5]=31 (rank1 local2 → global5)");
    }

    /* 4 rows, P=4: each rank contributes exactly 1 element */
    {
        int P = 4;
        int all_lr[4]   = {1, 1, 1, 1};
        int displs_y[4] = {0, 1, 2, 3};
        float gathered_y[4] = {3.f, 3.f, 3.f, 2.f};
        float full_y[4] = {0};

        for (int r = 0; r < P; r++)
            for (int j = 0; j < all_lr[r]; j++)
                full_y[r + j * P] = gathered_y[displs_y[r] + j];

        CHECK(full_y[0]==3.f && full_y[1]==3.f && full_y[2]==3.f && full_y[3]==2.f,
              "P=4 reconstruction matches [3,3,3,2]");
    }

    /* P=1: identity — gathered_y maps directly to full_y */
    {
        int P = 1;
        int all_lr[1]   = {4};
        int displs_y[1] = {0};
        float gathered_y[4] = {1.f, 2.f, 3.f, 4.f};
        float full_y[4] = {0};

        for (int r = 0; r < P; r++)
            for (int j = 0; j < all_lr[r]; j++)
                full_y[r + j * P] = gathered_y[displs_y[r] + j];

        CHECK(full_y[0]==1.f && full_y[1]==2.f && full_y[2]==3.f && full_y[3]==4.f,
              "P=1 reconstruction is identity");
    }
}

/* ── Test 4: global→local row remapping ──────────────────────────────────── */
static void test_row_remapping(void)
{
    printf("\n=== global→local row remap  (global_row / P) ===\n");

    /* P=2: rows 0,2,4 → local 0,1,2 for rank 0; rows 1,3,5 → local 0,1,2 for rank 1 */
    CHECK(0/2 == 0, "global 0 / P=2 → local 0");
    CHECK(2/2 == 1, "global 2 / P=2 → local 1");
    CHECK(4/2 == 2, "global 4 / P=2 → local 2");
    CHECK(1/2 == 0, "global 1 / P=2 → local 0 (rank 1)");
    CHECK(3/2 == 1, "global 3 / P=2 → local 1 (rank 1)");
    CHECK(5/2 == 2, "global 5 / P=2 → local 2 (rank 1)");

    /* P=4: each row maps to local row 0 (only one row per rank for 4-row matrix) */
    CHECK(0/4 == 0, "global 0 / P=4 → local 0");
    CHECK(1/4 == 0, "global 1 / P=4 → local 0");
    CHECK(2/4 == 0, "global 2 / P=4 → local 0");
    CHECK(3/4 == 0, "global 3 / P=4 → local 0");

    /* P=3, 9 rows: ranks 0,1,2 each get 3 rows
     * rank 0: global 0,3,6 → local 0,1,2
     * rank 1: global 1,4,7 → local 0,1,2
     * rank 2: global 2,5,8 → local 0,1,2  */
    CHECK(3/3 == 1, "global 3 / P=3 → local 1");
    CHECK(6/3 == 2, "global 6 / P=3 → local 2");
    CHECK(7/3 == 2, "global 7 / P=3 → local 2");
}

/* ── Driver ───────────────────────────────────────────────────────────────── */
int main(void)
{
    printf("=== D2 Index Math Unit Tests ===\n");
    test_local_row_count();
    test_x_block_distribution();
    test_y_reconstruction();
    test_row_remapping();

    printf("\n%s — %d test(s) failed\n",
           nerr == 0 ? COL_PASS "ALL PASSED" COL_RST
                     : COL_FAIL "FAILURES"   COL_RST,
           nerr);
    return nerr ? 1 : 0;
}
