/* gen_random_csr.c — random sparse matrix generator in Matrix Market format
 *
 * Produces a square N×N matrix with exactly K non-zeros per row,
 * column indices chosen uniformly at random without replacement per row.
 * Output is written to stdout and can be redirected to a .mtx file.
 *
 * Usage:
 *   ./gen_random_csr --rows N --nnz-per-row K [--seed S]
 *
 * Weak-scaling targets (K=10 fixed):
 *   P=1  →  N=500000    NNZ=5M
 *   P=2  →  N=1000000   NNZ=10M
 *   P=4  →  N=2000000   NNZ=20M
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s --rows N --nnz-per-row K [--seed S]\n", prog);
    exit(1);
}

int main(int argc, char *argv[])
{
    long         N    = 0;
    int          K    = 0;
    unsigned int seed = 42;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--rows")       == 0 && i+1 < argc) N    = atol(argv[++i]);
        else if (strcmp(argv[i], "--nnz-per-row") == 0 && i+1 < argc) K    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")        == 0 && i+1 < argc) seed = (unsigned int)atoi(argv[++i]);
        else usage(argv[0]);
    }

    if (N <= 0 || K <= 0) usage(argv[0]);
    if (K > N) { fprintf(stderr, "Error: K (%d) must be <= N (%ld)\n", K, N); exit(1); }

    long nnz = N * (long)K;

    /* Matrix Market header */
    fprintf(stdout, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(stdout, "%% gen_random_csr: rows=%ld nnz-per-row=%d seed=%u\n", N, K, seed);
    fprintf(stdout, "%ld %ld %ld\n", N, N, nnz);

    srand(seed);

    /*
     * For each row pick K distinct column indices using rejection sampling.
     * K is small (10) and N is large (>=500K), so collision probability
     * per pick is at most (K-1)/N < 0.002% — rejections are negligible.
     */
    int *sel = (int *)malloc((size_t)K * sizeof(int));
    if (!sel) { fprintf(stderr, "malloc failed\n"); exit(1); }

    for (long row = 0; row < N; row++) {

        /* Pick K distinct columns */
        int picked = 0;
        while (picked < K) {
            int col = (int)((long)rand() % N);
            int dup = 0;
            for (int j = 0; j < picked; j++)
                if (sel[j] == col) { dup = 1; break; }
            if (!dup) sel[picked++] = col;
        }

        /* Sort within the row for clean output (not required by MTX spec
         * but makes the file easier to inspect and avoids any reader quirks) */
        for (int a = 0; a < K - 1; a++)
            for (int b = a + 1; b < K; b++)
                if (sel[a] > sel[b]) { int tmp = sel[a]; sel[a] = sel[b]; sel[b] = tmp; }

        /* Emit entries — values are uniform random in (0, 1] */
        for (int j = 0; j < K; j++) {
            float val = (float)(rand() + 1) / ((float)RAND_MAX + 1.0f);
            fprintf(stdout, "%ld %d %.6f\n", row + 1, sel[j] + 1, val);
        }
    }

    free(sel);
    return 0;
}
