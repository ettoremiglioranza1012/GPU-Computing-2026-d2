/* test_mpi_spmv_cpu.c — MPI integration test for the distributed SpMV logic
 *
 * Replicates spmv_mpi_baseline.cu faithfully with two substitutions:
 *   - cudaMalloc/cudaMemcpy/cudaMemset → malloc/memcpy/memset  (no GPU needed)
 *   - spmv_tpv CUDA kernel             → plain host loop
 *
 * All MPI communication calls (MPI_Send/Recv, MPI_Allgatherv, MPI_Gatherv,
 * MPI_Gather, MPI_Bcast) are used without modification, so every distributed
 * logic path is exercised.
 *
 * Test matrix (4×4 upper-bidiagonal, COO):
 *   A = [[2,1,0,0],
 *        [0,2,1,0],
 *        [0,0,2,1],
 *        [0,0,0,2]]
 *   x = [1, 1, 1, 1]
 *   y_ref = [3, 3, 3, 2]
 *
 * Works correctly for P = 1, 2, or 4 (any divisor or factor of 4 up to 4).
 *
 * Compile: mpicc -O0 -g -o test_mpi_spmv_cpu test_mpi_spmv_cpu.c -lm
 * Run:
 *   mpirun -np 1 ./test_mpi_spmv_cpu
 *   mpirun -np 2 ./test_mpi_spmv_cpu
 *   mpirun -np 4 ./test_mpi_spmv_cpu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* ── Known test matrix ───────────────────────────────────────────────────── */
/*
 * 4×4 upper-bidiagonal matrix:
 *   NNZ = 7, y_ref = [3, 3, 3, 2] for x = [1,1,1,1]
 */
static const int   NROWS   = 4;
static const int   NCOLS   = 4;
static const int   NNZ_ALL = 7;

static const int   AROWS_GLOBAL[] = {0, 0, 1, 1, 2, 2, 3};
static const int   ACOLS_GLOBAL[] = {0, 1, 1, 2, 2, 3, 3};
static const float AVALS_GLOBAL[] = {2.f, 1.f, 2.f, 1.f, 2.f, 1.f, 2.f};

static const float X_FULL[] = {1.f, 1.f, 1.f, 1.f};
static const float Y_REF[]  = {3.f, 3.f, 3.f, 2.f};

/* ── Helpers (copied verbatim from spmv_mpi_baseline.cu) ────────────────── */

static int local_row_count(int nrows, int rank, int P)
{
    if (rank >= nrows) return 0;
    return (nrows - rank - 1) / P + 1;
}

/* ── Host SpMV (replaces CUDA TPV kernel) ───────────────────────────────── */

static void spmv_host(const int *rows, const int *cols, const float *vals,
                      const float *x, float *y, int local_nnz)
{
    for (int i = 0; i < local_nnz; i++)
        y[rows[i]] += vals[i] * x[cols[i]];
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    /* ── 1. Distribute matrix from rank 0 ─────────────────────────────── */

    int nrows = 0, ncols = 0, nnz_global = 0;
    int    local_nnz = 0;
    int   *h_lrows   = NULL;
    int   *h_lcols   = NULL;
    float *h_lvals   = NULL;

    if (rank == 0) {
        nrows      = NROWS;
        ncols      = NCOLS;
        nnz_global = NNZ_ALL;

        /* Pass 1: count NNZ per rank */
        int *cnt = (int *)calloc(P, sizeof(int));
        for (int i = 0; i < nnz_global; i++) cnt[AROWS_GLOBAL[i] % P]++;

        /* Allocate per-rank send buffers */
        int   **rk_r = (int   **)malloc(P * sizeof(int   *));
        int   **rk_c = (int   **)malloc(P * sizeof(int   *));
        float **rk_v = (float **)malloc(P * sizeof(float *));
        for (int r = 0; r < P; r++) {
            rk_r[r] = (int   *)malloc((size_t)cnt[r] * sizeof(int));
            rk_c[r] = (int   *)malloc((size_t)cnt[r] * sizeof(int));
            rk_v[r] = (float *)malloc((size_t)cnt[r] * sizeof(float));
        }

        /* Pass 2: fill per-rank buffers */
        int *idx = (int *)calloc(P, sizeof(int));
        for (int i = 0; i < nnz_global; i++) {
            int r = AROWS_GLOBAL[i] % P;
            rk_r[r][idx[r]] = AROWS_GLOBAL[i];
            rk_c[r][idx[r]] = ACOLS_GLOBAL[i];
            rk_v[r][idx[r]] = AVALS_GLOBAL[i];
            idx[r]++;
        }
        free(idx);

        /* Send to ranks 1..P-1 */
        for (int r = 1; r < P; r++) {
            MPI_Send(&cnt[r],  1,       MPI_INT,   r, 0, MPI_COMM_WORLD);
            MPI_Send(rk_r[r], cnt[r],   MPI_INT,   r, 1, MPI_COMM_WORLD);
            MPI_Send(rk_c[r], cnt[r],   MPI_INT,   r, 2, MPI_COMM_WORLD);
            MPI_Send(rk_v[r], cnt[r],   MPI_FLOAT, r, 3, MPI_COMM_WORLD);
            free(rk_r[r]); free(rk_c[r]); free(rk_v[r]);
        }

        local_nnz = cnt[0];
        h_lrows   = rk_r[0];
        h_lcols   = rk_c[0];
        h_lvals   = rk_v[0];
        free(cnt); free(rk_r); free(rk_c); free(rk_v);

    } else {
        MPI_Recv(&local_nnz, 1,         MPI_INT,   0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        h_lrows = (int   *)malloc((size_t)local_nnz * sizeof(int));
        h_lcols = (int   *)malloc((size_t)local_nnz * sizeof(int));
        h_lvals = (float *)malloc((size_t)local_nnz * sizeof(float));
        MPI_Recv(h_lrows, local_nnz, MPI_INT,   0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_lcols, local_nnz, MPI_INT,   0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_lvals, local_nnz, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Broadcast matrix dimensions */
    int dims[2] = {nrows, ncols};
    MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    nrows = dims[0]; ncols = dims[1];

    /* Remap: global_row / P → local row index */
    for (int i = 0; i < local_nnz; i++) h_lrows[i] /= P;

    int lr_count = local_row_count(nrows, rank, P);

    /* ── 2. Block-distribute x ────────────────────────────────────────── */

    int x_blk   = (ncols + P - 1) / P;
    int x_start = rank * x_blk;
    int x_count = x_start < ncols
                  ? (x_start + x_blk <= ncols ? x_blk : ncols - x_start)
                  : 0;

    /* Each rank initialises its block; rest is zero (mirrors cudaMemset+cudaMemcpy) */
    float *h_x = (float *)calloc((size_t)ncols, sizeof(float));
    for (int i = 0; i < x_count; i++)
        h_x[x_start + i] = X_FULL[x_start + i];

    /* Allgatherv displacement/count arrays */
    int *x_counts = (int *)malloc(P * sizeof(int));
    int *x_displs = (int *)malloc(P * sizeof(int));
    for (int r = 0; r < P; r++) {
        int s = r * x_blk;
        x_displs[r] = s < ncols ? s : ncols;
        int e = s + x_blk < ncols ? s + x_blk : ncols;
        x_counts[r] = e - x_displs[r];
    }

    /* ── 3. CPU reference on rank 0 ───────────────────────────────────── */

    float *h_y_ref = NULL;
    if (rank == 0) {
        h_y_ref = (float *)calloc((size_t)nrows, sizeof(float));
        for (int i = 0; i < NNZ_ALL; i++)
            h_y_ref[AROWS_GLOBAL[i]] += AVALS_GLOBAL[i] * X_FULL[ACOLS_GLOBAL[i]];
    }

    /* ── 4. Allgatherv: gather full x to all ranks (no GPU here) ─────── */

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   h_x, x_counts, x_displs, MPI_FLOAT, MPI_COMM_WORLD);

    /* Verify x is correct on all ranks after allgatherv */
    int x_ok = 1;
    for (int i = 0; i < ncols; i++)
        if (fabsf(h_x[i] - X_FULL[i]) > 1e-6f) { x_ok = 0; break; }

    /* ── 5. Local SpMV (host loop, no CUDA) ──────────────────────────── */

    float *h_y = (float *)calloc((size_t)lr_count, sizeof(float));
    spmv_host(h_lrows, h_lcols, h_lvals, h_x, h_y, local_nnz);

    /* ── 6. Gather y → rank 0, reconstruct, check ────────────────────── */

    int   *all_lr     = (rank == 0) ? (int   *)malloc(P * sizeof(int))               : NULL;
    int   *displs_y   = (rank == 0) ? (int   *)malloc(P * sizeof(int))               : NULL;
    float *gathered_y = (rank == 0) ? (float *)malloc((size_t)nrows * sizeof(float)) : NULL;

    MPI_Gather(&lr_count, 1, MPI_INT, all_lr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs_y[0] = 0;
        for (int r = 1; r < P; r++) displs_y[r] = displs_y[r-1] + all_lr[r-1];
    }

    MPI_Gatherv(h_y, lr_count, MPI_FLOAT,
                gathered_y, all_lr, displs_y, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int correct = 1;
    if (rank == 0) {
        float *full_y = (float *)calloc((size_t)nrows, sizeof(float));
        for (int r = 0; r < P; r++)
            for (int j = 0; j < all_lr[r]; j++)
                full_y[r + (long)j * P] = gathered_y[displs_y[r] + j];

        float tol = 1e-4f;
        for (int i = 0; i < nrows; i++) {
            if (fabsf(full_y[i] - h_y_ref[i]) > tol) {
                fprintf(stderr,
                    "[rank0] MISMATCH at row %d: got %.6f expected %.6f\n",
                    i, full_y[i], h_y_ref[i]);
                correct = 0;
            }
        }

        /* Also verify against the hardcoded reference */
        for (int i = 0; i < nrows; i++) {
            if (fabsf(full_y[i] - Y_REF[i]) > tol) {
                fprintf(stderr,
                    "[rank0] full_y[%d]=%.6f != Y_REF[%d]=%.6f\n",
                    i, full_y[i], i, Y_REF[i]);
                correct = 0;
            }
        }

        if (correct && x_ok)
            printf("[rank0] P=%d  x-allgather: OK  SpMV: PASSED  y=[%.0f,%.0f,%.0f,%.0f]\n",
                   P, full_y[0], full_y[1], full_y[2], full_y[3]);
        else
            printf("[rank0] P=%d  FAILED (x_ok=%d correct=%d)\n", P, x_ok, correct);

        free(full_y);
        free(h_y_ref);
        free(all_lr); free(displs_y); free(gathered_y);
    }

    /* Broadcast result to all ranks so non-zero exit propagates */
    MPI_Bcast(&correct, 1, MPI_INT, 0, MPI_COMM_WORLD);

    free(h_lrows); free(h_lcols); free(h_lvals);
    free(h_x); free(h_y);
    free(x_counts); free(x_displs);

    MPI_Finalize();
    return (correct && x_ok) ? 0 : 1;
}
