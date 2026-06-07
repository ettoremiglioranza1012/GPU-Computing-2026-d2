/* spmv_mpi_baseline.cu — Distributed SpMV: 1D cyclic partition, TPV kernel
 *
 * Foster's 4-stage design:
 *   Partition  : 1D cyclic — row i → rank (i mod P)
 *   Communicate: MPI_Allgatherv of x vector every SpMV iteration (GPU-aware)
 *   Aggregate  : each rank runs TPV kernel on its local COO sub-matrix
 *   Map        : one MPI rank per GPU (cudaSetDevice(rank % ngpus))
 *
 * Communication pattern:
 *   x is block-distributed across ranks (rank r owns x[r*blk .. (r+1)*blk-1]).
 *   Before each SpMV, MPI_Allgatherv gathers the full x to every GPU directly
 *   (CUDA-aware MPI — device pointers passed to MPI).
 *   This mirrors the pattern required in an iterative solver where x changes
 *   every iteration.  Communication and compute are timed independently.
 *
 * Compile (see makefile):
 *   nvcc --gpu-architecture=sm_80 -I include/ MPI/spmv_mpi_baseline.cu \
 *        -o bin/MPI/spmv_mpi_baseline $(mpicc --showme:compile) $(mpicc --showme:link)
 *
 * Run:
 *   mpirun -np 4 ./bin/MPI/spmv_mpi_baseline Data/rajat31/rajat31.mtx
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include "mtx_io.h"

#define WARMUP            2
#define NITER             50
#define THREADS_PER_BLOCK 256

/* ── TPV kernel ──────────────────────────────────────────────────────────── */

__global__ void spmv_tpv(const int   * __restrict__ rows,
                          const int   * __restrict__ cols,
                          const float * __restrict__ vals,
                          const float * __restrict__ x,
                          float *y, int local_nnz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < local_nnz)
        atomicAdd(&y[rows[tid]], vals[tid] * __ldg(&x[cols[tid]]));
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static int local_row_count(int nrows, int rank, int P)
{
    if (rank >= nrows) return 0;
    return (nrows - rank - 1) / P + 1;
}

static void extract_matrix_name(const char *path, char *out, int len)
{
    const char *slash = strrchr(path, '/');
    strncpy(out, slash ? slash + 1 : path, len - 1);
    out[len - 1] = '\0';
    char *dot = strrchr(out, '.');
    if (dot) *dot = '\0';
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    /* Bind each rank to a GPU */
    int ngpus = 0;
    cudaGetDeviceCount(&ngpus);
    if (ngpus == 0) {
        fprintf(stderr, "[rank %d] No CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaSetDevice(rank % ngpus);

    if (argc != 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s path/to/matrix.mtx\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char mname[256];
    extract_matrix_name(argv[1], mname, sizeof(mname));

    /* ── 1. Read + scatter matrix ──────────────────────────────────────── */

    int nrows = 0, ncols = 0, nnz_global = 0;
    int    local_nnz  = 0;
    int   *h_lrows    = NULL;
    int   *h_lcols    = NULL;
    float *h_lvals    = NULL;

    int   *h_Arows_ref = NULL;
    int   *h_Acols_ref = NULL;
    float *h_Avals_ref = NULL;

    double t_scatter = 0.0;

    if (rank == 0) {
        mtx_read_coo(argv[1], &nrows, &ncols, &nnz_global,
                     &h_Arows_ref, &h_Acols_ref, &h_Avals_ref);

        t_scatter = MPI_Wtime();

        /* Pass 1: count NNZ per rank */
        int *cnt = (int *)calloc(P, sizeof(int));
        for (int i = 0; i < nnz_global; i++) cnt[h_Arows_ref[i] % P]++;

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
            int r = h_Arows_ref[i] % P;
            rk_r[r][idx[r]] = h_Arows_ref[i];
            rk_c[r][idx[r]] = h_Acols_ref[i];
            rk_v[r][idx[r]] = h_Avals_ref[i];
            idx[r]++;
        }
        free(idx);

        /* Send to ranks 1..P-1, free buffer immediately after each send */
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
        t_scatter = MPI_Wtime() - t_scatter;

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

    /* Remap: global_row / P → local_row (0-based, contiguous) */
    for (int i = 0; i < local_nnz; i++) h_lrows[i] /= P;

    int lr_count = local_row_count(nrows, rank, P);

    /* ── 2. Block-distribute x across ranks ────────────────────────────── */
    /*
     * Rank r owns x[ r*blk .. min((r+1)*blk, ncols) - 1 ].
     * blk = ceil(ncols / P).  MPI_Allgatherv in the benchmark loop
     * gathers the full x to every GPU before each kernel launch —
     * this is the required MPI GPU-aware communication step.
     */

    int x_blk   = (ncols + P - 1) / P;            /* block size (ceil) */
    int x_start = rank * x_blk;                   /* first x index owned by this rank */
    int x_count = x_start < ncols
                  ? (x_start + x_blk <= ncols ? x_blk : ncols - x_start)
                  : 0;                             /* elements owned by this rank */

    /* Generate full x on host with fixed seed (same values on all ranks —
     * in a real solver x would differ; here srand(42) gives reproducibility
     * for the correctness check) */
    float *h_x = (float *)malloc((size_t)ncols * sizeof(float));
    srand(42);
    for (int i = 0; i < ncols; i++) h_x[i] = (float)rand() / (float)RAND_MAX;

    /* MPI_Allgatherv displacement/count arrays */
    int *x_counts = (int *)malloc(P * sizeof(int));
    int *x_displs = (int *)malloc(P * sizeof(int));
    for (int r = 0; r < P; r++) {
        int s = r * x_blk;
        x_displs[r] = s < ncols ? s : ncols;
        int e = s + x_blk < ncols ? s + x_blk : ncols;
        x_counts[r] = e - x_displs[r];
    }

    /* ── 3. CPU reference on rank 0 ────────────────────────────────────── */

    float *h_y_ref = NULL;
    if (rank == 0) {
        h_y_ref = (float *)calloc((size_t)nrows, sizeof(float));
        for (int i = 0; i < nnz_global; i++)
            h_y_ref[h_Arows_ref[i]] += h_Avals_ref[i] * h_x[h_Acols_ref[i]];
    }

    /* ── 4. H2D ──────────────────────────────────────────────────────────── */

    int   *d_rows, *d_cols;
    float *d_vals, *d_x, *d_y;

    cudaMalloc((void **)&d_rows, (size_t)local_nnz * sizeof(int));
    cudaMalloc((void **)&d_cols, (size_t)local_nnz * sizeof(int));
    cudaMalloc((void **)&d_vals, (size_t)local_nnz * sizeof(float));
    cudaMalloc((void **)&d_x,   (size_t)ncols      * sizeof(float));
    cudaMalloc((void **)&d_y,   (size_t)lr_count   * sizeof(float));

    cudaMemcpy(d_rows, h_lrows, (size_t)local_nnz * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, h_lcols, (size_t)local_nnz * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_lvals, (size_t)local_nnz * sizeof(float), cudaMemcpyHostToDevice);

    /* Each rank copies only its block of x to the GPU.
     * The Allgatherv in the benchmark loop fills in the rest. */
    cudaMemset(d_x, 0, (size_t)ncols * sizeof(float));
    if (x_count > 0)
        cudaMemcpy(d_x + x_start, h_x + x_start,
                   (size_t)x_count * sizeof(float), cudaMemcpyHostToDevice);

    int grid = (local_nnz + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    /* ── 5. Benchmark: 2 warmup + 50 timed iterations ───────────────────── */
    /*
     * Each iteration:
     *   a) MPI_Allgatherv — gather full x to all GPUs (GPU-aware MPI)
     *   b) spmv_tpv kernel — local compute on each GPU
     * Both phases are timed independently.
     */

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    double kernel_ms[NITER], comm_ms[NITER];

    for (int iter = -WARMUP; iter < NITER; iter++) {
        cudaMemset(d_y, 0, (size_t)lr_count * sizeof(float));
        MPI_Barrier(MPI_COMM_WORLD);

        /* -- Communication: gather full x to every GPU -- */
        double t_c0 = MPI_Wtime();
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       d_x, x_counts, x_displs, MPI_FLOAT, MPI_COMM_WORLD);
        double t_c1 = MPI_Wtime();

        /* -- Compute: TPV kernel on local COO -- */
        cudaEventRecord(ev_start);
        if (local_nnz > 0)
            spmv_tpv<<<grid, THREADS_PER_BLOCK>>>(d_rows, d_cols, d_vals,
                                                    d_x, d_y, local_nnz);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);

        if (iter >= 0) {
            kernel_ms[iter] = (double)ms;
            comm_ms[iter]   = (t_c1 - t_c0) * 1e3;
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    /* Averages */
    double avg_kernel_ms = 0.0, avg_comm_ms = 0.0;
    for (int i = 0; i < NITER; i++) {
        avg_kernel_ms += kernel_ms[i];
        avg_comm_ms   += comm_ms[i];
    }
    avg_kernel_ms /= NITER;
    avg_comm_ms   /= NITER;

    /* ── 6. D2H ──────────────────────────────────────────────────────────── */

    float *h_y_local = (float *)calloc((size_t)lr_count, sizeof(float));
    cudaMemcpy(h_y_local, d_y, (size_t)lr_count * sizeof(float), cudaMemcpyDeviceToHost);

    /* ── 7. Gather y → rank 0, reconstruct, correctness check ───────────── */

    int   *all_lr     = (rank == 0) ? (int   *)malloc(P * sizeof(int))               : NULL;
    int   *displs_y   = (rank == 0) ? (int   *)malloc(P * sizeof(int))               : NULL;
    float *gathered_y = (rank == 0) ? (float *)malloc((size_t)nrows * sizeof(float)) : NULL;

    MPI_Gather(&lr_count, 1, MPI_INT, all_lr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs_y[0] = 0;
        for (int r = 1; r < P; r++) displs_y[r] = displs_y[r-1] + all_lr[r-1];
    }

    MPI_Gatherv(h_y_local, lr_count, MPI_FLOAT,
                gathered_y, all_lr, displs_y, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int correct = 1;
    if (rank == 0) {
        /* Reconstruct: rank r local y[j] → global y[r + j*P] */
        float *full_y = (float *)calloc((size_t)nrows, sizeof(float));
        for (int r = 0; r < P; r++)
            for (int j = 0; j < all_lr[r]; j++)
                full_y[r + (long)j * P] = gathered_y[displs_y[r] + j];

        float y_max = 0.0f;
        for (int i = 0; i < nrows; i++)
            if (fabsf(h_y_ref[i]) > y_max) y_max = fabsf(h_y_ref[i]);
        float tol = 1e-3f * y_max + 1e-5f;

        for (int i = 0; i < nrows && correct; i++)
            if (fabsf(full_y[i] - h_y_ref[i]) > tol) correct = 0;

        free(full_y);
        free(h_y_ref);
        free(h_Arows_ref); free(h_Acols_ref); free(h_Avals_ref);
    }

    /* ── 8. Gather per-rank stats → rank 0 prints results ───────────────── */

    /* Per-rank compute bandwidth (kernel only) */
    double bytes_compute = (double)local_nnz * (2.0*sizeof(int) + 2.0*sizeof(float))
                         + (double)lr_count  * sizeof(float);
    double bw_compute = bytes_compute / (avg_kernel_ms * 1e-3) / 1e9;
    double gflops     = 2.0 * local_nnz / (avg_kernel_ms * 1e-3) / 1e9;

    /* Per-rank communication volume: each rank receives (P-1)/P of the full x
     * (its own block is already local), but MPI_Allgatherv transfers the full
     * x buffer so we count ncols * sizeof(float) as the communication volume. */
    double comm_bytes_MB = (double)ncols * sizeof(float) / 1e6;

    /* Memory footprint per rank (device allocations) */
    double mem_MB = ((double)local_nnz * (2*sizeof(int) + sizeof(float))
                  + (double)ncols      * sizeof(float)
                  + (double)lr_count   * sizeof(float)) / 1e6;

    /* Gather all per-rank metrics to rank 0 */
    double *all_k_ms   = (rank == 0) ? (double *)malloc(P*sizeof(double)) : NULL;
    double *all_c_ms   = (rank == 0) ? (double *)malloc(P*sizeof(double)) : NULL;
    double *all_bwc    = (rank == 0) ? (double *)malloc(P*sizeof(double)) : NULL;
    double *all_gfl    = (rank == 0) ? (double *)malloc(P*sizeof(double)) : NULL;
    double *all_cvol   = (rank == 0) ? (double *)malloc(P*sizeof(double)) : NULL;
    double *all_mem    = (rank == 0) ? (double *)malloc(P*sizeof(double)) : NULL;
    int    *all_nnz    = (rank == 0) ? (int    *)malloc(P*sizeof(int))    : NULL;
    int    *all_lrc    = (rank == 0) ? (int    *)malloc(P*sizeof(int))    : NULL;

    MPI_Gather(&avg_kernel_ms, 1, MPI_DOUBLE, all_k_ms,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&avg_comm_ms,   1, MPI_DOUBLE, all_c_ms,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&bw_compute,    1, MPI_DOUBLE, all_bwc,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&gflops,        1, MPI_DOUBLE, all_gfl,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&comm_bytes_MB, 1, MPI_DOUBLE, all_cvol,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&mem_MB,        1, MPI_DOUBLE, all_mem,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_nnz,     1, MPI_INT,    all_nnz,   1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Gather(&lr_count,      1, MPI_INT,    all_lrc,   1, MPI_INT,    0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Correctness: %s\n", correct ? "PASSED" : "FAILED");
        printf("Matrix: %s  nrows=%d  ncols=%d  nnz=%d  P=%d\n",
               mname, nrows, ncols, nnz_global, P);
        printf("Scatter time (startup, rank 0): %.4f s\n\n", t_scatter);

        /* NNZ load balance (bonus +0.25 pt) */
        int nnz_min = all_nnz[0], nnz_max = all_nnz[0]; long nnz_sum = 0;
        for (int r = 0; r < P; r++) {
            nnz_sum += all_nnz[r];
            if (all_nnz[r] < nnz_min) nnz_min = all_nnz[r];
            if (all_nnz[r] > nnz_max) nnz_max = all_nnz[r];
        }
        printf("NNZ balance — min: %d  avg: %.0f  max: %d  imbalance: %.2f%%\n\n",
               nnz_min, (double)nnz_sum/P, nnz_max,
               100.0*(nnz_max-nnz_min)/((double)nnz_sum/P));

        /* Per-rank CSV — all required + bonus metrics */
        printf("rank,matrix,P,local_nnz,local_rows,"
               "kernel_ms,comm_ms,comm_vol_MB,mem_MB,bandwidth_GBs,gflops\n");
        double max_kernel_ms = 0.0, max_comm_ms = 0.0;
        for (int r = 0; r < P; r++) {
            printf("%d,%s,%d,%d,%d,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f\n",
                   r, mname, P, all_nnz[r], all_lrc[r],
                   all_k_ms[r], all_c_ms[r],
                   all_cvol[r], all_mem[r],
                   all_bwc[r], all_gfl[r]);
            if (all_k_ms[r] > max_kernel_ms) max_kernel_ms = all_k_ms[r];
            if (all_c_ms[r] > max_comm_ms)   max_comm_ms   = all_c_ms[r];
        }

        /* Summary (required metrics: time, GFLOPs; optional: comm breakdown) */
        double total_ms = max_kernel_ms + max_comm_ms;
        printf("\nWall-clock kernel time   (max across ranks): %.4f ms\n", max_kernel_ms);
        printf("Wall-clock Allgather time (max across ranks): %.4f ms\n", max_comm_ms);
        printf("Total SpMV time (comm + compute):             %.4f ms\n", total_ms);
        printf("Comm fraction:                                %.1f%%\n",
               100.0*max_comm_ms/total_ms);
        printf("Aggregate GFLOPs (2*NNZ/max_kernel_time):    %.4f\n",
               2.0*nnz_global/(max_kernel_ms*1e-3)/1e9);
        printf("\n[speedup and efficiency computed in post-processing from P=1 baseline]\n");

        if (!correct) fprintf(stderr, "CORRECTNESS FAILED — %s P=%d\n", mname, P);

        free(all_k_ms); free(all_c_ms); free(all_bwc); free(all_gfl);
        free(all_cvol); free(all_mem);
        free(all_nnz);  free(all_lrc);
        free(gathered_y); free(all_lr); free(displs_y);
    }

    /* ── Cleanup ─────────────────────────────────────────────────────────── */

    cudaFree(d_rows); cudaFree(d_cols); cudaFree(d_vals);
    cudaFree(d_x);    cudaFree(d_y);
    free(h_lrows); free(h_lcols); free(h_lvals);
    free(h_x);     free(h_y_local);
    free(x_counts); free(x_displs);

    MPI_Finalize();
    return correct ? 0 : 1;
}
