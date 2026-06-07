# tests/Makefile — build local (no-GPU) test binaries
#
# Run from repo root:  make -C tests
# Or from here:        make

CC    = gcc
MPICC = mpicc
CFLAGS = -O0 -g -Wall -Wextra -Wno-unused-parameter
INC    = -I../include

.PHONY: all clean

all: test_index_math test_mpi_spmv_cpu test_mtx_reader

test_index_math: test_index_math.c
	$(CC) $(CFLAGS) -o $@ $<

test_mpi_spmv_cpu: test_mpi_spmv_cpu.c
	$(MPICC) $(CFLAGS) -o $@ $< -lm

test_mtx_reader: test_mtx_reader.c
	$(CC) $(CFLAGS) $(INC) -o $@ $< -lm

clean:
	rm -f test_index_math test_mpi_spmv_cpu test_mtx_reader
