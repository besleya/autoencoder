# SPDX-License-Identifier: MIT
# Makefile for log_normalize_columns_kernel benchmark

CUDA_PATH ?= /usr/local/cuda
NVCC       = $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS = -O3 -std=c++17 --gpu-architecture=sm_90 --extended-lambda
NVCC_INC   = -I$(CUDA_PATH)/include
NVCC_LIB   = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread

SOURCES    = original.cu replacement.cu
OBJECTS    = original.o replacement.o

# Default target
all: test_accuracy bench

# Object files
original.o: original.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -c $< -o $@

replacement.o: replacement.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -c $< -o $@

# Executables
test_accuracy: test_accuracy.cu $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $^ -o $@ $(NVCC_LIB)

bench: bench.cu $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $^ -o $@ $(NVCC_LIB)

# Compile-only check (for CI/validation without GPU)
check:
	$(NVCC) -c $(NVCC_FLAGS) $(NVCC_INC) original.cu -o /tmp/orig_klnc.o
	$(NVCC) -c $(NVCC_FLAGS) $(NVCC_INC) replacement.cu -o /tmp/repl_klnc.o
	@echo "Compilation check passed."

clean:
	rm -f *.o test_accuracy bench /tmp/orig_klnc.o /tmp/repl_klnc.o

.PHONY: all check clean
