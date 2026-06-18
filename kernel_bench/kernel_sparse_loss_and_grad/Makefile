# Makefile for kernel_sparse_loss_and_grad benchmark

NVCC := /usr/local/cuda/bin/nvcc
NVCC_FLAGS := -O3 -std=c++17 --gpu-architecture=sm_90 -I/usr/local/cuda/include
LDFLAGS := -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread

TARGETS := test_accuracy bench

.PHONY: all clean

all: $(TARGETS)

test_accuracy: test_accuracy.cu original.cu replacement.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

bench: bench.cu original.cu replacement.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o
