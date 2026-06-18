# Makefile for kernel_relu_forward benchmark

NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -O3 -std=c++17 --gpu-architecture=sm_90 --extended-lambda

# CUDA libraries
CUDA_LDFLAGS := -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread
CUDA_INCLUDE := -I/usr/local/cuda/include

# Target binaries
TARGETS := test_accuracy bench

# Default rule
all: $(TARGETS)

# Test accuracy
test_accuracy: test_accuracy.cu original.cu replacement.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) $^ -o $@ $(CUDA_LDFLAGS)

# Benchmark
bench: bench.cu original.cu replacement.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) $^ -o $@ $(CUDA_LDFLAGS)

# Clean
clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
