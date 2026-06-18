# Makefile for ReLU backward kernel benchmark

NVCC = /usr/local/cuda/bin/nvcc
CUDA_ARCH = sm_90
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

CFLAGS = -O3 -std=c++17 --gpu-architecture=$(CUDA_ARCH) --extended-lambda
LDFLAGS = -L$(CUDA_LIB) -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread

# Targets
all: test_accuracy bench

test_accuracy: test_accuracy.cu original.cu replacement.cu
	$(NVCC) $(CFLAGS) -I$(CUDA_INCLUDE) test_accuracy.cu original.cu replacement.cu -o test_accuracy $(LDFLAGS)

bench: bench.cu original.cu replacement.cu
	$(NVCC) $(CFLAGS) -I$(CUDA_INCLUDE) bench.cu original.cu replacement.cu -o bench $(LDFLAGS)

clean:
	rm -f test_accuracy bench *.o

.PHONY: all clean
