# Makefile for kernel_accumulate_loss benchmark
# Builds test_accuracy and bench executables

NVCC := /usr/local/cuda/bin/nvcc
CXXFLAGS := -O3 -std=c++17 --gpu-architecture=sm_90
CUDA_INCLUDE := -I/usr/local/cuda/include
CUDA_LIB := -L/usr/local/cuda/lib64
LDFLAGS := -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread

# Targets
TARGETS := test_accuracy bench

all: $(TARGETS)

test_accuracy: test_accuracy.cu original.cu replacement.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_INCLUDE) test_accuracy.cu original.cu replacement.cu -o $@ $(CUDA_LIB) $(LDFLAGS)

bench: bench.cu original.cu replacement.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_INCLUDE) bench.cu original.cu replacement.cu -o $@ $(CUDA_LIB) $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
