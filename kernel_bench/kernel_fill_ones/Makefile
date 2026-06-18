NVCC := /usr/local/cuda/bin/nvcc
CUDA_ARCH := sm_90
CUDA_PATH := /usr/local/cuda
CXXFLAGS := -O3 -std=c++17 --gpu-architecture=$(CUDA_ARCH) --extended-lambda
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread

.PHONY: all clean test_accuracy bench

all: test_accuracy bench

# Compile original.cu to object file
original.o: original.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Compile replacement.cu to object file
replacement.o: replacement.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Test accuracy executable
test_accuracy: test_accuracy.cu original.o replacement.o
	$(NVCC) $(CXXFLAGS) -o $@ $< original.o replacement.o $(LDFLAGS)

# Benchmark executable
bench: bench.cu original.o replacement.o
	$(NVCC) $(CXXFLAGS) -o $@ $< original.o replacement.o $(LDFLAGS)

clean:
	rm -f *.o test_accuracy bench slurm-*.out

distclean: clean
	rm -f expected_v.bin meta.txt
