# Makefile for kernel_adam_update benchmarking

NVCC := /usr/local/cuda/bin/nvcc
CUDA_ARCH := sm_90
CUDA_PATH := /usr/local/cuda
CXXFLAGS := -O3 -std=c++17 --gpu-architecture=$(CUDA_ARCH) --extended-lambda
INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread

# Object files
OBJS_COMMON := original.o replacement.o

# Targets
all: test_accuracy bench

# Compile individual source files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link test_accuracy
test_accuracy: test_accuracy.o $(OBJS_COMMON)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $^ $(LIBS) -o $@

# Link bench
bench: bench.o $(OBJS_COMMON)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $^ $(LIBS) -o $@

# Clean
clean:
	rm -f *.o test_accuracy bench

.PHONY: all clean
