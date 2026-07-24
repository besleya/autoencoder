# Makefile for Slot unit tests

NVCC := /usr/local/cuda/bin/nvcc
CXXFLAGS := -O3 -std=c++17 --gpu-architecture=sm_90
CUDA_INCLUDE := -I/usr/local/cuda/include
CUDA_LIB := -L/usr/local/cuda/lib64
LDFLAGS := -lcudart -lpthread

# Target binary
TARGET := test_slot

# Source files (test_slot.cu and slot.cu)
SRCS := test_slot.cu ../../slot.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CXXFLAGS) $(CUDA_INCLUDE) $(SRCS) -o $@ $(CUDA_LIB) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
