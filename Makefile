# SPDX-License-Identifier: MIT
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp
LDFLAGS  ?= -fopenmp
LDLIBS   ?= -lzstd

# CUDA configuration
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_INCLUDE := -I$(CUDA_HOME)/include
CUDA_LIBS := -L$(CUDA_HOME)/lib64 -lcudart

# Singlet configuration
SINGLET_INCLUDE := -I/mnt/home/besleya/singlet/include

# NVCC flags for device code compilation
NVCCFLAGS ?= -O3 -std=c++17 -x cu -dc --gpu-architecture=sm_90

OBJS = main.o autoencoder.o data.o
TARGET = main_ae

.PHONY: all clean test alt_obj

# Default target: build test only (main_ae is broken pending downstream
# Dataset refactor to DeviceCSC)
all: test

# Legacy main_ae target (currently broken; main.cpp incompatible with new
# DeviceCSC-based Dataset)
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LDLIBS) $(CUDA_LIBS)

main.o: main.cpp autoencoder.h data.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o $@

autoencoder.o: autoencoder.cpp autoencoder.h
	$(CXX) $(CXXFLAGS) -c autoencoder.cpp -o $@

data.o: data.cpp data.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c data.cpp -o $@

# Test executable: loads dataset and inspects structure
test: test.o data.o
	$(NVCC) test.o data.o -o $@ -lzstd -L$(CUDA_HOME)/lib64 -lcudart -Xcompiler -fopenmp

test.o: test.cpp data.h
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c test.cpp -o $@

# Alternative implementation (host-side concat): builds only, no link
data-alt.o: data-alt.cpp data.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c data-alt.cpp -o $@

alt_obj: data-alt.o

clean:
	rm -f $(OBJS) $(TARGET) test.o test data-alt.o

