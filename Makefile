# SPDX-License-Identifier: MIT
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp
LDFLAGS  ?= -fopenmp -pthread
LDLIBS   ?= -lzstd

# Profiling configuration
ARGS ?=
NSYS_OUT ?= nsys_main_gpu

# CUDA configuration
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_INCLUDE := -I$(CUDA_HOME)/include
CUDA_LIBS := -L$(CUDA_HOME)/lib64 -lcudart

# Singlet configuration
SINGLET_INCLUDE := -I/mnt/home/besleya/singlet/include

# NVCC flags for device code compilation
NVCCFLAGS ?= -O3 -std=c++17 -x cu -dc --gpu-architecture=sm_90 -Xcompiler "-pthread -fopenmp"

OBJS = main.o autoencoder.o data.o
TARGET = main_ae

.PHONY: all clean test alt_obj version profile

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
	$(NVCC) -O3 -std=c++17 -x cu --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c data.cpp -o $@

# Test executable: loads dataset and inspects structure
test: test.o data.o
	$(NVCC) test.o data.o -o $@ -lzstd -L$(CUDA_HOME)/lib64 -lcudart -Xcompiler -fopenmp

test.o: test.cpp data.h
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c test.cpp -o $@

# Alternative implementation (host-side concat): builds only, no link
data-alt.o: data-alt.cpp data.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c data-alt.cpp -o $@

alt_obj: data-alt.o

version: version.o
	$(CXX) $(LDFLAGS) version.o -o $@ $(LDLIBS)

version.o: version.cpp
	$(CXX) $(CXXFLAGS) $(SINGLET_INCLUDE) -c version.cpp -o $@

# Test loader: load pileup files to GPU memory
test_loader: load_pz.o test_loader.o
	$(NVCC) -dlink -o test_loader.dlink.o load_pz.o test_loader.o --gpu-architecture=sm_90
	$(CXX) $(LDFLAGS) load_pz.o test_loader.o test_loader.dlink.o -o test_loader -Wl,--allow-multiple-definition $(CUDA_LIBS) -lzstd

load_pz.o: load_pz.cpp load_pz.h
	$(NVCC) -O3 -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c load_pz.cpp -o $@

test_loader.o: test_loader.cpp load_pz.h
	$(NVCC) -O3 -std=c++17 -x cu --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c test_loader.cpp -o $@

# GPU autoencoder driver: mirrors main.cpp but uses GPU modules
main_gpu: main_gpu.o gpu_autoencoder.o gpu_data_loader.o data.o main_gpu.dlink.o load_pz.o layer.o
	$(CXX) -pthread -fopenmp $(LDFLAGS) main_gpu.o gpu_autoencoder.o gpu_data_loader.o data.o main_gpu.dlink.o load_pz.o layer.o -o $@ -Wl,--allow-multiple-definition $(CUDA_LIBS) -lcublas -lcusparse -lnvToolsExt -lzstd

main_gpu.o: main_gpu.cpp gpu_autoencoder.h gpu_data_loader.h data.h load_pz.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -dc --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c main_gpu.cpp -o $@

gpu_autoencoder.o: gpu_autoencoder.cu gpu_autoencoder.h layer.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c gpu_autoencoder.cu -o $@

gpu_data_loader.o: gpu_data_loader.cu gpu_data_loader.h data.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c gpu_data_loader.cu -o $@

layer.o: layer.cu layer.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c layer.cu -o $@

main_gpu.dlink.o: main_gpu.o gpu_autoencoder.o gpu_data_loader.o load_pz.o layer.o
	$(NVCC) -dlink -o main_gpu.dlink.o main_gpu.o gpu_autoencoder.o gpu_data_loader.o load_pz.o layer.o --gpu-architecture=sm_90

# profile: main_gpu
# 	# Usage: make profile ARGS="--epochs 2 --batch 1024 --lr 0.0001 --hidden 1024,128 --chunk 10 --seed 42 files..."
# 	#        Optional: NSYS_OUT=name (default nsys_main_gpu)
# 	nsys profile --force-overwrite=true --stats=true --trace=cuda,nvtx,osrt -o $(NSYS_OUT) ./main_gpu $(ARGS)
profile: main_gpu
	nsys profile --force-overwrite=true --stats=true --trace=cuda,nvtx --sample=none --cpuctxsw=none -o $(NSYS_OUT) ./main_gpu $(ARGS)

clean:
	rm -f $(OBJS) $(TARGET) test.o test data-alt.o load_pz.o test_loader.o test_loader main_gpu.o gpu_autoencoder.o gpu_data_loader.o main_gpu.dlink.o main_gpu layer.o

