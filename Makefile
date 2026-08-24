# SPDX-License-Identifier: MIT
CXX      ?= g++
CXX_OPTIONS ?= -O3 -std=c++17
CXXFLAGS ?= $(CXX_OPTIONS) -Wall -Wextra -c
LDFLAGS  ?= -pthread -fopenmp
LDLIBS   ?= -lcublas -lcusparse -ldl -lzstd

# Profiling configuration
ARGS ?=
NSYS_OUT ?= nsys_main_gpu

# CUDA configuration
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_INCLUDE := -I$(CUDA_HOME)/include
CUDA_LIBS := -L$(CUDA_HOME)/lib64 -lcudart

# NVCC flags for device code compilation
NVCCFLAGS ?= $(CXX_OPTIONS) -lineinfo -x cu -dc --gpu-architecture=sm_90 
# -Xcompiler "$(LDFLAGS)" <- had been on the end of that ^ but I don't know what it does

# Singlet configuration
SINGLET_INCLUDE := -I/mnt/home/besleya/singlet/include

# BS::thread_pool
BS_THREAD_POOL_INCLUDE := -I/mnt/home/besleya/include
INCLUDES := $(CUDA_INCLUDE) $(SINGLET_INCLUDE)

.PHONY: all clean profile validate

# Default target: build main only
# TODO: validate target still references old gpu_data_loader and needs updating for new architecture
all: main

# GPU autoencoder driver: multi-species training with shared layers
main: main.o autoencoder.o translator.o slot.o batch.o data_loader.o ring.o layer.o validate_1pz.o main.dlink.o
	$(CXX) $(LDFLAGS) $^ -Wl,--allow-multiple-definition $(CUDA_LIBS) $(LDLIBS) -o $@

main.o: main.cpp translator.h ring.h data_loader.h batch.h autoencoder.h gpu_timer.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(BS_THREAD_POOL_INCLUDE) -c $< -o $@

bench: bench.o slot.o batch.o data_loader.o ring.o validate_1pz.o bench.dlink.o
	$(CXX) $(LDFLAGS) $^ -Wl,--allow-multiple-definition $(CUDA_LIBS) $(LDLIBS) -o $@

bench.o: bench.cpp ring.h data_loader.h batch.h gpu_timer.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(BS_THREAD_POOL_INCLUDE) -c $< -o $@

autoencoder.o: autoencoder.cu autoencoder.h layer.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

translator.o: translator.cpp translator.h autoencoder.h layer.h batch.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $< -o $@

slot.o: slot.cu slot.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

batch.o: batch.cu batch.h slot.h ring.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

data_loader.o: data_loader.cu data_loader.h slot.h batch.h validate_1pz.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(BS_THREAD_POOL_INCLUDE) $< -o $@

ring.o: ring.cu ring.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(BS_THREAD_POOL_INCLUDE) $< -o ring.o

layer.o: layer.cu layer.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

validate_1pz.o: validate_1pz.cpp validate_1pz.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $< -o $@

main.dlink.o: main.o autoencoder.o translator.o slot.o batch.o data_loader.o ring.o layer.o
	$(NVCC) -dlink --gpu-architecture=sm_90 $^ -o $@

bench.dlink.o: bench.o slot.o batch.o data_loader.o ring.o
	$(NVCC) -dlink --gpu-architecture=sm_90 $^ -o $@


# TODO: validate target needs to be updated to use new DataLoader, Ring, and Translator APIs.
# Currently references obsolete gpu_data_loader.cu. Left in place for reference.
# To fix: update tests/validate/validate.cpp and its .o rules below.
#
# validate: tests/validate/validate
#
# tests/validate/validate: tests/validate/validate.o autoencoder.o gpu_data_loader.o ring.o tests/validate/validate.dlink.o layer.o
# 	$(CXX) $(LDFLAGS) tests/validate/validate.o autoencoder.o gpu_data_loader.o ring.o tests/validate/validate.dlink.o layer.o -o $@ -Wl,--allow-multiple-definition $(CUDA_LIBS) $(LDLIBS)
#
# tests/validate/validate.o: tests/validate/validate.cpp autoencoder.h gpu_data_loader.h layer.h
# 	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -dc --gpu-architecture=sm_90 $(INCLUDES) -I. -c tests/validate/validate.cpp -o tests/validate/validate.o
#
# tests/validate/validate.dlink.o: tests/validate/validate.o autoencoder.o gpu_data_loader.o ring.o layer.o
# 	$(NVCC) -dlink -o tests/validate/validate.dlink.o tests/validate/validate.o autoencoder.o gpu_data_loader.o ring.o layer.o --gpu-architecture=sm_90

# profile: main
# 	# Usage: make profile ARGS="--epochs 2 --batch 1024 --lr 0.0001 --hidden 1024,128 --chunk 10 --seed 42 files..."
# 	#        Optional: NSYS_OUT=name (default nsys_main_gpu)
# 	nsys profile --force-overwrite=true --stats=true --trace=cuda,nvtx,osrt -o $(NSYS_OUT) ./main $(ARGS)
profile: main
	nsys profile --force-overwrite=true --stats=true --trace=cuda,nvtx --sample=none --cpuctxsw=none -o $(NSYS_OUT) ./main $(ARGS)

clean:
	$(RM) main.o autoencoder.o translator.o slot.o batch.o data_loader.o ring.o main.dlink.o main layer.o validate_1pz.o
	$(RM) tests/validate/validate.o tests/validate/validate.dlink.o tests/validate/validate
	$(RM) bench.o bench.dlink.o bench

