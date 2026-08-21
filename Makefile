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

# BS::thread_pool
BS_THREAD_POOL_INCLUDE := -I/mnt/home/besleya/include

# NVCC flags for device code compilation
NVCCFLAGS ?= -O3 -std=c++17 -x cu -dc --gpu-architecture=sm_90 -Xcompiler "-pthread -fopenmp"

.PHONY: all clean profile validate

# Default target: build main_gpu only
# TODO: validate target still references old gpu_data_loader and needs updating for new architecture
all: main_gpu

# GPU autoencoder driver: multi-species training with shared layers
main_gpu: main_gpu.o autoencoder.o translator.o slot.o batch.o data_loader.o ring.o layer.o validate_1pz.o main_gpu.dlink.o
	$(CXX) -pthread -fopenmp $(LDFLAGS) main_gpu.o autoencoder.o translator.o slot.o batch.o data_loader.o ring.o layer.o validate_1pz.o main_gpu.dlink.o -o $@ -Wl,--allow-multiple-definition $(CUDA_LIBS) -lcublas -lcusparse -ldl -lzstd

main_gpu.o: main_gpu.cpp translator.h ring.h data_loader.h batch.h autoencoder.h gpu_timer.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -dc --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) $(BS_THREAD_POOL_INCLUDE) -c main_gpu.cpp -o $@

autoencoder.o: autoencoder.cu autoencoder.h layer.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c autoencoder.cu -o $@

translator.o: translator.cpp translator.h autoencoder.h layer.h batch.h
	$(CXX) -O3 -std=c++17 -pthread -fopenmp $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c translator.cpp -o $@

slot.o: slot.cu slot.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c slot.cu -o $@

batch.o: batch.cu batch.h slot.h ring.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c batch.cu -o $@

data_loader.o: data_loader.cu data_loader.h slot.h batch.h validate_1pz.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) $(BS_THREAD_POOL_INCLUDE) -c data_loader.cu -o $@

ring.o: ring.cu ring.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) $(BS_THREAD_POOL_INCLUDE) -c ring.cu -o ring.o

layer.o: layer.cu layer.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c layer.cu -o $@

validate_1pz.o: validate_1pz.cpp validate_1pz.h
	$(CXX) -O3 -std=c++17 -pthread -fopenmp $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c validate_1pz.cpp -o $@

main_gpu.dlink.o: main_gpu.o autoencoder.o translator.o slot.o batch.o data_loader.o ring.o layer.o
	$(NVCC) -dlink -o main_gpu.dlink.o main_gpu.o autoencoder.o translator.o slot.o batch.o data_loader.o ring.o layer.o --gpu-architecture=sm_90


# TODO: validate target needs to be updated to use new DataLoader, Ring, and Translator APIs.
# Currently references obsolete gpu_data_loader.cu. Left in place for reference.
# To fix: update tests/validate/validate.cpp and its .o rules below.
#
# validate: tests/validate/validate
#
# tests/validate/validate: tests/validate/validate.o autoencoder.o gpu_data_loader.o ring.o tests/validate/validate.dlink.o layer.o
# 	$(CXX) -pthread -fopenmp $(LDFLAGS) tests/validate/validate.o autoencoder.o gpu_data_loader.o ring.o tests/validate/validate.dlink.o layer.o -o $@ -Wl,--allow-multiple-definition $(CUDA_LIBS) -lcublas -lcusparse -lnvToolsExt -lzstd
#
# tests/validate/validate.o: tests/validate/validate.cpp autoencoder.h gpu_data_loader.h layer.h
# 	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -dc --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -I. -c tests/validate/validate.cpp -o tests/validate/validate.o
#
# tests/validate/validate.dlink.o: tests/validate/validate.o autoencoder.o gpu_data_loader.o ring.o layer.o
# 	$(NVCC) -dlink -o tests/validate/validate.dlink.o tests/validate/validate.o autoencoder.o gpu_data_loader.o ring.o layer.o --gpu-architecture=sm_90

# profile: main_gpu
# 	# Usage: make profile ARGS="--epochs 2 --batch 1024 --lr 0.0001 --hidden 1024,128 --chunk 10 --seed 42 files..."
# 	#        Optional: NSYS_OUT=name (default nsys_main_gpu)
# 	nsys profile --force-overwrite=true --stats=true --trace=cuda,nvtx,osrt -o $(NSYS_OUT) ./main_gpu $(ARGS)
profile: main_gpu
	nsys profile --force-overwrite=true --stats=true --trace=cuda,nvtx --sample=none --cpuctxsw=none -o $(NSYS_OUT) ./main_gpu $(ARGS)

clean:
	rm -f main_gpu.o autoencoder.o translator.o slot.o batch.o data_loader.o ring.o main_gpu.dlink.o main_gpu layer.o validate_1pz.o
	rm -f tests/validate/validate.o tests/validate/validate.dlink.o tests/validate/validate

