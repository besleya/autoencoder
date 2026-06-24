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

.PHONY: all clean profile validate

# Default target: build main_gpu and validate
all: main_gpu validate

# GPU autoencoder driver: mirrors main.cpp but uses GPU modules
main_gpu: main_gpu.o gpu_autoencoder.o gpu_data_loader.o ring.o main_gpu.dlink.o layer.o
	$(CXX) -pthread -fopenmp $(LDFLAGS) main_gpu.o gpu_autoencoder.o gpu_data_loader.o ring.o main_gpu.dlink.o layer.o -o $@ -Wl,--allow-multiple-definition $(CUDA_LIBS) -lcublas -lcusparse -lnvToolsExt -lzstd

main_gpu.o: main_gpu.cpp gpu_autoencoder.h gpu_data_loader.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -dc --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c main_gpu.cpp -o $@

gpu_autoencoder.o: gpu_autoencoder.cu gpu_autoencoder.h layer.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c gpu_autoencoder.cu -o $@

gpu_data_loader.o: gpu_data_loader.cu gpu_data_loader.h ring.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c gpu_data_loader.cu -o $@
ring.o: ring.cu ring.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c ring.cu -o ring.o
layer.o: layer.cu layer.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -rdc=true --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -c layer.cu -o $@

main_gpu.dlink.o: main_gpu.o gpu_autoencoder.o gpu_data_loader.o ring.o layer.o
	$(NVCC) -dlink -o main_gpu.dlink.o main_gpu.o gpu_autoencoder.o gpu_data_loader.o ring.o layer.o --gpu-architecture=sm_90


# Validation binary: deterministic training check (writes CSVs; run on GPU node)
validate: tests/validate/validate

tests/validate/validate: tests/validate/validate.o gpu_autoencoder.o gpu_data_loader.o ring.o tests/validate/validate.dlink.o layer.o
	$(CXX) -pthread -fopenmp $(LDFLAGS) tests/validate/validate.o gpu_autoencoder.o gpu_data_loader.o ring.o tests/validate/validate.dlink.o layer.o -o $@ -Wl,--allow-multiple-definition $(CUDA_LIBS) -lcublas -lcusparse -lnvToolsExt -lzstd

tests/validate/validate.o: tests/validate/validate.cpp gpu_autoencoder.h gpu_data_loader.h layer.h
	$(NVCC) -O3 -lineinfo -std=c++17 -x cu -dc --gpu-architecture=sm_90 $(CUDA_INCLUDE) $(SINGLET_INCLUDE) -I. -c tests/validate/validate.cpp -o tests/validate/validate.o

tests/validate/validate.dlink.o: tests/validate/validate.o gpu_autoencoder.o gpu_data_loader.o ring.o layer.o
	$(NVCC) -dlink -o tests/validate/validate.dlink.o tests/validate/validate.o gpu_autoencoder.o gpu_data_loader.o ring.o layer.o --gpu-architecture=sm_90

# profile: main_gpu
# 	# Usage: make profile ARGS="--epochs 2 --batch 1024 --lr 0.0001 --hidden 1024,128 --chunk 10 --seed 42 files..."
# 	#        Optional: NSYS_OUT=name (default nsys_main_gpu)
# 	nsys profile --force-overwrite=true --stats=true --trace=cuda,nvtx,osrt -o $(NSYS_OUT) ./main_gpu $(ARGS)
profile: main_gpu
	nsys profile --force-overwrite=true --stats=true --trace=cuda,nvtx --sample=none --cpuctxsw=none -o $(NSYS_OUT) ./main_gpu $(ARGS)

clean:
	rm -f main_gpu.o gpu_autoencoder.o gpu_data_loader.o ring.o main_gpu.dlink.o main_gpu layer.o
	rm -f tests/validate/validate.o tests/validate/validate.dlink.o tests/validate/validate

