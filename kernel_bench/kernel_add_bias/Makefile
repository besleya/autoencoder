NVCC := /usr/local/cuda/bin/nvcc
CXXFLAGS := -O3 -std=c++17 --gpu-architecture=sm_90
LDFLAGS := -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusparse -lnvToolsExt -lpthread

TARGETS := test_accuracy bench

.PHONY: all clean

all: $(TARGETS)

test_accuracy: test_accuracy.cu original.cu replacement.cu
	$(NVCC) $(CXXFLAGS) -I/usr/local/cuda/include test_accuracy.cu original.cu replacement.cu -o test_accuracy $(LDFLAGS)

bench: bench.cu original.cu replacement.cu
	$(NVCC) $(CXXFLAGS) -I/usr/local/cuda/include bench.cu original.cu replacement.cu -o bench $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: compile-check
compile-check:
	$(NVCC) -c $(CXXFLAGS) -I/usr/local/cuda/include original.cu -o /tmp/orig_kab.o
	$(NVCC) -c $(CXXFLAGS) -I/usr/local/cuda/include replacement.cu -o /tmp/repl_kab.o
	$(NVCC) $(CXXFLAGS) -I/usr/local/cuda/include test_accuracy.cu original.cu replacement.cu -o /tmp/link_kab_test $(LDFLAGS)
	$(NVCC) $(CXXFLAGS) -I/usr/local/cuda/include bench.cu original.cu replacement.cu -o /tmp/link_kab_bench $(LDFLAGS)
	@echo "Compile-check passed!"
