# Detect Operating System
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    CXX := clang++
    CXXFLAGS := -O3 -std=c++20 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp
else
    CXX := g++
    CXXFLAGS := -O3 -std=c++20 -fopenmp
    LDFLAGS := -fopenmp
endif

# NVIDIA CUDA Compiler (For NVIDIA T4 / V100 / A100 / Linux / Colab)
NVCC := nvcc
NVCCFLAGS := -O3 -std=c++20 -arch=sm_75 --use_fast_math -Xcompiler "-O3 -fopenmp"

# Metal Framework Flags (macOS)
METAL_FLAGS := -framework Metal -framework Foundation

.PHONY: all cpu metal cuda clean

all: cpu

cpu: test_engine bench reproduce_paper_bench

test_engine: test_engine.cpp dirichlet_engine.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) test_engine.cpp -o test_engine

bench: bench.cpp dirichlet_engine.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) bench.cpp -o bench

reproduce_paper_bench: reproduce_paper_bench.cpp dirichlet_engine.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) reproduce_paper_bench.cpp -o reproduce_paper_bench

# Apple Metal GPU Targets (macOS)
metal: bench_gpu reproduce_gpu_bench

bench_gpu: bench_gpu.mm metal_engine.mm metal_engine.hpp dirichlet_engine.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(METAL_FLAGS) metal_engine.mm bench_gpu.mm -o bench_gpu

reproduce_gpu_bench: reproduce_gpu_bench.mm metal_engine.mm metal_engine.hpp dirichlet_engine.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(METAL_FLAGS) metal_engine.mm reproduce_gpu_bench.mm -o reproduce_gpu_bench

# NVIDIA CUDA Targets (NVIDIA T4 16GB / Tesla / Linux / Colab)
cuda: bench_cuda

bench_cuda: bench_cuda.cu cuda_engine.cu cuda_engine.cuh
	$(NVCC) $(NVCCFLAGS) cuda_engine.cu bench_cuda.cu -o bench_cuda

clean:
	rm -f test_engine bench reproduce_paper_bench bench_gpu reproduce_gpu_bench bench_cuda *.o
