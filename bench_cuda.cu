#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include "cuda_engine.cuh"

using namespace dirichlet;

int main(int argc, char* argv[]) {
    int64 X = 1000000000000LL; // default 10^12
    if (argc > 1) {
        X = std::stoll(argv[1]);
    }
    int block_size = 256;
    if (argc > 2) {
        block_size = std::stoi(argv[2]);
    }

    std::cout << "================================================================================\n";
    std::cout << " NVIDIA CUDA DIRICHLET ENGINE BENCHMARK (NVIDIA T4 16GB / CUDA Architecture)\n";
    std::cout << "================================================================================\n";
    
    CUDADirichletEngine::print_device_info();

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << " Target X: " << X << " | CUDA Threads / Block: " << block_size << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Warmup
    std::cout << "Running Warmup on GPU...\n";
    CUDADirichletEngine::compute_mertens(100000000LL, block_size);

    std::cout << "Executing GPU Mertens M(" << X << ")...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    int64 res = CUDADirichletEngine::compute_mertens(X, block_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "================================================================================\n";
    std::cout << " GPU Output:  M(" << X << ") = " << res << "\n";
    std::cout << " GPU Runtime: " << std::fixed << std::setprecision(4) << elapsed << " seconds\n";
    std::cout << "================================================================================\n";

    return 0;
}
