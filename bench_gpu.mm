#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "dirichlet_engine.hpp"
#include "metal_engine.hpp"

using namespace dirichlet;

int main(int argc, char* argv[]) {
    int64 X = 1000000000000LL; // default 10^12
    if (argc > 1) {
        X = std::stoll(argv[1]);
    }
    int threads = 8;
    if (argc > 2) {
        threads = std::stoi(argv[2]);
    }

    std::cout << "================================================================\n";
    std::cout << " HETEROGENEOUS DIRICHLET ENGINE BENCHMARK (CPU + Apple Metal)\n";
    std::cout << " Target X: " << X << " | CPU Threads: " << threads << "\n";
    std::cout << " Metal Available: " << (MetalDirichletEngine::is_available() ? "YES" : "NO") << "\n";
    std::cout << "================================================================\n";

    // Warmup
    MetalDirichletEngine::compute_mertens(1000000000LL, threads);

    // 1. CPU OpenMP Run
    auto t0 = std::chrono::high_resolution_clock::now();
    int64 cpu_res = DirichletEngine::compute_mertens(X, threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "CPU OpenMP:      M(" << X << ") = " << std::setw(15) << cpu_res 
              << "  [" << std::fixed << std::setprecision(4) << cpu_time << " s]\n";

    // 2. Heterogeneous CPU + Metal GPU Run
    auto t2 = std::chrono::high_resolution_clock::now();
    int64 metal_res = MetalDirichletEngine::compute_mertens(X, threads);
    auto t3 = std::chrono::high_resolution_clock::now();
    double metal_time = std::chrono::duration<double>(t3 - t2).count();

    std::cout << "Heterogeneous:   M(" << X << ") = " << std::setw(15) << metal_res 
              << "  [" << std::fixed << std::setprecision(4) << metal_time << " s]\n";

    std::cout << "================================================================\n";
    return 0;
}
