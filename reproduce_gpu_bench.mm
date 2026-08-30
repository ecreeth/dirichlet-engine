#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "dirichlet_engine.hpp"
#include "metal_engine.hpp"

using namespace dirichlet;

int main(int argc, char* argv[]) {
    int threads = 8;
    if (argc > 1) threads = std::stoi(argv[1]);
    int max_exp = 16;
    if (argc > 2) max_exp = std::stoi(argv[2]);

    std::vector<std::pair<int64, std::string>> all_scales = {
        {10000000LL, "10^7"},
        {100000000LL, "10^8"},
        {1000000000LL, "10^9"},
        {10000000000LL, "10^{10}"},
        {100000000000LL, "10^{11}"},
        {1000000000000LL, "10^{12}"},
        {10000000000000LL, "10^{13}"},
        {100000000000000LL, "10^{14}"},
        {1000000000000000LL, "10^{15}"},
        {10000000000000000LL, "10^{16}"}
    };

    std::cout << "========================================================================================================\n";
    std::cout << " HETEROGENEOUS GPU ANTICHAIN KERNEL BENCHMARK (Apple M1 8-Core CPU + Metal GPU)\n";
    std::cout << " Max Scale: 10^" << max_exp << " | CPU Threads: " << threads << " | Metal: " 
              << (MetalDirichletEngine::is_available() ? "AVAILABLE" : "UNAVAILABLE") << "\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::setw(12) << "Target X"
              << std::setw(18) << "Exact M(X)"
              << std::setw(18) << "CPU Time (s)"
              << std::setw(22) << "Heterogeneous Time (s)"
              << std::setw(16) << "Match Status"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    // Warmup
    MetalDirichletEngine::compute_mertens(100000000LL, threads);

    for (const auto& [X, label] : all_scales) {
        int exp_val = static_cast<int>(std::round(std::log10(static_cast<double>(X))));
        if (exp_val > max_exp) continue;

        // CPU run
        auto t0 = std::chrono::high_resolution_clock::now();
        int64 cpu_res = DirichletEngine::compute_mertens(X, threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double>(t1 - t0).count();

        // Heterogeneous run
        auto t2 = std::chrono::high_resolution_clock::now();
        int64 gpu_res = MetalDirichletEngine::compute_mertens(X, threads);
        auto t3 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double>(t3 - t2).count();

        bool match = (cpu_res == gpu_res);

        std::cout << std::setw(12) << label
                  << std::setw(18) << gpu_res
                  << std::setw(16) << std::fixed << std::setprecision(4) << cpu_time << " s"
                  << std::setw(20) << std::fixed << std::setprecision(4) << gpu_time << " s"
                  << std::setw(16) << (match ? "[EXACT MATCH]" : "[MISMATCH]")
                  << "\n";
        std::cout.flush();
    }

    std::cout << "========================================================================================================\n";
    return 0;
}
