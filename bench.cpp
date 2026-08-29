#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include "dirichlet_engine.hpp"

void print_separator() {
    std::cout << std::string(80, '=') << "\n";
}

int main(int argc, char* argv[]) {
    using namespace dirichlet;

    std::string mode = "all";
    int64 X = 1000000000000LL; // 10^12
    int threads = 0; // default (all available)

    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "mertens" || arg1 == "totient" || arg1 == "liouville" || arg1 == "prime_pi" || arg1 == "divisor" || arg1 == "all") {
            mode = arg1;
            if (argc > 2) X = std::stoll(argv[2]);
            if (argc > 3) threads = std::stoi(argv[3]);
        } else {
            X = std::stoll(arg1);
            if (argc > 2) threads = std::stoi(argv[2]);
        }
    }

    print_separator();
    std::cout << " PARALLEL DIRICHLET ENGINE BENCHMARK\n";
    std::cout << " Mode: " << mode << " | Target X: " << X << " | Threads: " 
              << (threads > 0 ? std::to_string(threads) : "Max Available") << "\n";
    print_separator();

    auto run_benchmark = [&](const std::string& name, auto func) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = func();
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        
        std::string res_str;
        if constexpr (std::is_same_v<decltype(result), int128>) {
            res_str = to_string_128(result);
        } else {
            res_str = std::to_string(result);
        }

        std::cout << std::left << std::setw(28) << (name + "(" + std::to_string(X) + ")") 
                  << " = " << std::right << std::setw(26) << res_str 
                  << "  [" << std::fixed << std::setprecision(4) << elapsed << " s]\n";
    };

    if (mode == "all" || mode == "mertens") {
        run_benchmark("Mertens M", [&]() { return DirichletEngine::compute_mertens(X, threads); });
    }
    if (mode == "all" || mode == "totient") {
        run_benchmark("Totient Sum Phi", [&]() { return DirichletEngine::compute_totient_sum(X, threads); });
    }
    if (mode == "all" || mode == "liouville") {
        run_benchmark("Liouville Sum L", [&]() { return DirichletEngine::compute_liouville_sum(X, threads); });
    }
    if (mode == "all" || mode == "prime_pi") {
        run_benchmark("PrimeCount pi", [&]() { return DirichletEngine::compute_prime_pi(X); });
    }
    if (mode == "all" || mode == "divisor") {
        run_benchmark("Divisor Sum D", [&]() { return DirichletEngine::compute_divisor_sum(X); });
    }

    print_separator();
    std::cout << " Benchmark completed successfully.\n";
    print_separator();

    return 0;
}