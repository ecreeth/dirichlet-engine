#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include "dirichlet_engine.hpp"

using namespace dirichlet;

bool test_mertens() {
    std::cout << "--- Testing Mertens Function M(X) ---\n";
    std::vector<std::pair<int64, int64>> test_cases = {
        {1LL, 1LL},
        {10LL, -1LL},
        {100LL, 1LL},
        {1000LL, 2LL},
        {10000LL, -23LL},
        {100000LL, -48LL},
        {1000000LL, 212LL},
        {10000000LL, 1037LL},
        {100000000LL, 1928LL}
    };

    bool all_passed = true;
    for (const auto& [X, expected] : test_cases) {
        int64 actual = DirichletEngine::compute_mertens(X, 1);
        bool ok = (actual == expected);
        if (ok) {
            std::cout << "[PASS] M(" << X << ") = " << actual << "\n";
        } else {
            std::cout << "[FAIL] M(" << X << ") = " << actual << " (expected " << expected << ")\n";
            all_passed = false;
        }
        assert(ok);
    }
    return all_passed;
}

bool test_prime_pi() {
    std::cout << "--- Testing Prime Counting Function pi(X) ---\n";
    std::vector<std::pair<int64, int64>> test_cases = {
        {10LL, 4LL},
        {100LL, 25LL},
        {1000LL, 168LL},
        {10000LL, 1229LL},
        {100000LL, 9592LL},
        {1000000LL, 78498LL},
        {10000000LL, 664579LL},
        {100000000LL, 5761455LL}
    };

    bool all_passed = true;
    for (const auto& [X, expected] : test_cases) {
        int64 actual = DirichletEngine::compute_prime_pi(X);
        bool ok = (actual == expected);
        if (ok) {
            std::cout << "[PASS] pi(" << X << ") = " << actual << "\n";
        } else {
            std::cout << "[FAIL] pi(" << X << ") = " << actual << " (expected " << expected << ")\n";
            all_passed = false;
        }
        assert(ok);
    }
    return all_passed;
}

bool test_totient_sum() {
    std::cout << "--- Testing Totient Summatory Function Phi(X) ---\n";
    std::vector<std::pair<int64, int128>> test_cases = {
        {10LL, static_cast<int128>(32LL)},
        {100LL, static_cast<int128>(3044LL)},
        {1000LL, static_cast<int128>(304192LL)},
        {10000000LL, static_cast<int128>(30396356427242LL)}
    };

    bool all_passed = true;
    for (const auto& [X, expected] : test_cases) {
        int128 actual = DirichletEngine::compute_totient_sum(X, 1);
        bool ok = (actual == expected);
        if (ok) {
            std::cout << "[PASS] Phi(" << X << ") = " << to_string_128(actual) << "\n";
        } else {
            std::cout << "[FAIL] Phi(" << X << ") = " << to_string_128(actual) 
                      << " (expected " << to_string_128(expected) << ")\n";
            all_passed = false;
        }
        assert(ok);
    }
    return all_passed;
}

bool test_liouville_sum() {
    std::cout << "--- Testing Liouville Summatory Function L(X) ---\n";
    std::vector<std::pair<int64, int64>> test_cases = {
        {10LL, 0LL},
        {100LL, -2LL},
        {1000LL, -14LL},
        {10000000LL, -842LL}
    };

    bool all_passed = true;
    for (const auto& [X, expected] : test_cases) {
        int64 actual = DirichletEngine::compute_liouville_sum(X, 1);
        bool ok = (actual == expected);
        if (ok) {
            std::cout << "[PASS] L(" << X << ") = " << actual << "\n";
        } else {
            std::cout << "[FAIL] L(" << X << ") = " << actual << " (expected " << expected << ")\n";
            all_passed = false;
        }
        assert(ok);
    }
    return all_passed;
}

int main() {
    std::cout << "Running Dirichlet Engine Unit Tests...\n\n";

    bool pass = true;
    pass &= test_mertens();
    std::cout << "\n";
    pass &= test_prime_pi();
    std::cout << "\n";
    pass &= test_totient_sum();
    std::cout << "\n";
    pass &= test_liouville_sum();
    std::cout << "\n";

    if (pass) {
        std::cout << "All Dirichlet Engine tests PASSED!\n";
        return 0;
    } else {
        std::cerr << "Some Dirichlet Engine tests FAILED!\n";
        return 1;
    }
}
