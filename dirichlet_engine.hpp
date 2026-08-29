#ifndef DIRICHLET_ENGINE_HPP
#define DIRICHLET_ENGINE_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <functional>
#include <cstdint>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dirichlet {

using int64 = long long;
using uint64 = unsigned long long;
using int128 = __int128_t;

/**
 * Converts __int128_t to decimal string.
 */
inline std::string to_string_128(int128 n) {
    if (n == 0) return "0";
    std::string s;
    bool neg = (n < 0);
    if (neg) n = -n;
    while (n > 0) {
        s += static_cast<char>('0' + (n % 10));
        n /= 10;
    }
    if (neg) s += '-';
    std::reverse(s.begin(), s.end());
    return s;
}

/**
 * Exact integer square root using floor semantics.
 */
inline int64 integer_sqrt(int64 n) {
    if (n <= 0) return 0;
    int64 s = static_cast<int64>(std::sqrt(static_cast<double>(n)));
    while ((s + 1) * (s + 1) <= n) ++s;
    while (s * s > n) --s;
    return s;
}

/**
 * QuotientSpace: Represents the hyperbola projection space V(X) = { floor(X/i) : 1 <= i <= X }
 * Provides an exact O(1) coordinate bijection tau(q) mapping states to array indices [0, N-1].
 */
class QuotientSpace {
public:
    int64 X;
    int64 S;              // floor(sqrt(X))
    int n;                // Total number of unique quotient states |V(X)|
    std::vector<int64> V; // Distinct quotient values in strictly ascending order

    explicit QuotientSpace(int64 target_X) : X(target_X) {
        if (X < 1) {
            S = 0;
            n = 0;
            return;
        }

        // Generate distinct values floor(X / i) in descending order
        std::vector<int64> v_desc;
        v_desc.reserve(static_cast<size_t>(2 * std::sqrt(static_cast<double>(X)) + 10));
        for (int64 i = 1; i <= X; ) {
            int64 v = X / i;
            v_desc.push_back(v);
            i = X / v + 1;
        }

        // Reverse to strictly ascending order [1, 2, ..., S, ..., X]
        V.assign(v_desc.rbegin(), v_desc.rend());
        n = static_cast<int>(V.size());
        S = integer_sqrt(X);
    }

    /**
     * Coordinate Bijection tau: V(X) -> {0, ..., n-1}
     * Computes the array index in O(1) arithmetic operations with NO hashing or search.
     */
    inline int tau(int64 q) const noexcept {
        return (q <= S) ? static_cast<int>(q - 1) : (n - static_cast<int>(X / q));
    }

    inline size_t size() const noexcept {
        return static_cast<size_t>(n);
    }

    inline int64 operator[](int idx) const noexcept {
        return V[idx];
    }
};

/**
 * Generalized Dirichlet Hyperbola DP Engine
 */
class DirichletEngine {
public:
    /**
     * Solves the Mertens Function: M(X) = sum_{n=1}^X mu(n)
     * Recurrence: M(v) = 1 - sum_{k=2}^v M(floor(v/k))
     * Complexity: Work O(X^(3/4)), Span O(sqrt(X))
     */
    static int64 compute_mertens(int64 X, int num_threads = 0) {
        if (X < 1) return 0;
        QuotientSpace qs(X);
        const int n = qs.n;
        const int64 S = qs.S;
        std::vector<int> M(n, 0);

        // Hurst-style Linear Pre-sieve up to u = min(S, 20,000,000)
        int64 u = std::min(S, 20000000LL);
        if (u >= 2) {
            std::vector<int8_t> mu(u + 1, 0);
            std::vector<int> primes;
            primes.reserve(static_cast<size_t>(u / 10));
            std::vector<bool> is_prime(u + 1, true);
            mu[1] = 1;
            for (int64 i = 2; i <= u; ++i) {
                if (is_prime[i]) {
                    primes.push_back(static_cast<int>(i));
                    mu[i] = -1;
                }
                for (size_t j = 0; j < primes.size() && i * primes[j] <= u; ++j) {
                    int p = primes[j];
                    is_prime[i * p] = false;
                    if (i % p == 0) {
                        mu[i * p] = 0;
                        break;
                    } else {
                        mu[i * p] = -mu[i];
                    }
                }
            }
            int run_sum = 0;
            for (int64 i = 1; i <= u; ++i) {
                run_sum += mu[i];
                M[i - 1] = run_sum;
            }
        } else {
            M[0] = 1;
        }

        auto solve_block = [&](int start_idx, int end_idx) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads()) if(num_threads != 1)
            #endif
            for (int i = start_idx; i < end_idx; ++i) {
                int64 v = qs.V[i];
                int64 K = integer_sqrt(v);
                int64 k_max = v / (K + 1);
                int total = 1;

                if (v <= S) {
                    for (int64 k = 2; k <= k_max; ++k) {
                        total -= M[(v / k) - 1];
                    }
                } else {
                    int64 k_split = v / (S + 1);
                    int64 k_lim = std::min(k_max, k_split);
                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = v / k;
                        total -= M[n - static_cast<int>(X / q)];
                    }
                    for (int64 k = k_lim + 1; k <= k_max; ++k) {
                        total -= M[(v / k) - 1];
                    }
                }

                // Part 2: Dense q (Sequential memory traversal M[0...K-1])
                int64 q = 1;
                int64 v_div_q = v;
                for (; q <= K; ++q) {
                    int64 v_div_q_next = v / (q + 1);
                    int64 count = v_div_q - v_div_q_next;
                    total -= static_cast<int>(count * M[q - 1]);
                    v_div_q = v_div_q_next;
                }

                M[i] = total;
            }
        };

        // Find initial starting index beyond pre-sieved range
        int cur_start = 0;
        while (cur_start < n && qs.V[cur_start] <= u) {
            ++cur_start;
        }

        // Lock-free doubling stages
        while (cur_start < n) {
            int64 max_safe = qs.V[cur_start - 1] * 2;
            int cur_end = cur_start;
            while (cur_end < n && qs.V[cur_end] <= max_safe) {
                ++cur_end;
            }
            if (cur_end == cur_start) cur_end = cur_start + 1;

            solve_block(cur_start, cur_end);
            cur_start = cur_end;
        }

        return M.back();
    }

    /**
     * Solves Totient Summatory Function: Phi(X) = sum_{n=1}^X phi(n)
     * Identity: sum_{d|n} phi(d) = n  =>  sum_{n<=X} n = X(X+1)/2 = sum_{k<=X} Phi(floor(X/k))
     * Recurrence: Phi(v) = v(v+1)/2 - sum_{k=2}^v Phi(floor(v/k))
     */
    static int128 compute_totient_sum(int64 X, int num_threads = 0) {
        if (X < 1) return 0;
        QuotientSpace qs(X);
        const int n = qs.n;
        const int64 S = qs.S;
        std::vector<int128> Phi(n, 0);

        int64 u = std::min(S, 20000000LL);
        if (u >= 2) {
            std::vector<int> phi(u + 1);
            std::vector<int> primes;
            primes.reserve(static_cast<size_t>(u / 10));
            std::vector<bool> is_prime(u + 1, true);
            phi[1] = 1;
            for (int64 i = 2; i <= u; ++i) {
                if (is_prime[i]) {
                    primes.push_back(static_cast<int>(i));
                    phi[i] = static_cast<int>(i - 1);
                }
                for (size_t j = 0; j < primes.size() && i * primes[j] <= u; ++j) {
                    int p = primes[j];
                    is_prime[i * p] = false;
                    if (i % p == 0) {
                        phi[i * p] = phi[i] * p;
                        break;
                    } else {
                        phi[i * p] = phi[i] * (p - 1);
                    }
                }
            }
            int128 run_sum = 0;
            for (int64 i = 1; i <= u; ++i) {
                run_sum += phi[i];
                Phi[i - 1] = run_sum;
            }
        } else {
            Phi[0] = 1;
        }

        auto solve_block = [&](int start_idx, int end_idx) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads()) if(num_threads != 1)
            #endif
            for (int i = start_idx; i < end_idx; ++i) {
                int64 v = qs.V[i];
                int64 K = integer_sqrt(v);
                int64 k_max = v / (K + 1);
                int128 total = static_cast<int128>(v) * (v + 1) / 2;

                if (v <= S) {
                    for (int64 k = 2; k <= k_max; ++k) {
                        total -= Phi[(v / k) - 1];
                    }
                } else {
                    int64 k_split = v / (S + 1);
                    int64 k_lim = std::min(k_max, k_split);
                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = v / k;
                        total -= Phi[n - static_cast<int>(X / q)];
                    }
                    for (int64 k = k_lim + 1; k <= k_max; ++k) {
                        total -= Phi[(v / k) - 1];
                    }
                }

                int64 q = 1;
                int64 v_div_q = v;
                for (; q <= K; ++q) {
                    int64 v_div_q_next = v / (q + 1);
                    int64 count = v_div_q - v_div_q_next;
                    total -= static_cast<int128>(count) * Phi[q - 1];
                    v_div_q = v_div_q_next;
                }

                Phi[i] = total;
            }
        };

        int cur_start = 0;
        while (cur_start < n && qs.V[cur_start] <= u) {
            ++cur_start;
        }

        // Lock-free doubling stages
        while (cur_start < n) {
            int64 max_safe = qs.V[cur_start - 1] * 2;
            int cur_end = cur_start;
            while (cur_end < n && qs.V[cur_end] <= max_safe) {
                ++cur_end;
            }
            if (cur_end == cur_start) cur_end = cur_start + 1;

            solve_block(cur_start, cur_end);
            cur_start = cur_end;
        }

        return Phi.back();
    }

    /**
     * Solves Liouville Summatory Function: L(X) = sum_{n=1}^X lambda(n)
     * Identity: sum_{d|n} lambda(d) = [n is a square]
     * Recurrence: L(v) = floor(sqrt(v)) - sum_{k=2}^v L(floor(v/k))
     */
    static int64 compute_liouville_sum(int64 X, int num_threads = 0) {
        if (X < 1) return 0;
        QuotientSpace qs(X);
        const int n = qs.n;
        const int64 S = qs.S;
        std::vector<int> L(n, 0);

        int64 u = std::min(S, 20000000LL);
        if (u >= 2) {
            std::vector<int8_t> lmb(u + 1);
            std::vector<int> primes;
            primes.reserve(static_cast<size_t>(u / 10));
            std::vector<bool> is_prime(u + 1, true);
            lmb[1] = 1;
            for (int64 i = 2; i <= u; ++i) {
                if (is_prime[i]) {
                    primes.push_back(static_cast<int>(i));
                    lmb[i] = -1;
                }
                for (size_t j = 0; j < primes.size() && i * primes[j] <= u; ++j) {
                    int p = primes[j];
                    is_prime[i * p] = false;
                    lmb[i * p] = -lmb[i];
                    if (i % p == 0) break;
                }
            }
            int run_sum = 0;
            for (int64 i = 1; i <= u; ++i) {
                run_sum += lmb[i];
                L[i - 1] = run_sum;
            }
        } else {
            L[0] = 1;
        }

        auto solve_block = [&](int start_idx, int end_idx) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads()) if(num_threads != 1)
            #endif
            for (int i = start_idx; i < end_idx; ++i) {
                int64 v = qs.V[i];
                int64 K = integer_sqrt(v);
                int64 k_max = v / (K + 1);
                int total = static_cast<int>(integer_sqrt(v));

                if (v <= S) {
                    for (int64 k = 2; k <= k_max; ++k) {
                        total -= L[(v / k) - 1];
                    }
                } else {
                    int64 k_split = v / (S + 1);
                    int64 k_lim = std::min(k_max, k_split);
                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = v / k;
                        total -= L[n - static_cast<int>(X / q)];
                    }
                    for (int64 k = k_lim + 1; k <= k_max; ++k) {
                        total -= L[(v / k) - 1];
                    }
                }

                int64 q = 1;
                int64 v_div_q = v;
                for (; q <= K; ++q) {
                    int64 v_div_q_next = v / (q + 1);
                    int64 count = v_div_q - v_div_q_next;
                    total -= static_cast<int>(count * L[q - 1]);
                    v_div_q = v_div_q_next;
                }

                L[i] = total;
            }
        };

        int cur_start = 0;
        while (cur_start < n && qs.V[cur_start] <= u) {
            ++cur_start;
        }

        // Doubling stages
        while (cur_start < n) {
            int64 max_safe = qs.V[cur_start - 1] * 2;
            int cur_end = cur_start;
            while (cur_end < n && qs.V[cur_end] <= max_safe) {
                ++cur_end;
            }
            if (cur_end == cur_start) cur_end = cur_start + 1;

            solve_block(cur_start, cur_end);
            cur_start = cur_end;
        }

        return L.back();
    }

    /**
     * Prime Counting Function: pi(X) via Sublinear Sieve (Lucy DP)
     * Work: O(X^(3/4) / log X), Space: O(sqrt(X))
     */
    static int64 compute_prime_pi(int64 X) {
        if (X < 2) return 0;
        QuotientSpace qs(X);
        const int n = qs.n;
        const int64 S = qs.S;

        std::vector<int64> S_arr(n);
        for (int i = 0; i < n; ++i) {
            S_arr[i] = qs.V[i] - 1;
        }

        for (int64 p = 2; p <= S; ++p) {
            int idx_p = qs.tau(p);
            int idx_p_prev = qs.tau(p - 1);
            if (S_arr[idx_p] <= S_arr[idx_p_prev]) {
                continue; // composite
            }

            int64 sp = S_arr[idx_p_prev];
            int64 p2 = p * p;

            for (int i = n - 1; i >= 0; --i) {
                int64 v = qs.V[i];
                if (v < p2) break;
                S_arr[i] -= (S_arr[qs.tau(v / p)] - sp);
            }
        }

        return S_arr.back();
    }

    /**
     * Divisor Summatory Function: D(X) = sum_{n=1}^X d(n)
     * Direct Dirichlet hyperbola method in O(sqrt(X)) time and O(1) memory.
     */
    static int64 compute_divisor_sum(int64 X) {
        if (X < 1) return 0;
        int64 S = integer_sqrt(X);
        int64 sum = 0;
        for (int64 i = 1; i <= S; ++i) {
            sum += X / i;
        }
        return 2 * sum - S * S;
    }
};

} // namespace dirichlet

#endif // DIRICHLET_ENGINE_HPP
