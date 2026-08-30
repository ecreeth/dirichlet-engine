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

#if defined(__ARM_NEON)
#include <arm_neon.h>
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

        S = integer_sqrt(X);
        V.reserve(static_cast<size_t>(2 * S + 5));

        // Prefix dense states: 1, 2, ..., S
        for (int64 i = 1; i <= S; ++i) {
            V.push_back(i);
        }

        // Suffix sparse states: floor(X / k) for k = S down to 1
        for (int64 k = S; k >= 1; --k) {
            int64 v = X / k;
            if (v > S && v != V.back()) {
                V.push_back(v);
            }
        }

        n = static_cast<int>(V.size());
    }

    /**
     * Coordinate Bijection tau: V(X) -> {0, ..., n-1}
     * Computes the array index in O(1) arithmetic operations with NO hashing or search.
     */
    inline int tau(int64 q) const noexcept {
        if (q <= S) return static_cast<int>(q - 1);
        int idx = n - static_cast<int>(X / q);
        if (__builtin_expect(idx >= n, 0)) idx = n - 1;
        if (__builtin_expect(idx < 0, 0)) idx = 0;
        return idx;
    }

    inline size_t size() const noexcept {
        return static_cast<size_t>(n);
    }

    inline int64 operator[](int idx) const noexcept {
        return V[idx];
    }
};

inline int64 safe_fast_div(int64 v, double dv, int64 k) noexcept {
    if (__builtin_expect(v <= 9007199254740992LL, 1)) {
        return static_cast<int64>(dv / static_cast<double>(k));
    } else {
        return v / k;
    }
}

/**
 * Generalized Dirichlet Hyperbola DP Engine
 * Optimized with extended pre-sieve, fast reciprocal division, and vectorization.
 */
class DirichletEngine {
private:
    static inline int64 choose_pre_sieve_limit(int64 X, int threads) {
        int64 S = integer_sqrt(X);
        double c_ratio = (threads > 1) ? (0.20 / std::cbrt(static_cast<double>(threads))) : 0.80;
        int64 target_u = static_cast<int64>(c_ratio * std::pow(static_cast<double>(X), 2.0 / 3.0));
        if (target_u < S) target_u = S;
        return std::min(target_u, 100000000LL);
    }

public:
    /**
     * Solves the Mertens Function: M(X) = sum_{n=1}^X mu(n)
     * Recurrence: M(v) = 1 - sum_{k=2}^v M(floor(v/k))
     */
    static int64 compute_mertens(int64 X, int num_threads = 0) {
        if (X < 1) return 0;
        int threads = (num_threads > 0 ? num_threads :
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
        );
        QuotientSpace qs(X);
        const int n = qs.n;
        const int64 S = qs.S;

        int64 u = choose_pre_sieve_limit(X, threads);

        // Fast odd-only linear sieve for Mobius function
        std::vector<int> M_sieve(u + 1, 0);
        {
            int half_u = static_cast<int>((u + 1) / 2);
            std::vector<int8_t> mu_odd(half_u + 1, 0);
            std::vector<int> primes;
            primes.reserve(static_cast<size_t>(u / 10));
            std::vector<uint8_t> is_prime(half_u + 1, 1);
            mu_odd[0] = 0;
            mu_odd[1] = 1;
            
            for (int i = 2; 2 * i - 1 <= u; ++i) {
                int num = 2 * i - 1;
                if (is_prime[i]) {
                    primes.push_back(num);
                    mu_odd[i] = -1;
                }
                for (size_t j = 0; j < primes.size(); ++j) {
                    int p = primes[j];
                    int64_t prod = static_cast<int64_t>(num) * p;
                    if (prod > u) break;
                    int idx = static_cast<int>((prod + 1) / 2);
                    is_prime[idx] = 0;
                    if (num % p == 0) {
                        mu_odd[idx] = 0;
                        break;
                    } else {
                        mu_odd[idx] = -mu_odd[i];
                    }
                }
            }
            
            int run_sum = 0;
            for (int64 i = 1; i <= u; ++i) {
                int8_t m;
                if (i % 2 != 0) {
                    m = mu_odd[(i + 1) / 2];
                } else if ((i / 2) % 2 != 0) {
                    m = -mu_odd[((i / 2) + 1) / 2];
                } else {
                    m = 0;
                }
                run_sum += m;
                M_sieve[i] = run_sum;
            }
        }

        if (X <= u) return M_sieve[X];

        std::vector<int> M_dp(n, 0);

        auto solve_block = [&](int start_idx, int end_idx) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 16) num_threads(threads) if(threads != 1)
            #endif
            for (int i = start_idx; i < end_idx; ++i) {
                int64 v = qs.V[i];
                double dv = static_cast<double>(v);
                int64 K = static_cast<int64>(std::sqrt(dv));
                int64 k_max = static_cast<int64>(dv / static_cast<double>(K + 1));
                int total = 1;

                if (v <= 9007199254740992LL) {
                    int64 k_split = static_cast<int64>(dv / static_cast<double>(u + 1));
                    int64 k_lim = std::min(k_max, k_split);

                    // Part 1a: v/k > u (queried from DP table)
                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = static_cast<int64>(dv / static_cast<double>(k));
                        total -= M_dp[n - static_cast<int>(X / q)];
                    }
                    // Part 1b: v/k <= u (queried from linear sieve buffer)
                    int64 k = k_lim + 1;
                    for (; k + 3 <= k_max; k += 4) {
                        total -= M_sieve[static_cast<int64>(dv / static_cast<double>(k))];
                        total -= M_sieve[static_cast<int64>(dv / static_cast<double>(k + 1))];
                        total -= M_sieve[static_cast<int64>(dv / static_cast<double>(k + 2))];
                        total -= M_sieve[static_cast<int64>(dv / static_cast<double>(k + 3))];
                    }
                    for (; k <= k_max; ++k) {
                        total -= M_sieve[static_cast<int64>(dv / static_cast<double>(k))];
                    }

                    // Part 2: Dense q <= K (Sequential memory traversal M_sieve[1...K])
                    int64 q = 1;
                    int64 v_div_q = v;

                    #if defined(__ARM_NEON)
                    float64x2_t v_vec = vdupq_n_f64(dv);
                    for (; q + 3 <= K; q += 4) {
                        float64x2_t d1 = { static_cast<double>(q + 1), static_cast<double>(q + 2) };
                        float64x2_t d2 = { static_cast<double>(q + 3), static_cast<double>(q + 4) };
                        float64x2_t res1 = vdivq_f64(v_vec, d1);
                        float64x2_t res2 = vdivq_f64(v_vec, d2);
                        
                        int64 n1 = static_cast<int64>(vgetq_lane_f64(res1, 0));
                        int64 n2 = static_cast<int64>(vgetq_lane_f64(res1, 1));
                        int64 n3 = static_cast<int64>(vgetq_lane_f64(res2, 0));
                        int64 n4 = static_cast<int64>(vgetq_lane_f64(res2, 1));
                        
                        total -= static_cast<int>((v_div_q - n1) * M_sieve[q]);
                        total -= static_cast<int>((n1 - n2) * M_sieve[q + 1]);
                        total -= static_cast<int>((n2 - n3) * M_sieve[q + 2]);
                        total -= static_cast<int>((n3 - n4) * M_sieve[q + 3]);
                        v_div_q = n4;
                    }
                    #endif

                    for (; q <= K; ++q) {
                        int64 v_div_q_next = static_cast<int64>(dv / static_cast<double>(q + 1));
                        int64 count = v_div_q - v_div_q_next;
                        total -= static_cast<int>(count * M_sieve[q]);
                        v_div_q = v_div_q_next;
                    }
                } else {
                    // Exact 64-bit integer path for v > 2^53
                    int64 k_split = v / (u + 1);
                    int64 k_lim = std::min(k_max, k_split);

                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = v / k;
                        total -= M_dp[n - static_cast<int>(X / q)];
                    }
                    for (int64 k = k_lim + 1; k <= k_max; ++k) {
                        total -= M_sieve[v / k];
                    }

                    int64 q = 1;
                    int64 v_div_q = v;
                    for (; q <= K; ++q) {
                        int64 v_div_q_next = v / (q + 1);
                        int64 count = v_div_q - v_div_q_next;
                        total -= static_cast<int>(count * M_sieve[q]);
                        v_div_q = v_div_q_next;
                    }
                }

                M_dp[i] = total;
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

        return M_dp.back();
    }

    /**
     * Solves Totient Summatory Function: Phi(X) = sum_{n=1}^X phi(n)
     * Identity: sum_{d|n} phi(d) = n  =>  sum_{n<=X} n = X(X+1)/2 = sum_{k<=X} Phi(floor(X/k))
     * Recurrence: Phi(v) = v(v+1)/2 - sum_{k=2}^v Phi(floor(v/k))
     */
    static int128 compute_totient_sum(int64 X, int num_threads = 0) {
        if (X < 1) return 0;
        int threads = (num_threads > 0 ? num_threads :
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
        );
        QuotientSpace qs(X);
        const int n = qs.n;
        const int64 S = qs.S;

        int64 u = choose_pre_sieve_limit(X, threads);

        std::vector<int64> Phi_sieve(u + 1, 0);
        {
            std::vector<int> phi(u + 1);
            std::vector<int> primes;
            primes.reserve(static_cast<size_t>(u / 10));
            std::vector<uint8_t> is_prime(u + 1, 1);
            phi[1] = 1;
            for (int64 i = 2; i <= u; ++i) {
                if (is_prime[i]) {
                    primes.push_back(static_cast<int>(i));
                    phi[i] = static_cast<int>(i - 1);
                }
                for (size_t j = 0; j < primes.size() && i * primes[j] <= u; ++j) {
                    int p = primes[j];
                    is_prime[i * p] = 0;
                    if (i % p == 0) {
                        phi[i * p] = phi[i] * p;
                        break;
                    } else {
                        phi[i * p] = phi[i] * (p - 1);
                    }
                }
            }
            int64 run_sum = 0;
            for (int64 i = 1; i <= u; ++i) {
                run_sum += phi[i];
                Phi_sieve[i] = run_sum;
            }
        }

        if (X <= u) return Phi_sieve[X];

        std::vector<int128> Phi_dp(n, 0);

        auto solve_block = [&](int start_idx, int end_idx) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 16) num_threads(threads) if(threads != 1)
            #endif
            for (int i = start_idx; i < end_idx; ++i) {
                int64 v = qs.V[i];
                double dv = static_cast<double>(v);
                int64 K = static_cast<int64>(std::sqrt(dv));
                int64 k_max = static_cast<int64>(dv / static_cast<double>(K + 1));
                int128 total = static_cast<int128>(v) * (v + 1) / 2;

                if (v <= 9007199254740992LL) {
                    int64 k_split = static_cast<int64>(dv / static_cast<double>(u + 1));
                    int64 k_lim = std::min(k_max, k_split);

                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = static_cast<int64>(dv / static_cast<double>(k));
                        total -= Phi_dp[n - static_cast<int>(X / q)];
                    }
                    int64 k = k_lim + 1;
                    for (; k + 3 <= k_max; k += 4) {
                        total -= Phi_sieve[static_cast<int64>(dv / static_cast<double>(k))];
                        total -= Phi_sieve[static_cast<int64>(dv / static_cast<double>(k + 1))];
                        total -= Phi_sieve[static_cast<int64>(dv / static_cast<double>(k + 2))];
                        total -= Phi_sieve[static_cast<int64>(dv / static_cast<double>(k + 3))];
                    }
                    for (; k <= k_max; ++k) {
                        total -= Phi_sieve[static_cast<int64>(dv / static_cast<double>(k))];
                    }

                    int64 q = 1;
                    int64 v_div_q = v;

                    #if defined(__ARM_NEON)
                    float64x2_t v_vec = vdupq_n_f64(dv);
                    for (; q + 3 <= K; q += 4) {
                        float64x2_t d1 = { static_cast<double>(q + 1), static_cast<double>(q + 2) };
                        float64x2_t d2 = { static_cast<double>(q + 3), static_cast<double>(q + 4) };
                        float64x2_t res1 = vdivq_f64(v_vec, d1);
                        float64x2_t res2 = vdivq_f64(v_vec, d2);
                        
                        int64 n1 = static_cast<int64>(vgetq_lane_f64(res1, 0));
                        int64 n2 = static_cast<int64>(vgetq_lane_f64(res1, 1));
                        int64 n3 = static_cast<int64>(vgetq_lane_f64(res2, 0));
                        int64 n4 = static_cast<int64>(vgetq_lane_f64(res2, 1));
                        
                        total -= static_cast<int128>(v_div_q - n1) * Phi_sieve[q];
                        total -= static_cast<int128>(n1 - n2) * Phi_sieve[q + 1];
                        total -= static_cast<int128>(n2 - n3) * Phi_sieve[q + 2];
                        total -= static_cast<int128>(n3 - n4) * Phi_sieve[q + 3];
                        v_div_q = n4;
                    }
                    #endif

                    for (; q <= K; ++q) {
                        int64 v_div_q_next = static_cast<int64>(dv / static_cast<double>(q + 1));
                        int64 count = v_div_q - v_div_q_next;
                        total -= static_cast<int128>(count) * Phi_sieve[q];
                        v_div_q = v_div_q_next;
                    }
                } else {
                    int64 k_split = v / (u + 1);
                    int64 k_lim = std::min(k_max, k_split);

                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = v / k;
                        total -= Phi_dp[n - static_cast<int>(X / q)];
                    }
                    for (int64 k = k_lim + 1; k <= k_max; ++k) {
                        total -= Phi_sieve[v / k];
                    }

                    int64 q = 1;
                    int64 v_div_q = v;
                    for (; q <= K; ++q) {
                        int64 v_div_q_next = v / (q + 1);
                        int64 count = v_div_q - v_div_q_next;
                        total -= static_cast<int128>(count) * Phi_sieve[q];
                        v_div_q = v_div_q_next;
                    }
                }

                Phi_dp[i] = total;
            }
        };

        int cur_start = 0;
        while (cur_start < n && qs.V[cur_start] <= u) {
            ++cur_start;
        }

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

        return Phi_dp.back();
    }

    /**
     * Solves Liouville Summatory Function: L(X) = sum_{n=1}^X lambda(n)
     * Identity: sum_{d|n} lambda(d) = [n is a square]
     * Recurrence: L(v) = floor(sqrt(v)) - sum_{k=2}^v L(floor(v/k))
     */
    static int64 compute_liouville_sum(int64 X, int num_threads = 0) {
        if (X < 1) return 0;
        int threads = (num_threads > 0 ? num_threads :
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
        );
        QuotientSpace qs(X);
        const int n = qs.n;
        const int64 S = qs.S;

        int64 u = choose_pre_sieve_limit(X, threads);

        std::vector<int> L_sieve(u + 1, 0);
        {
            std::vector<int8_t> lmb(u + 1);
            std::vector<int> primes;
            primes.reserve(static_cast<size_t>(u / 10));
            std::vector<uint8_t> is_prime(u + 1, 1);
            lmb[1] = 1;
            for (int64 i = 2; i <= u; ++i) {
                if (is_prime[i]) {
                    primes.push_back(static_cast<int>(i));
                    lmb[i] = -1;
                }
                for (size_t j = 0; j < primes.size() && i * primes[j] <= u; ++j) {
                    int p = primes[j];
                    is_prime[i * p] = 0;
                    lmb[i * p] = -lmb[i];
                    if (i % p == 0) break;
                }
            }
            int run_sum = 0;
            for (int64 i = 1; i <= u; ++i) {
                run_sum += lmb[i];
                L_sieve[i] = run_sum;
            }
        }

        if (X <= u) return L_sieve[X];

        std::vector<int> L_dp(n, 0);

        auto solve_block = [&](int start_idx, int end_idx) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 16) num_threads(threads) if(threads != 1)
            #endif
            for (int i = start_idx; i < end_idx; ++i) {
                int64 v = qs.V[i];
                double dv = static_cast<double>(v);
                int64 K = static_cast<int64>(std::sqrt(dv));
                int64 k_max = static_cast<int64>(dv / static_cast<double>(K + 1));
                int total = static_cast<int>(K);

                if (v <= 9007199254740992LL) {
                    int64 k_split = static_cast<int64>(dv / static_cast<double>(u + 1));
                    int64 k_lim = std::min(k_max, k_split);

                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = static_cast<int64>(dv / static_cast<double>(k));
                        total -= L_dp[n - static_cast<int>(X / q)];
                    }
                    int64 k = k_lim + 1;
                    for (; k + 3 <= k_max; k += 4) {
                        total -= L_sieve[static_cast<int64>(dv / static_cast<double>(k))];
                        total -= L_sieve[static_cast<int64>(dv / static_cast<double>(k + 1))];
                        total -= L_sieve[static_cast<int64>(dv / static_cast<double>(k + 2))];
                        total -= L_sieve[static_cast<int64>(dv / static_cast<double>(k + 3))];
                    }
                    for (; k <= k_max; ++k) {
                        total -= L_sieve[static_cast<int64>(dv / static_cast<double>(k))];
                    }

                    int64 q = 1;
                    int64 v_div_q = v;

                    #if defined(__ARM_NEON)
                    float64x2_t v_vec = vdupq_n_f64(dv);
                    for (; q + 3 <= K; q += 4) {
                        float64x2_t d1 = { static_cast<double>(q + 1), static_cast<double>(q + 2) };
                        float64x2_t d2 = { static_cast<double>(q + 3), static_cast<double>(q + 4) };
                        float64x2_t res1 = vdivq_f64(v_vec, d1);
                        float64x2_t res2 = vdivq_f64(v_vec, d2);
                        
                        int64 n1 = static_cast<int64>(vgetq_lane_f64(res1, 0));
                        int64 n2 = static_cast<int64>(vgetq_lane_f64(res1, 1));
                        int64 n3 = static_cast<int64>(vgetq_lane_f64(res2, 0));
                        int64 n4 = static_cast<int64>(vgetq_lane_f64(res2, 1));
                        
                        total -= static_cast<int>((v_div_q - n1) * L_sieve[q]);
                        total -= static_cast<int>((n1 - n2) * L_sieve[q + 1]);
                        total -= static_cast<int>((n2 - n3) * L_sieve[q + 2]);
                        total -= static_cast<int>((n3 - n4) * L_sieve[q + 3]);
                        v_div_q = n4;
                    }
                    #endif

                    for (; q <= K; ++q) {
                        int64 v_div_q_next = static_cast<int64>(dv / static_cast<double>(q + 1));
                        int64 count = v_div_q - v_div_q_next;
                        total -= static_cast<int>(count * L_sieve[q]);
                        v_div_q = v_div_q_next;
                    }
                } else {
                    int64 k_split = v / (u + 1);
                    int64 k_lim = std::min(k_max, k_split);

                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = v / k;
                        total -= L_dp[n - static_cast<int>(X / q)];
                    }
                    for (int64 k = k_lim + 1; k <= k_max; ++k) {
                        total -= L_sieve[v / k];
                    }

                    int64 q = 1;
                    int64 v_div_q = v;
                    for (; q <= K; ++q) {
                        int64 v_div_q_next = v / (q + 1);
                        int64 count = v_div_q - v_div_q_next;
                        total -= static_cast<int>(count * L_sieve[q]);
                        v_div_q = v_div_q_next;
                    }
                }

                L_dp[i] = total;
            }
        };

        int cur_start = 0;
        while (cur_start < n && qs.V[cur_start] <= u) {
            ++cur_start;
        }

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

        return L_dp.back();
    }

    /**
     * Prime Counting Function: pi(X) via Sublinear Sieve (Lucy DP)
     * Optimized with direct prime sieving and tau coordinate resolution.
     */
    static int64 compute_prime_pi(int64 X) {
        if (X < 2) return 0;
        int64 S = integer_sqrt(X);
        QuotientSpace qs(X);
        const int n = qs.n;

        std::vector<int> primes;
        std::vector<uint8_t> is_p(S + 1, 1);
        for (int64 p = 2; p <= S; ++p) {
            if (is_p[p]) {
                primes.push_back(static_cast<int>(p));
                for (int64 j = p * p; j <= S; j += p) {
                    is_p[j] = 0;
                }
            }
        }

        std::vector<int64> S_arr(n);
        for (int i = 0; i < n; ++i) {
            S_arr[i] = qs.V[i] - 1;
        }

        for (int p : primes) {
            int64 sp = S_arr[p - 2];
            int64 p2 = static_cast<int64>(p) * p;

            for (int i = n - 1; i >= 0; --i) {
                int64 v = qs.V[i];
                if (v < p2) break;
                int64 q = v / p;
                int idx_q = (q <= S) ? static_cast<int>(q - 1) : (n - static_cast<int>(X / q));
                S_arr[i] -= (S_arr[idx_q] - sp);
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

