#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using int64 = long long;

/**
 * Computes the Mertens function M(X) = sum_{n=1}^X mu(n) using the Dirichlet hyperbola DP.
 *
 * Direct O(1) indexing:
 *   For state q in V (sorted ascending):
 *     if q <= S: index is q - 1
 *     if q > S:  index is |V| - (X / q)
 *
 * Thread-safe parallelization:
 *   Since M(v) only depends on M(q) for q <= v/2, we process states in doubling rounds
 *   [U, 2U]. In each round, all required dependencies q <= U are already fully computed,
 *   allowing race-free parallel evaluation of all v in (U, 2U].
 */
int64 mertens_hyperbola(int64 X) {
    if (X < 1) return 0;

    // 1. Collect all distinct values floor(X / i)
    std::vector<int64> V;
    for (int64 i = 1; i <= X; ) {
        int64 v = X / i;
        V.push_back(v);
        i = X / v + 1;
    }
    std::reverse(V.begin(), V.end());  // ascending

    int64 S = std::sqrt(X);
    while ((S + 1) * (S + 1) <= X) ++S;
    while (S * S > X) --S;

    int n = (int)V.size();
    std::vector<int> M(n, 0);

    auto get_idx = [&](int64 q) -> int {
        return (q <= S) ? (int)(q - 1) : (n - (int)(X / q));
    };

    // Sequential base case for small values (e.g. up to 1000)
    int base_idx = 0;
    while (base_idx < n && V[base_idx] <= 1000) {
        int64 v = V[base_idx];
        if (v == 1) {
            M[base_idx] = 1;
        } else {
            int total = 1;
            int64 k = 2;
            while (k <= v) {
                int64 q = v / k;
                int64 next_k = v / q + 1;
                total -= (int)((next_k - k) * M[get_idx(q)]);
                k = next_k;
            }
            M[base_idx] = total;
        }
        ++base_idx;
    }

    // Doubling rounds: all v in (U, 2U] depend only on q <= U (already computed)
    int cur_start = base_idx;
    while (cur_start < n) {
        int64 max_safe = V[cur_start - 1] * 2;
        int cur_end = cur_start;
        while (cur_end < n && V[cur_end] <= max_safe) {
            ++cur_end;
        }
        if (cur_end == cur_start) cur_end = cur_start + 1;

        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = cur_start; i < cur_end; ++i) {
            int64 v = V[i];
            int total = 1;
            int64 k = 2;
            while (k <= v) {
                int64 q = v / k;
                int64 next_k = v / q + 1;
                total -= (int)((next_k - k) * M[get_idx(q)]);
                k = next_k;
            }
            M[i] = total;
        }

        cur_start = cur_end;
    }

    return M.back();
}

int main(int argc, char* argv[]) {
    int64 X = 1000000000000LL;  // 10^12 default for quick test
    if (argc > 1) {
        X = std::stoll(argv[1]);
    }

    std::cout << "Computing M(" << X << ") ...\n";
    auto start = std::chrono::steady_clock::now();
    int64 result = mertens_hyperbola(X);
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "M(" << X << ") = " << result << "\n";
    std::cout << "Time: " << elapsed << " s\n";
    double ratio = std::abs(result) / std::sqrt((double)X);
    std::cout << "|M|/sqrt(X) = " << ratio << "\n";
    return 0;
}