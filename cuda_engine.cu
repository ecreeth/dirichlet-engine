#include "cuda_engine.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

namespace dirichlet {

inline int64 integer_sqrt_cpu(int64 n) {
    if (n <= 0) return 0;
    int64 s = static_cast<int64>(std::sqrt(static_cast<double>(n)));
    while ((s + 1) * (s + 1) <= n) ++s;
    while (s * s > n) --s;
    return s;
}

void CUDADirichletEngine::print_device_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "No CUDA-capable GPU detected.\n";
        return;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "--- GPU Device " << i << ": " << prop.name << " ---\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Global VRAM:  " << std::fixed << std::setprecision(2) 
                  << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n";
        std::cout << "  SM Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max Threads / Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Memory Bus Width:    " << prop.memoryBusWidth << " bits\n";
        std::cout << "  L2 Cache Size:       " << prop.l2CacheSize / 1024 << " KB\n";
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Mertens Function Antichain Stage
// -----------------------------------------------------------------------------
__global__ void mertens_stage_kernel(
    const int64_t* __restrict__ V,
    const int32_t* __restrict__ M_sieve,
    int64_t* __restrict__ M_sparse,
    int64_t X,
    int64_t S,
    int64_t u,
    int32_t n,
    int32_t start_idx,
    int32_t num_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    int idx = start_idx + tid;
    int64_t v = V[idx];
    if (v <= u) return;

    double dv = (double)v;
    int64_t K = (int64_t)sqrt(dv);
    while ((K + 1) * (K + 1) <= v) ++K;
    while (K * K > v) --K;

    int64_t k_max = (v <= 9007199254740992LL) ? (int64_t)(dv / (double)(K + 1)) : (v / (K + 1));
    int64_t total = 1;

    int64_t k_split = v / (u + 1);
    int64_t k_lim = (k_max < k_split) ? k_max : k_split;

    if (v <= 9007199254740992LL) {
        // Fast float64 / double division path for v <= 2^53
        for (int64_t k = 2; k <= k_lim; ++k) {
            int64_t q = (int64_t)(dv / (double)k);
            int32_t dep_idx = (n - (int32_t)(X / q)) - (int32_t)S;
            total -= M_sparse[dep_idx];
        }
        for (int64_t k = k_lim + 1; k <= k_max; ++k) {
            int64_t q = (int64_t)(dv / (double)k);
            total -= (int64_t)M_sieve[q];
        }

        int64_t q = 1;
        int64_t v_div_q = v;
        for (; q <= K; ++q) {
            int64_t v_div_q_next = (int64_t)(dv / (double)(q + 1));
            int64_t count = v_div_q - v_div_q_next;
            total -= count * (int64_t)M_sieve[q];
            v_div_q = v_div_q_next;
        }
    } else {
        // Exact 64-bit integer path for v > 2^53
        for (int64_t k = 2; k <= k_lim; ++k) {
            int64_t q = v / k;
            int32_t dep_idx = (n - (int32_t)(X / q)) - (int32_t)S;
            total -= M_sparse[dep_idx];
        }
        for (int64_t k = k_lim + 1; k <= k_max; ++k) {
            int64_t q = v / k;
            total -= (int64_t)M_sieve[q];
        }

        int64_t q = 1;
        int64_t v_div_q = v;
        for (; q <= K; ++q) {
            int64_t v_div_q_next = v / (q + 1);
            int64_t count = v_div_q - v_div_q_next;
            total -= count * (int64_t)M_sieve[q];
            v_div_q = v_div_q_next;
        }
    }

    M_sparse[idx - (int32_t)S] = total;
}

int64 CUDADirichletEngine::compute_mertens(int64 X, int block_size) {
    if (X < 1) return 0;
    int64 S = integer_sqrt_cpu(X);
    int64 target_u = static_cast<int64>(0.20 * std::pow(static_cast<double>(X), 2.0 / 3.0));
    if (target_u < S) target_u = S;
    int64 u = std::min(target_u, 100000000LL);

    // Host: Odd-only linear pre-sieve
    std::vector<int32_t> M_sieve(u + 1, 0);
    {
        int half_u = static_cast<int>((u + 1) / 2);
        std::vector<int8_t> mu_odd(half_u + 1, 0);
        std::vector<int> primes;
        primes.reserve(static_cast<size_t>(u / 10));
        std::vector<uint8_t> is_prime(half_u + 1, 1);
        mu_odd[0] = 0; mu_odd[1] = 1;
        for (int i = 2; 2 * i - 1 <= u; ++i) {
            int num = 2 * i - 1;
            if (is_prime[i]) { primes.push_back(num); mu_odd[i] = -1; }
            for (size_t j = 0; j < primes.size(); ++j) {
                int p = primes[j];
                int64 prod = static_cast<int64>(num) * p;
                if (prod > u) break;
                int idx = static_cast<int>((prod + 1) / 2);
                is_prime[idx] = 0;
                if (num % p == 0) { mu_odd[idx] = 0; break; }
                else mu_odd[idx] = -mu_odd[i];
            }
        }
        int run_sum = 0;
        for (int64 i = 1; i <= u; ++i) {
            int8_t m;
            if (i % 2 != 0) m = mu_odd[(i + 1) / 2];
            else if ((i / 2) % 2 != 0) m = -mu_odd[((i / 2) + 1) / 2];
            else m = 0;
            run_sum += m;
            M_sieve[i] = run_sum;
        }
    }

    if (X <= u) return M_sieve[X];

    // Host: Generate Quotient Space V
    std::vector<int64> V;
    V.reserve(static_cast<size_t>(2 * S + 5));
    for (int64 i = 1; i <= S; ++i) V.push_back(i);
    for (int64 k = S; k >= 1; --k) {
        int64 v = X / k;
        if (v > S && v != V.back()) V.push_back(v);
    }
    int32_t n = static_cast<int32_t>(V.size());
    int32_t sparse_size = n - static_cast<int32_t>(S);

    // Host: Sparse initialization for S < v <= u
    std::vector<int64_t> host_sparse(sparse_size, 0);
    for (int32_t i = static_cast<int32_t>(S); i < n; ++i) {
        if (V[i] <= u) {
            host_sparse[i - static_cast<int32_t>(S)] = M_sieve[V[i]];
        } else {
            break;
        }
    }

    // Allocate GPU Device Memory (16 GB VRAM on T4)
    int64_t* d_V = nullptr;
    int32_t* d_sieve = nullptr;
    int64_t* d_sparse = nullptr;

    CUDA_CHECK(cudaMalloc(&d_V, V.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_sieve, M_sieve.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_sparse, sparse_size * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_V, V.data(), V.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sieve, M_sieve.data(), M_sieve.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sparse, host_sparse.data(), sparse_size * sizeof(int64_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int32_t cur_start = 0;
    while (cur_start < n && V[cur_start] <= u) {
        ++cur_start;
    }

    // Lock-Free Doubling-Stage DAG Scheduler on GPU
    while (cur_start < n) {
        int64_t max_safe = V[cur_start - 1] * 2;
        int32_t cur_end = cur_start;
        while (cur_end < n && V[cur_end] <= max_safe) {
            ++cur_end;
        }
        if (cur_end == cur_start) cur_end = cur_start + 1;

        int32_t num_states = cur_end - cur_start;
        int grid_blocks = (num_states + block_size - 1) / block_size;

        mertens_stage_kernel<<<grid_blocks, block_size, 0, stream>>>(
            d_V, d_sieve, d_sparse, X, S, u, n, cur_start, num_states
        );

        cur_start = cur_end;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Retrieve Final Result M(X) = M_sparse[sparse_size - 1]
    int64_t final_res = 0;
    CUDA_CHECK(cudaMemcpy(&final_res, &d_sparse[sparse_size - 1], sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_sieve));
    CUDA_CHECK(cudaFree(d_sparse));

    return final_res;
}

int128 CUDADirichletEngine::compute_totient_sum(int64 X, int block_size) {
    return 0; // Totient specialization
}

int64 CUDADirichletEngine::compute_liouville_sum(int64 X, int block_size) {
    return 0; // Liouville specialization
}

int64 CUDADirichletEngine::compute_prime_pi(int64 X, int block_size) {
    return 0; // PrimePi specialization
}

} // namespace dirichlet
