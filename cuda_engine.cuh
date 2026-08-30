#ifndef CUDA_ENGINE_CUH
#define CUDA_ENGINE_CUH

#include <cstdint>
#include <string>

namespace dirichlet {

using int64 = long long;
using uint64 = unsigned long long;
using int128 = __int128_t;

/**
 * CUDADirichletEngine: Massively Parallel NVIDIA CUDA Dirichlet Convolution Engine.
 * Optimized for NVIDIA T4 16GB (Turing sm_75), V100 (sm_70), A100 (sm_80), and RTX GPUs.
 * 
 * Exploits lock-free Doubling-Stage DAG Antichains over GPU warp blocks with 
 * coalesced memory access, hardware-accelerated 64-bit integer division, and L1 caching.
 */
class CUDADirichletEngine {
public:
    /**
     * Prints GPU device properties and VRAM capacity.
     */
    static void print_device_info();

    /**
     * Solves Mertens function M(X) = sum_{n<=X} mu(n) entirely on NVIDIA GPU.
     * @param X Target scale (e.g. 10^12 to 10^17)
     * @param block_size Number of CUDA threads per block (default: 256)
     */
    static int64 compute_mertens(int64 X, int block_size = 256);

    /**
     * Solves Totient Summatory function Phi(X) = sum_{n<=X} phi(n) on NVIDIA GPU.
     */
    static int128 compute_totient_sum(int64 X, int block_size = 256);

    /**
     * Solves Liouville Summatory function L(X) = sum_{n<=X} lambda(n) on NVIDIA GPU.
     */
    static int64 compute_liouville_sum(int64 X, int block_size = 256);

    /**
     * Solves Prime Counting function pi(X) on NVIDIA GPU.
     */
    static int64 compute_prime_pi(int64 X, int block_size = 256);
};

} // namespace dirichlet

#endif // CUDA_ENGINE_CUH
