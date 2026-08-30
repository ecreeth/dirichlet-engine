#ifndef METAL_ENGINE_HPP
#define METAL_ENGINE_HPP

#include <string>
#include <cstdint>

namespace dirichlet {

using int64 = long long;
using int128 = __int128_t;

/**
 * MetalDirichletEngine: Apple Silicon Metal GPU & Heterogeneous Antichain Scheduler.
 * Leverages Apple Unified Memory (zero-copy CPU/GPU sharing) and dynamically routes
 * wide antichains to GPU shaders and high-intensity stages to multi-threaded CPU.
 */
class MetalDirichletEngine {
public:
    /**
     * Checks if Apple Metal compute device and shaders are initialized.
     */
    static bool is_available();

    /**
     * Computes Mertens function M(X) using Heterogeneous CPU+GPU Antichain Scheduling.
     */
    static int64 compute_mertens(int64 X, int cpu_threads = 8);

    /**
     * Computes Totient summatory function Phi(X) using Heterogeneous CPU+GPU.
     */
    static int128 compute_totient_sum(int64 X, int cpu_threads = 8);

    /**
     * Computes Liouville summatory function L(X) using Heterogeneous CPU+GPU.
     */
    static int64 compute_liouville_sum(int64 X, int cpu_threads = 8);
};

} // namespace dirichlet

#endif // METAL_ENGINE_HPP
