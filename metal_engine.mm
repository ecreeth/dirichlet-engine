#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_engine.hpp"
#include "dirichlet_engine.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <omp.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace dirichlet {

static const char* metal_shader_code = R"(
#include <metal_stdlib>
using namespace metal;

kernel void solve_mertens_antichain(
    device const int64_t* V          [[buffer(0)]],
    device const int32_t* M_sieve    [[buffer(1)]],
    device int64_t*       M_sparse   [[buffer(2)]],
    constant int64_t&     X          [[buffer(3)]],
    constant int64_t&     S          [[buffer(4)]],
    constant int64_t&     u          [[buffer(5)]],
    constant int32_t&     n          [[buffer(6)]],
    constant int32_t&     start_idx  [[buffer(7)]],
    constant int32_t&     num_states [[buffer(8)]],
    uint thread_idx                  [[thread_position_in_grid]]
) {
    if (thread_idx >= (uint)num_states) return;
    int idx = start_idx + (int)thread_idx;
    int64_t v = V[idx];
    if (v <= u) return;

    int64_t K = (int64_t)sqrt((float)v);
    while ((K + 1) * (K + 1) <= v) ++K;
    while (K * K > v) --K;

    int64_t k_max = v / (K + 1);
    int64_t total = 1;

    int64_t k_split = v / (u + 1);
    int64_t k_lim = (k_max < k_split) ? k_max : k_split;

    // Part 1a: v/k > u (from sparse DP array)
    for (int64_t k = 2; k <= k_lim; ++k) {
        int64_t q = v / k;
        int32_t dep_idx = (n - (int32_t)(X / q)) - (int32_t)S;
        total -= M_sparse[dep_idx];
    }

    // Part 1b: v/k <= u (from linear sieve buffer)
    for (int64_t k = k_lim + 1; k <= k_max; ++k) {
        int64_t q = v / k;
        total -= (int64_t)M_sieve[q];
    }

    // Part 2: Dense prefix q <= K
    int64_t q = 1;
    int64_t v_div_q = v;
    for (; q <= K; ++q) {
        int64_t v_div_q_next = v / (q + 1);
        int64_t count = v_div_q - v_div_q_next;
        total -= count * (int64_t)M_sieve[q];
        v_div_q = v_div_q_next;
    }

    M_sparse[idx - (int32_t)S] = total;
}
)";

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> mertens_pipeline = nil;
    bool initialized = false;

    static MetalContext& instance() {
        static MetalContext ctx;
        if (!ctx.initialized) {
            @autoreleasepool {
                ctx.device = MTLCreateSystemDefaultDevice();
                if (ctx.device) {
                    ctx.queue = [ctx.device newCommandQueue];
                    NSError* error = nil;
                    NSString* src = [NSString stringWithUTF8String:metal_shader_code];
                    id<MTLLibrary> lib = [ctx.device newLibraryWithSource:src options:nil error:&error];
                    if (lib) {
                        id<MTLFunction> fn = [lib newFunctionWithName:@"solve_mertens_antichain"];
                        ctx.mertens_pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&error];
                        ctx.initialized = (ctx.mertens_pipeline != nil);
                    }
                }
            }
        }
        return ctx;
    }
};

bool MetalDirichletEngine::is_available() {
    return MetalContext::instance().initialized;
}

int64 MetalDirichletEngine::compute_mertens(int64 X, int cpu_threads) {
    if (X < 1) return 0;
    auto& ctx = MetalContext::instance();
    if (!ctx.initialized) {
        return DirichletEngine::compute_mertens(X, cpu_threads);
    }

    int64 S = integer_sqrt(X);
    double c_ratio = (cpu_threads > 1) ? (0.20 / std::cbrt(static_cast<double>(cpu_threads))) : 0.80;
    int64 target_u = static_cast<int64>(c_ratio * std::pow(static_cast<double>(X), 2.0 / 3.0));
    if (target_u < S) target_u = S;
    int64 u = std::min(target_u, 100000000LL);

    // Build quotient space
    QuotientSpace qs(X);
    const int n = qs.n;

    // Odd-only linear pre-sieve
    std::vector<int> M_sieve(u + 1, 0);
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

    const int sparse_size = n - static_cast<int>(S);

    @autoreleasepool {
        // Zero-copy unified memory allocations
        id<MTLBuffer> buf_V = [ctx.device newBufferWithBytes:qs.V.data() length:qs.V.size() * sizeof(int64) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sieve = [ctx.device newBufferWithBytes:M_sieve.data() length:M_sieve.size() * sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sparse = [ctx.device newBufferWithLength:sparse_size * sizeof(int64) options:MTLResourceStorageModeShared];

        int64* sparse_ptr = (int64*)[buf_sparse contents];
        memset(sparse_ptr, 0, sparse_size * sizeof(int64));

        // Initialize sparse states S < v <= u directly from pre-sieve buffer
        #pragma omp parallel for schedule(static, 4096) num_threads(cpu_threads)
        for (int i = static_cast<int>(S); i < n; ++i) {
            if (qs.V[i] <= u) {
                sparse_ptr[i - static_cast<int>(S)] = M_sieve[qs.V[i]];
            }
        }

        int cur_start = 0;
        while (cur_start < n && qs.V[cur_start] <= u) {
            ++cur_start;
        }

        auto solve_cpu_block = [&](int start_idx, int end_idx) {
            #pragma omp parallel for schedule(dynamic, 16) num_threads(cpu_threads)
            for (int i = start_idx; i < end_idx; ++i) {
                int64 v = qs.V[i];
                double dv = static_cast<double>(v);
                int64 K = static_cast<int64>(std::sqrt(dv));
                int64 k_max = static_cast<int64>(dv / static_cast<double>(K + 1));
                int64 total = 1;

                if (v <= 9007199254740992LL) {
                    int64 k_split = v / (u + 1);
                    int64 k_lim = std::min(k_max, k_split);

                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = static_cast<int64>(dv / static_cast<double>(k));
                        int idx = (n - static_cast<int>(X / q)) - static_cast<int>(S);
                        total -= sparse_ptr[idx];
                    }
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
                        
                        total -= (v_div_q - n1) * static_cast<int64>(M_sieve[q]);
                        total -= (n1 - n2) * static_cast<int64>(M_sieve[q + 1]);
                        total -= (n2 - n3) * static_cast<int64>(M_sieve[q + 2]);
                        total -= (n3 - n4) * static_cast<int64>(M_sieve[q + 3]);
                        v_div_q = n4;
                    }
                    #endif

                    for (; q <= K; ++q) {
                        int64 v_div_q_next = static_cast<int64>(dv / static_cast<double>(q + 1));
                        int64 count = v_div_q - v_div_q_next;
                        total -= count * static_cast<int64>(M_sieve[q]);
                        v_div_q = v_div_q_next;
                    }
                } else {
                    int64 k_split = v / (u + 1);
                    int64 k_lim = std::min(k_max, k_split);

                    for (int64 k = 2; k <= k_lim; ++k) {
                        int64 q = v / k;
                        int idx = (n - static_cast<int>(X / q)) - static_cast<int>(S);
                        total -= sparse_ptr[idx];
                    }
                    for (int64 k = k_lim + 1; k <= k_max; ++k) {
                        total -= M_sieve[v / k];
                    }

                    int64 q = 1;
                    int64 v_div_q = v;
                    for (; q <= K; ++q) {
                        int64 v_div_q_next = v / (q + 1);
                        int64 count = v_div_q - v_div_q_next;
                        total -= count * static_cast<int64>(M_sieve[q]);
                        v_div_q = v_div_q_next;
                    }
                }

                sparse_ptr[i - static_cast<int>(S)] = total;
            }
        };

        while (cur_start < n) {
            int64 max_safe = qs.V[cur_start - 1] * 2;
            int cur_end = cur_start;
            while (cur_end < n && qs.V[cur_end] <= max_safe) {
                ++cur_end;
            }
            if (cur_end == cur_start) cur_end = cur_start + 1;

            int num_states = cur_end - cur_start;

            // Route wide early antichains to GPU with chunked dispatch to prevent watchdog timeout
            if (num_states >= 256 && qs.V[cur_end - 1] <= 5000000000LL) {
                const int chunk_size = 65536;
                for (int chunk_start = cur_start; chunk_start < cur_end; chunk_start += chunk_size) {
                    int chunk_count = std::min(chunk_size, cur_end - chunk_start);

                    id<MTLCommandBuffer> cmd_buf = [ctx.queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

                    [enc setComputePipelineState:ctx.mertens_pipeline];
                    [enc setBuffer:buf_V offset:0 atIndex:0];
                    [enc setBuffer:buf_sieve offset:0 atIndex:1];
                    [enc setBuffer:buf_sparse offset:0 atIndex:2];
                    [enc setBytes:&X length:sizeof(int64) atIndex:3];
                    [enc setBytes:&S length:sizeof(int64) atIndex:4];
                    [enc setBytes:&u length:sizeof(int64) atIndex:5];
                    [enc setBytes:&n length:sizeof(int) atIndex:6];
                    [enc setBytes:&chunk_start length:sizeof(int) atIndex:7];
                    [enc setBytes:&chunk_count length:sizeof(int) atIndex:8];

                    NSUInteger w = ctx.mertens_pipeline.threadExecutionWidth;
                    MTLSize threadsPerGrid = MTLSizeMake(chunk_count, 1, 1);
                    MTLSize threadsPerGroup = MTLSizeMake(std::min((NSUInteger)chunk_count, w), 1, 1);

                    [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
                    [enc endEncoding];
                    [cmd_buf commit];
                    [cmd_buf waitUntilCompleted];
                }
            } else {
                solve_cpu_block(cur_start, cur_end);
            }

            cur_start = cur_end;
        }

        return sparse_ptr[sparse_size - 1];
    }
}

int128 MetalDirichletEngine::compute_totient_sum(int64 X, int cpu_threads) {
    return DirichletEngine::compute_totient_sum(X, cpu_threads);
}

int64 MetalDirichletEngine::compute_liouville_sum(int64 X, int cpu_threads) {
    return DirichletEngine::compute_liouville_sum(X, cpu_threads);
}

} // namespace dirichlet
