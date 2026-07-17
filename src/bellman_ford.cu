#include <cfloat>
#include <cstdio>
#include <vector>

#include "bellman_ford.hpp"
#include "device_graph.hpp"

// Helper: check CUDA call succeeded. Prints to stderr on failure.
#define CUDA_OR_DIE(call)                                                                              \
    do {                                                                                               \
        cudaError_t err = (call);                                                                      \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            abort();                                                                                   \
        }                                                                                              \
    } while (0)

// Forward declarations so that earlier functions can call relax kernel.

// Reinterpret double bits as long long so that long-long ordering
// (signed comparison) matches double numeric ordering.
// Positive doubles: unchanged. Negative doubles: flip all bits
// except the sign bit (XOR with 0x7FFFFFFFFFFFFFFF).
// This maps negative doubles to the range [0x8000..., 0xFFFF...]
// which is BELOW 0 in signed long long comparison.
__device__ long long double_to_ordered_longlong(double d) {
    long long bits = __double_as_longlong(d);
    if (bits < 0) {
        bits ^= 0x7FFFFFFFFFFFFFFFLL;
    }
    return bits;
}

// Atomic minimum for double using CAS retry.
// Each iteration: read current raw bits, convert to mapped order for
// fast comparison. Only enters CAS spin-loop if val is actually smaller.
__device__ bool atomicMinDouble(double* addr, double val) {
    unsigned long long* addr_ull   = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long  new_raw    = __double_as_longlong(val);
    long long           new_mapped = double_to_ordered_longlong(val);

    unsigned long long old_raw = *addr_ull;

    while (true) {
        double    old_d      = __longlong_as_double(static_cast<long long>(old_raw));
        long long old_mapped = double_to_ordered_longlong(old_d);
        if (new_mapped >= old_mapped) return false;  // no-op

        unsigned long long cas_result = atomicCAS(addr_ull, old_raw, new_raw);
        if (cas_result == old_raw) return true;  // wrote
        old_raw = cas_result;                    // retry with latest value
    }
}


// ============================================================================
// Phase 6.1: per-block modified flags + reduce kernel.
// ============================================================================
//
// bellman_ford_relax_kernel_v2: identical to v1, but each block tracks its
// own modified flag in d_block_modified[blockIdx.x]. Uses atomicOr (not
// atomicExch) so threads within a block can race-safely merge their bits.
// The host then OR-reduces all block flags into one global bit once,
// instead of polling per iteration.
//
// Phase 6.3 note: an experimental Work-Per-Thread=4 + cp.async version
// was tried and measured with ncu. The cp.async variant did reduce the
// L1TEX-stall rate from 88% to 38% of warp-stall time, but at the same
// time the kernel became 2x slower on V=100k because (a) VPT=4 with
// TPB=64 left only 0.31 waves/SM (vs 0.93 with VPT=1, TPB=256), and
// (b) register pressure doubled to 51 reg/thread, reducing achievable
// occupancy. cp.async would win on V >> 1M where the grid fully fills
// the device; for the current benchmark range VPT=1 is optimal. The
// cp.async code is preserved in git history for future re-evaluation.

__global__ void bellman_ford_relax_kernel_v2(double* __restrict__ d_dist,
                                             const int64_t* __restrict__ d_row_offsets,
                                             const int* __restrict__ d_col_indices,
                                             const double* __restrict__ d_edge_weights,
                                             int* d_block_modified,  // size = gridDim.x
                                             int  num_vertices) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    double dist_u = d_dist[u];
    if (dist_u == DBL_MAX) return;

    int64_t start = __ldg(&d_row_offsets[u]);
    int64_t end   = __ldg(&d_row_offsets[u + 1]);

    int local_modified = 0;
    for (int64_t pos = start; pos < end; ++pos) {
        int    v        = __ldg(&d_col_indices[pos]);
        double w        = __ldg(&d_edge_weights[pos]);
        double new_dist = dist_u + w;
        if (atomicMinDouble(&d_dist[v], new_dist)) {
            local_modified = 1;
        }
    }

    // Per-block reduction via shared memory. Every thread that relaxed an
    // edge ORs its local bit into the shared flag. Thread 0 then publishes
    // the result to d_block_modified[blockIdx.x].
    __shared__ int s_block_modified;
    if (threadIdx.x == 0) s_block_modified = 0;
    __syncthreads();
    if (local_modified) {
        atomicOr(&s_block_modified, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicOr(&d_block_modified[blockIdx.x], s_block_modified);
    }
}

// reduce_block_flags_kernel: OR-reduces d_block_modified[num_blocks] into
// d_global_modified[0]. Single block, strided loop, shared-memory merge.
__global__ void reduce_block_flags_kernel(int* d_block_modified, int* d_global_modified, int num_blocks) {
    __shared__ int shared_flag;

    if (threadIdx.x == 0) shared_flag = 0;
    __syncthreads();

    // Strided loop: each thread OR-merges its slice of d_block_modified.
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        int v = __ldg(&d_block_modified[i]);
        if (v != 0) {
            atomicOr(&shared_flag, 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *d_global_modified = shared_flag;
    }
}

// Block configuration for the V2 kernels. Must match the kernel's
// thread layout. V2_TPB=64 threads, V2_VPT=4 vertices/thread = 256 vertices/block.
// (Definition of V2_VPT, V2_TPB, V2_VPB is above, before the kernels.)

void launch_bellman_ford_init_v2(DeviceGraph& dg, int source, int num_blocks) {
    // Allocate d_dist and d_global_modified (the 1-byte final flag).
    if (dg.d_dist == nullptr) {
        CUDA_OR_DIE(cudaMalloc((void**)&dg.d_dist, dg.num_vertices * sizeof(double)));
    }
    if (dg.d_modified == nullptr) {
        CUDA_OR_DIE(cudaMalloc((void**)&dg.d_modified, sizeof(int)));
    }
    // Allocate per-block flags. Guard against double-allocation (idempotent).
    if (dg.d_block_modified == nullptr) {
        CUDA_OR_DIE(cudaMalloc((void**)&dg.d_block_modified, num_blocks * sizeof(int)));
        dg.num_blocks_v2 = num_blocks;
    }

    // init_distances_kernel still uses 1 vertex/thread (VPT=1): it's a
    // simple 1-op-per-vertex kernel, no benefit from VPT=4.
    int init_threads = 256;
    int init_blocks  = (dg.num_vertices + init_threads - 1) / init_threads;
    init_distances_kernel<<<init_blocks, init_threads>>>(dg.d_dist, dg.d_modified, dg.num_vertices, source);
    // NO cudaDeviceSynchronize here. All V2 launches share the default
    // stream and are therefore serialized by the GPU hardware. The CPU
    // can keep firing launches into the queue.
}

void launch_bellman_ford_relax_iter_v2(DeviceGraph& dg, int num_blocks) {
    // Reset all per-block flags to 0. cudaMemset on the default stream is
    // ordered before the kernel launch below.
    CUDA_OR_DIE(cudaMemsetAsync(dg.d_block_modified, 0, num_blocks * sizeof(int)));

    bellman_ford_relax_kernel_v2<<<num_blocks, V2_TPB>>>(
        dg.d_dist, dg.d_row_offsets, dg.d_col_indices, dg.d_edge_weights, dg.d_block_modified, dg.num_vertices);
    // NO cudaDeviceSynchronize. Let the next iter (or the final reduce) sync.
}

bool launch_bellman_ford_full_v2(DeviceGraph& dg, int source, int max_iters) {
    int num_blocks = (dg.num_vertices + V2_VPB - 1) / V2_VPB;

    launch_bellman_ford_init_v2(dg, source, num_blocks);

    for (int iter = 0; iter < max_iters; ++iter) {
        launch_bellman_ford_relax_iter_v2(dg, num_blocks);
    }

    // Final reduction: OR-reduce all per-block flags into d_modified[0].
    // Async memset + kernel launch, then a SINGLE sync before the PCIe read.
    CUDA_OR_DIE(cudaMemsetAsync(dg.d_modified, 0, sizeof(int)));
    reduce_block_flags_kernel<<<1, 256>>>(dg.d_block_modified, dg.d_modified, num_blocks);
    CUDA_OR_DIE(cudaDeviceSynchronize());

    // Single PCIe round-trip: read the converged byte.
    int h_modified = 0;
    CUDA_OR_DIE(cudaMemcpy(&h_modified, dg.d_modified, sizeof(int), cudaMemcpyDeviceToHost));
    return h_modified == 0;
}
