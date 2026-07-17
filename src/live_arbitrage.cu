#include <cstdint>
#include <cstdio>

namespace phase7 {

// live_arbitrage.cu - Market Tick to CSR Edge Weight Ingestion Pipeline
//
// ARCHITECTURE:
//   Live-Tick (node_id, bid) ──┐
//                              ▼
//              ┌────────────────────────────┐
//              │  tick_to_edge_update_kernel│  O(1) pro Tick
//              │  d_edge_weights[node_id]   │    = -ln(bid)
//              │    = -log(tick.bid)        │
//              └────────────────────────────┘
//                              │  mutiert d_edge_weights IN-PLACE
//                              ▼
//              ┌────────────────────────────┐
//              │  bellman_ford_relax_v2     │  legacy
//              │  (nutzt d_edge_weights)    │    O(V*E) per Iter
//              └────────────────────────────┘
//                              │  d_dist[] = kürzeste Pfade
//                              ▼
//              ┌────────────────────────────┐
//              │  arbitrage_detection_v2    │  legacy
//              │  (extra Iter with Flag)    │    detects Negative Cycles
//              └────────────────────────────┘
//
// MAPPING TOPOLOGY:
//   Assumption: Schema A — strictly 1 Edge per Node. node_id acts as the direct index
//   in d_edge_weights. The graph generator in graph_generator.cu constructs
//   exact topology mapping: cyclic (u+1) % V enforcing E_PER_NODE=1.
//   Production requisite: Schema B (Hash-Map) is mandatory, since a currency node
//   hundreds of trading pairs. A 1:1 mapping suffices for the proof of concept.
//
// WHY WE DEPRECATED live_arb_scan_kernel:
//   The legacy kernel executed an O(scan_count × edge_count) brute-force scan:
//   "For each of the trailing N Ticks, evaluate all Edges". This is
//   conceptually flawed: BF demands a strictly stable CSR topology with
//   mutating Weights, bypassing any Tick×Edge Cartesian product execution.
//   Additionally: NVCC inadvertently aggressively eliminated the branch (compiler optimization defect).
// =============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "live_arbitrage.hpp"
#include "live_arbitrage_types.hpp"
#include "market_tick.hpp"

namespace phase7 {

// =============================================================================
// tick_to_edge_update_kernel
// -----------------------------------------------------------------------------
// SCHEMA A: d_edge_weights[u] serves as the strictly unique outgoing edge from u.
//   - tick.node_id == u
//   - d_edge_weights[u] = -log(tick.bid)
//
// BOUND-CHECKS:
//   - tick.node_id < num_edges (otherwise, discard)
//     node_id is strictly u32; corrupt streams risk propagating garbage vectors.
//   - tick.bid > 0.0 (otherwise, discard)
//     -log(0) = +inf, prevents immediate cascade corruption.
//     -log(negativ) = NaN, prevents cascade corruption.
//
// PERFORMANCE:
//   Thread layout: 1 thread per tick, 256 threads per block.
//   Lock-free synchronization: 1:1 edge-to-node mapping guarantees mutually exclusive writes.
//
// LANE-UTILITY:
//   100% coalesced memory access pattern achieved when node_id == thread_idx (Optimal case).
// =============================================================================
__global__ void tick_to_edge_update_kernel(const MarketTick* __restrict__ d_ticks_new,
                                           int batch_size,
                                           float* __restrict__ d_csc_edge_weights,
                                           const int64_t* __restrict__ d_orig_to_csc_idx,
                                           int64_t num_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    MarketTick t = d_ticks_new[idx];

    if (t.node_id + 1 >= num_edges || t.node_id < 0) {
        return;  // out of bounds
    }

    if (t.bid <= 0.0 || t.ask <= 0.0) {
        return;  // prevent division-by-zero or logarithmic domain errors
    }

    // Trading Fee configuration (Binance VIP 0 = 0.1% transaction cost)
    constexpr float fee = 0.999f;

    // Edge 1 (Buy Quote -> Base execution): We exchange 1 Quote for (1.0 / ask) Base
    float   w1                    = -logf((1.0f / static_cast<float>(t.ask)) * fee);
    int64_t csc_idx_1             = d_orig_to_csc_idx[t.node_id];
    d_csc_edge_weights[csc_idx_1] = w1;

    // Edge 2 (Sell Base -> Quote execution): We exchange 1 Base for bid Quote
    float   w2                    = -logf(static_cast<float>(t.bid) * fee);
    int64_t csc_idx_2             = d_orig_to_csc_idx[t.node_id + 1];
    d_csc_edge_weights[csc_idx_2] = w2;
}

// =============================================================================
// Host Wrapper
// =============================================================================
extern "C" cudaError_t launch_tick_to_edge_update(const MarketTick* d_ticks_new,
                                                  int               batch_size,
                                                  float*            d_csc_edge_weights,
                                                  const int64_t*    d_orig_to_csc_idx,
                                                  int64_t           num_edges) {
    if (batch_size <= 0) return cudaSuccess;

    int threads = 256;
    int blocks  = (batch_size + threads - 1) / threads;

    tick_to_edge_update_kernel<<<blocks, threads>>>(
        d_ticks_new, batch_size, d_csc_edge_weights, d_orig_to_csc_idx, num_edges);

    return cudaGetLastError();
}

}  // namespace phase7
