// =============================================================================
// live_arbitrage.hpp - Public C-API for live_arbitrage.cu
// =============================================================================
//
// PHASE 8.5 REFACTORING: The legacy `live_arb_scan_kernel` (brute-force
// execution over Ticks×Edges) is deprecated and removed. It is replaced by a
// single, deterministic function: `launch_tick_to_edge_update()`. This kernel
// mutates live ticks directly into the CSR `edge_weights` structure. The
// subsequent BF-V2 kernel (Phase 6) then processes these mutated weights to
// execute the core arbitrage detection.
//
// SCHEMA A: node_id == Edge-Index. Refer to live_arbitrage.cu for mapping details.
// =============================================================================
#pragma once

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#else
typedef int cudaError_t;
#endif

#include <cstdint>

#include "live_arbitrage_types.hpp"
#include "market_tick.hpp"

namespace phase7 {

// =============================================================================
// tick_to_edge_update
// -----------------------------------------------------------------------------
// Mutates d_edge_weights IN-PLACE: for every tick in the batch, assigns:
//   d_edge_weights[tick.node_id] = -log(tick.bid)
// Ticks exhibiting bid <= 0.0 or node_id >= num_edges are safely bypassed.
//
// PARAMETERS:
//   d_ticks_new       : Device pointer targeting `batch_size` MarketTicks (32B
//                       each, 16-byte aligned). Populated via cudaMemcpyAsync.
//   batch_size        : Total tick count. 0 results in a no-op.
//   d_csc_edge_weights: Device pointer targeting `num_edges` doubles (CSR format).
//                       Subject to IN-PLACE mutation.
//   d_orig_to_csc_idx : Mapping array.
//   num_edges         : Edge count (= Node count under Schema A).
//
// RETURNS:
//   cudaSuccess            : Kernel dispatch successful (NO implicit sync).
//   cudaErrorInvalidValue  : Nullptr argument encountered.
//   cudaError              : Kernel launch failure (via cudaGetErrorString).
//
// THREAD-SAFETY: Kernel dispatches asynchronously on the default stream.
// Callers MUST invoke cudaDeviceSynchronize() or cudaStreamSynchronize() prior
// to dependent kernels (e.g., BF) reading from d_edge_weights.
// =============================================================================
extern "C" cudaError_t launch_tick_to_edge_update(const MarketTick* d_ticks_new,
                                                  int               batch_size,
                                                  float*            d_csc_edge_weights,
                                                  const int64_t*    d_orig_to_csc_idx,
                                                  int64_t           num_edges);

}  // namespace phase7
