// =============================================================================
// arbitrage_optimized.hpp - Phase 8.7: Optimized Arbitrage Detection
// =============================================================================
// Header for the optimized cascade detection variant in arbitrage_optimized.cu.
// Currently provides the host wrapper; subcommand wrapper for main.cpp to follow.
// =============================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "device_graph.hpp"

namespace phase7 {
namespace optimized {

// Cascade detection following Bellman-Ford execution. The caller MUST invoke
// launch_bellman_ford_full_v2 prior to this, ensuring d_dist is populated
// and (idealerweise) non-convergent. If did_converge = true is passed
// via pointer, the function immediately returns an empty vector.
//
// HISTORICAL: arbitrare_detection_v2 in bellman_ford.cu. ADVANTAGES here:
//   - WPT=2 (work-per-thread: 2 vertices/thread, 1 dist-read per 2 vertices).
//   - Coalesced flag writes per warp (eliminating scattered global access).
//   - Early-Termination: Execution halts immediately if a block detects
//     zero new flags, drastically reducing iterations for small negative cycles.
//
// PARAMETERS:
//   dg: DeviceGraph with d_dist pre-populated by BF-V2.
//   source: Ignored (maintained solely for V1 signature compatibility).
//   out_did_converge: Optional. If != nullptr, the value *out_did_converge
//                     is treated as the BF convergence status. If true,
//                     returns {}. The pointer value is NOT mutated.
//
// RETURNS:
//   std::vector of edge indices constituting a negative cycle.
std::vector<int64_t> launch_arbitrage_detection_optimized(DeviceGraph& dg,
                                                          int          source,
                                                          bool*        out_did_converge = nullptr);

// Phase 8.7.4: Persistent-Kernel Variant. Identical semantic output to above,
// but launches a SINGLE kernel executing the entire V-iteration cascade
// autonomously via grid.sync(). This eliminates 100k kernel-launch overheads
// and OS driver queue contention. Requires cudaLaunchCooperativeKernel.
std::vector<int64_t> launch_arbitrage_detection_persistent(DeviceGraph& dg,
                                                           int          source,
                                                           bool*        out_did_converge = nullptr);

// Phase 9.2: Dense-Micro-Graph Bellman-Ford.
// Strictly limited to V <= 1024 (executes on a single SM, single block).
// For V > 1024, the wrapper automatically falls back to the persistent variant
// (emitting a warning to stdout).
void launch_arbitrage_detection_micro(const DeviceGraph& dg, ArbitrageOpportunity* d_opp);

}  // namespace optimized
}  // namespace phase7
