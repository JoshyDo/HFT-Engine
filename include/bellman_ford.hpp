#pragma once

#include <vector>

#include "device_graph.hpp"


// =============================================================================
// Phase 6.1: PCIe-bottleneck-free Bellman-Ford.
// =============================================================================
//
// The original `launch_bellman_ford_full` reads d_modified[0] from the device
// after EVERY kernel launch (V PCIe round-trips). At V=100M this is the
// dominant cost — ~1000 seconds of pure latency.
//
// The new design:
//   1. `launch_bellman_ford_init_v2` allocates a per-block modified-flag
//      array (`d_block_modified`, size = num_blocks).
//   2. `launch_bellman_ford_relax_iter_v2` resets ALL per-block flags to 0
//      and uses `atomicOr` (not just one global atomic write) so each block
//      records whether any of its threads relaxed a distance.
//   3. `launch_bellman_ford_full_v2` runs the loop WITHOUT host round-trips,
//      then does ONE final reduce-kernel that OR-reduces all per-block flags
//      into a single global flag, and reads that one byte back.
//
// Result: V kernel launches, but only 1 PCIe round-trip.

void launch_bellman_ford_init_v2(DeviceGraph& dg, int source, int num_blocks);
void launch_bellman_ford_relax_iter_v2(DeviceGraph& dg, int num_blocks);
bool launch_bellman_ford_full_v2(DeviceGraph& dg, int source, int max_iters);
