#pragma once
#include <cstdint>

struct DeviceGraph;

/// Allocates and fills col_indices and edge_weights directly on the GPU.
/// row_offsets must already be allocated on device (d_row_offsets).
/// num_vertices: number of nodes
/// edges_per_node: constant edge count per node
void launch_graph_generator(DeviceGraph& d_graph, uint32_t edges_per_node);