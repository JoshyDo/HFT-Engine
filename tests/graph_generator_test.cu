// Tests for graph_generator.cu — GPU-native graph generation.
//
// The kernel fills col_indices and edge_weights with a deterministic
// cyclic pattern: node u connects to (u+1) % V, (u+2) % V, ...
// with a fixed weight. We test the post-conditions on the device buffers.
//
// Properties to verify:
//   - Every edge index has a valid col_indices value in [0, V).
//   - Every edge weight is the fixed value (here: -log(1.5)).
//   - row_offsets is unchanged (we only fill col_indices and edge_weights).
//   - Self-loops: with edges_per_node = V, node V-1 connects to node 0
//     (the cyclic wrap), and node V-1 also has a self-edge (when e = V-1).
//   - For small graphs, all expected edges appear with correct targets.

#include <gtest/gtest.h>
#include <cfloat>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "device_graph.hpp"
#include "graph_generator.hpp"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        ASSERT_EQ(err, cudaSuccess)                                           \
            << "CUDA error: " << cudaGetErrorString(err)                      \
            << " at " << __FILE__ << ":" << __LINE__;                        \
    } while (0)

TEST(GraphGeneratorTest, ProducesExpectedTargetsForSmallGraph) {
    // Build a 5-node graph with 3 edges per node.
    constexpr int V = 5;
    constexpr uint32_t E_PER_NODE = 3;

    std::vector<int64_t> row_offsets(V + 1);
    for (int i = 0; i <= V; ++i) {
        row_offsets[i] = static_cast<int64_t>(i) * E_PER_NODE;
    }

    DeviceGraph dg = DeviceGraph::create_device_only(row_offsets, V, V * E_PER_NODE);
    launch_graph_generator(dg, E_PER_NODE);

    // Copy col_indices and edge_weights back.
    std::vector<int> h_col(V * E_PER_NODE);
    std::vector<double> h_w(V * E_PER_NODE);
    CUDA_CHECK(cudaMemcpy(h_col.data(), dg.d_col_indices,
                          h_col.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w.data(), dg.d_edge_weights,
                          h_w.size() * sizeof(double), cudaMemcpyDeviceToHost));

    // Targets: cyclic shift by 1..E_PER_NODE, with wrap to 0.
    for (int u = 0; u < V; ++u) {
        for (uint32_t e = 0; e < E_PER_NODE; ++e) {
            int64_t pos = static_cast<int64_t>(u) * E_PER_NODE + e;
            int expected = (u + e + 1) % V;
            EXPECT_EQ(h_col[pos], expected)
                << "node " << u << " edge " << e;
        }
    }

    // All weights should be -log(1.5) per the kernel.
    const double expected_w = -std::log(1.5);
    for (size_t i = 0; i < h_w.size(); ++i) {
        EXPECT_NEAR(h_w[i], expected_w, 1e-12) << "edge " << i;
    }
}

TEST(GraphGeneratorTest, RowOffsetsUnchanged) {
    constexpr int V = 4;
    constexpr uint32_t E_PER_NODE = 2;
    std::vector<int64_t> row_offsets(V + 1);
    for (int i = 0; i <= V; ++i) {
        row_offsets[i] = static_cast<int64_t>(i) * E_PER_NODE;
    }

    DeviceGraph dg = DeviceGraph::create_device_only(row_offsets, V, V * E_PER_NODE);
    launch_graph_generator(dg, E_PER_NODE);

    std::vector<int64_t> h_ro(V + 1);
    CUDA_CHECK(cudaMemcpy(h_ro.data(), dg.d_row_offsets,
                          h_ro.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i <= V; ++i) {
        EXPECT_EQ(h_ro[i], row_offsets[i]);
    }
}

TEST(GraphGeneratorTest, AllTargetsInValidRange) {
    // Even with random edges_per_node, no col_indices should fall outside [0, V).
    constexpr int V = 7;
    constexpr uint32_t E_PER_NODE = 5;

    std::vector<int64_t> row_offsets(V + 1);
    for (int i = 0; i <= V; ++i) {
        row_offsets[i] = static_cast<int64_t>(i) * E_PER_NODE;
    }

    DeviceGraph dg = DeviceGraph::create_device_only(row_offsets, V, V * E_PER_NODE);
    launch_graph_generator(dg, E_PER_NODE);

    std::vector<int> h_col(V * E_PER_NODE);
    CUDA_CHECK(cudaMemcpy(h_col.data(), dg.d_col_indices,
                          h_col.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int t : h_col) {
        EXPECT_GE(t, 0);
        EXPECT_LT(t, V);
    }
}

TEST(GraphGeneratorTest, SingleNodeEdgePerNodeProducesSelfLoop) {
    // V=3, E_PER_NODE=3 → every node has a self-loop (when e = V - (u+1)).
    constexpr int V = 3;
    constexpr uint32_t E_PER_NODE = 3;

    std::vector<int64_t> row_offsets(V + 1);
    for (int i = 0; i <= V; ++i) {
        row_offsets[i] = static_cast<int64_t>(i) * E_PER_NODE;
    }

    DeviceGraph dg = DeviceGraph::create_device_only(row_offsets, V, V * E_PER_NODE);
    launch_graph_generator(dg, E_PER_NODE);

    std::vector<int> h_col(V * E_PER_NODE);
    CUDA_CHECK(cudaMemcpy(h_col.data(), dg.d_col_indices,
                          h_col.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // Node u's edge e goes to (u + e + 1) % V. For u=2, e=1: (2+1+1)%3 = 1.
    // For self-loop: (u + e + 1) % V == u → e+1 ≡ 0 (mod V) → e = V-1.
    // So node u's edge at offset (V-1) is a self-loop.
    int self_loops = 0;
    for (int u = 0; u < V; ++u) {
        for (uint32_t e = 0; e < E_PER_NODE; ++e) {
            int target = h_col[u * E_PER_NODE + e];
            if (target == u) ++self_loops;
        }
    }
    EXPECT_EQ(self_loops, V);  // each of V nodes has exactly one self-loop
}
