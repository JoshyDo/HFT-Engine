// Tests for per-block modified flags synchronization.
//
// Goal: prove the V2 host loop produces the same result as V1, with the
// same convergence behavior. We do NOT benchmark here — that's Step 2
// (Nsight Compute). These tests are correctness-only.

#include <gtest/gtest.h>
#include <cfloat>
#include <vector>
#include <cstdint>
#include <tuple>
#include <algorithm>
#include <cuda_runtime.h>

#include "device_graph.hpp"
#include "bellman_ford.hpp"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        EXPECT_EQ(err, cudaSuccess)                                           \
            << "CUDA error: " << cudaGetErrorString(err)                      \
            << " at " << __FILE__ << ":" << __LINE__;                        \
    } while (0)

namespace {

DeviceGraph make_small_graph(int vertices,
                             const std::vector<std::tuple<int,int,double>>& edges) {
    std::vector<int> deg(vertices, 0);
    for (auto& [u, v, w] : edges) {
        EXPECT_LT(u, vertices);
        EXPECT_LT(v, vertices);
        deg[u]++;
    }
    std::vector<int64_t> row_offsets(vertices + 1, 0);
    for (int i = 0; i < vertices; ++i) {
        row_offsets[i + 1] = row_offsets[i] + deg[i];
    }
    std::vector<int> col_indices(edges.size());
    std::vector<double> edge_weights(edges.size());
    std::vector<int> cursor(vertices, 0);
    for (auto& [u, v, w] : edges) {
        int64_t pos = row_offsets[u] + cursor[u]++;
        col_indices[pos] = v;
        edge_weights[pos] = w;
    }
    CSRGraph host_graph;
    host_graph.num_vertices = vertices;
    host_graph.num_edges = static_cast<int64_t>(edges.size());
    host_graph.row_offsets = row_offsets;
    host_graph.col_indices = col_indices;
    host_graph.edge_weights = edge_weights;
    return DeviceGraph(host_graph);
}

template <typename T>
std::vector<T> copy_to_host(const T* d_ptr, size_t count) {
    std::vector<T> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), d_ptr, count * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return host;
}

}  // namespace

// =============================================================================
// Correctness equivalence with V1
// =============================================================================

TEST(BFV2Test, ConvergesOnLinearChain) {
    auto dg = make_small_graph(4, {
        {0, 1, 1.0},
        {1, 2, 2.0},
        {2, 3, 3.0},
    });
    // V iters: V-1 relax waves fill distances, the V-th confirms no more
    // changes (modified stays 0). Matches V1's contract in
    // BellmanFordFullTest.ConvergesOnLinearChain.
    bool converged = launch_bellman_ford_full_v2(dg, 0, 4);
    EXPECT_TRUE(converged);
    auto h_dist = copy_to_host(dg.d_dist, dg.num_vertices);
    EXPECT_DOUBLE_EQ(h_dist[0], 0.0);
    EXPECT_DOUBLE_EQ(h_dist[1], 1.0);
    EXPECT_DOUBLE_EQ(h_dist[2], 3.0);
    EXPECT_DOUBLE_EQ(h_dist[3], 6.0);
}

TEST(BFV2Test, NoConvergenceOnNegativeCycle) {
    auto dg = make_small_graph(3, {
        {0, 1, -1.0},
        {1, 2, -1.0},
        {2, 0, -1.0},
    });
    bool converged = launch_bellman_ford_full_v2(dg, 0, 2);
    EXPECT_FALSE(converged);
}

TEST(BFV2Test, SelfLoopArbitrage) {
    auto dg = make_small_graph(1, {{0, 0, -1.0}});
    bool converged = launch_bellman_ford_full_v2(dg, 0, 3);
    EXPECT_FALSE(converged);
}

TEST(BFV2Test, EarlyExitOnConvergedGraph) {
    // Single edge, no further work after iter 1.
    auto dg = make_small_graph(2, {{0, 1, 1.0}});
    bool converged = launch_bellman_ford_full_v2(dg, 0, 10);
    EXPECT_TRUE(converged);
    auto h_dist = copy_to_host(dg.d_dist, dg.num_vertices);
    EXPECT_DOUBLE_EQ(h_dist[1], 1.0);
}

TEST(BFV2Test, UnreachableNodesRemainInfinity) {
    auto dg = make_small_graph(3, {{0, 1, 1.0}});
    bool converged = launch_bellman_ford_full_v2(dg, 0, 5);
    EXPECT_TRUE(converged);
    auto h_dist = copy_to_host(dg.d_dist, dg.num_vertices);
    EXPECT_DOUBLE_EQ(h_dist[2], DBL_MAX);
}


