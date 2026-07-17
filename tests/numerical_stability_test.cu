// Numerical stability tests for the Bellman-Ford engine.
//
// Edge cases that IEEE 754 introduces:
//   - DBL_MAX arithmetic (saturates to DBL_MAX)
//   - Negative zero
//   - Infinity
//   - Loss of precision near the limits
//   - Self-loops with extreme weights
//
// These tests don't check exact values (those are not stable), they check
// that the engine doesn't CRASH, INFINITE-LOOP, or RETURN NaN.

#include <gtest/gtest.h>
#include <cfloat>
#include <vector>
#include <cstdint>
#include <tuple>
#include <cmath>
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
// IEEE 754 edge cases
// =============================================================================

TEST(NumericalStability, DblMaxArithmeticDoesNotProduceNaN) {
    // 0 -> 1 with weight DBL_MAX. dist[0] = 0, new_dist = 0 + DBL_MAX = DBL_MAX.
    // Then 1 -> 0 with weight -DBL_MAX. dist[1] = DBL_MAX, new_dist = DBL_MAX - DBL_MAX = NaN.
    //
    // If we don't catch this, the kernel writes NaN into dist[0], which then
    // poisons every subsequent iteration. We test that dist[0] is NOT NaN.
    auto dg = make_small_graph(2, {
        {0, 1, DBL_MAX},
        {1, 0, -DBL_MAX},
    });
    bool converged = launch_bellman_ford_full_v2(dg, 0, 3);
    // We don't care if it converges; we care that dist[0] is not NaN.
    auto h_dist = copy_to_host(dg.d_dist, dg.num_vertices);
    EXPECT_FALSE(std::isnan(h_dist[0]));
    EXPECT_FALSE(std::isnan(h_dist[1]));
}

TEST(NumericalStability, SelfLoopWithVeryNegativeWeightConvergesCleanly) {
    // Self-loop with weight -1e10. After 1 iteration, dist[0] = -1e10.
    // After 2 iterations, dist[0] = -2e10. ... After N iterations, dist[0] = -N * 1e10.
    // Eventually it should saturate at -DBL_MAX.
    auto dg = make_small_graph(1, {{0, 0, -1e10}});
    bool converged = launch_bellman_ford_full_v2(dg, 0, 100);
    EXPECT_FALSE(converged);  // self-loop with negative weight = always converging further
    auto h_dist = copy_to_host(dg.d_dist, 1);
    EXPECT_FALSE(std::isnan(h_dist[0]));
    EXPECT_LT(h_dist[0], 0.0);  // it went negative
    EXPECT_GT(h_dist[0], -DBL_MAX);  // but didn't underflow to -inf yet
}

TEST(NumericalStability, TriangleWithVeryNegativeCycleSaturates) {
    // 3-cycle with weights summing to -1e15. After enough iterations,
    // dist should saturate at -DBL_MAX, not crash.
    auto dg = make_small_graph(3, {
        {0, 1, -1e5},
        {1, 2, -1e5},
        {2, 0, -1e5},
    });
    bool converged = launch_bellman_ford_full_v2(dg, 0, 50);
    EXPECT_FALSE(converged);
    auto h_dist = copy_to_host(dg.d_dist, dg.num_vertices);
    for (double d : h_dist) {
        EXPECT_FALSE(std::isnan(d));
        EXPECT_FALSE(std::isinf(d));
    }
}

TEST(NumericalStability, DisconnectedComponentStaysAtDblMax) {
    // 3 nodes, only one edge. Dist[2] is uninitialized-ish: should be DBL_MAX
    // because it's unreachable.
    auto dg = make_small_graph(3, {{0, 1, 1.0}});
    bool converged = launch_bellman_ford_full_v2(dg, 0, 5);
    EXPECT_TRUE(converged);
    auto h_dist = copy_to_host(dg.d_dist, dg.num_vertices);
    EXPECT_DOUBLE_EQ(h_dist[2], DBL_MAX);
}

TEST(NumericalStability, SmallPositiveWeightsDoNotCauseConvergenceIssues) {
    // Very small positive weights (1e-300) — should still converge.
    auto dg = make_small_graph(2, {{0, 1, 1e-300}});
    bool converged = launch_bellman_ford_full_v2(dg, 0, 3);
    EXPECT_TRUE(converged);
    auto h_dist = copy_to_host(dg.d_dist, 2);
    EXPECT_NEAR(h_dist[1], 1e-300, 1e-310);
}

TEST(NumericalStability, MixedPositiveAndNegativeWeights) {
    // 0 -> 1 with +100, 1 -> 2 with -50. Total 50. Should converge.
    auto dg = make_small_graph(3, {
        {0, 1,  100.0},
        {1, 2, -50.0},
    });
    bool converged = launch_bellman_ford_full_v2(dg, 0, 5);
    EXPECT_TRUE(converged);
    auto h_dist = copy_to_host(dg.d_dist, 3);
    EXPECT_DOUBLE_EQ(h_dist[2], 50.0);
}

TEST(NumericalStability, ZeroWeightEdgesWork) {
    // 0 -> 1 with weight 0. dist[1] = 0 after one iter.
    auto dg = make_small_graph(2, {{0, 1, 0.0}});
    bool converged = launch_bellman_ford_full_v2(dg, 0, 3);
    EXPECT_TRUE(converged);
    auto h_dist = copy_to_host(dg.d_dist, 2);
    EXPECT_DOUBLE_EQ(h_dist[1], 0.0);
}
