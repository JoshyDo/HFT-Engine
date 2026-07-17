// Cross-validation tests: SPFA (CPU) vs Bellman-Ford (GPU).
//
// Goal: prove that both pipelines agree on whether a graph contains arbitrage.
// If both return the same boolean for many inputs, we have high confidence
// the GPU engine is correct.
//
// We build a GraphArena on the host, convert it to a CSRGraph, upload to
// DeviceGraph, then run both arbitrage detectors and compare results.
//
// This is the most powerful kind of test we can write: independent
// implementations of the same algorithm must agree.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <tuple>
#include <cmath>
#include <cuda_runtime.h>

#include "arena.hpp"
#include "algorithm.hpp"
#include "csr_graph.hpp"
#include "device_graph.hpp"
#include "bellman_ford.hpp"
#include "arbitrage_optimized.hpp"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        EXPECT_EQ(err, cudaSuccess)                                           \
            << "CUDA error: " << cudaGetErrorString(err)                      \
            << " at " << __FILE__ << ":" << __LINE__;                        \
    } while (0)

namespace {

// Build a small CPU graph and its GPU counterpart from a list of edges.
// edges: vector of (u, v, bid, ask) tuples.
struct CrossGraph {
    GraphArena cpu;
    DeviceGraph gpu;
    int V;
    int64_t E;

    CrossGraph(int vertices, const std::vector<std::tuple<int,int,double,double>>& edges)
        : V(vertices), E(static_cast<int64_t>(edges.size())),
          cpu(vertices, vertices * vertices)
    {
        // 1. CPU side: GraphArena with bid/ask prices.
        for (int i = 0; i < vertices; ++i) cpu.addNode(i);
        for (auto& [u, v, bid, ask] : edges) {
            cpu.addEdge(u, v, bid, ask);  // log_ask = -log(ask) computed internally
        }

        // 2. Build CSR row_offsets on the host.
        std::vector<int> deg(vertices, 0);
        for (auto& [u, v, bid, ask] : edges) ++deg[u];
        std::vector<int64_t> row_offsets(vertices + 1, 0);
        for (int i = 0; i < vertices; ++i) {
            row_offsets[i + 1] = row_offsets[i] + deg[i];
        }

        // 3. Fill col_indices and edge_weights in CSR order.
        //    Weight: log_ask = -log(ask). Same formula the GPU kernel uses.
        std::vector<int> col_indices(edges.size());
        std::vector<double> edge_weights(edges.size());
        std::vector<int> cursor(vertices, 0);
        for (auto& [u, v, bid, ask] : edges) {
            int64_t pos = row_offsets[u] + cursor[u]++;
            col_indices[pos] = v;
            edge_weights[pos] = -std::log(ask);
        }

        // 4. Build CSRGraph and copy to DeviceGraph.
        CSRGraph csr;
        csr.num_vertices = vertices;
        csr.num_edges = static_cast<int64_t>(edges.size());
        csr.row_offsets = row_offsets;
        csr.col_indices = col_indices;
        csr.edge_weights = edge_weights;
        gpu = DeviceGraph(csr);
    }
};

}  // namespace

// =============================================================================
// Agreement tests
// =============================================================================

TEST(CrossValidation, NoArbitrageOnLinearChain) {
    // 0 -> 1 -> 2, all positive (ask > 1.0), no cycle.
    CrossGraph g(3, {
        {0, 1, 1.0, 1.1},
        {1, 2, 1.0, 1.1},
    });

    SPFABuffer buf(g.V);
    bool cpu_result = BellmanFord::detectArbitrage(g.cpu, 0, buf);
    bool converged = launch_bellman_ford_full_v2(g.gpu, 0, g.gpu.num_vertices);
    auto gpu_edges = phase7::optimized::launch_arbitrage_detection_persistent(g.gpu, 0, &converged);

    EXPECT_FALSE(cpu_result);
    EXPECT_TRUE(gpu_edges.empty());
}

TEST(CrossValidation, TriangleArbitrageDetectedByBoth) {
    // 0 -> 1 -> 2 -> 0 with all ask = 1.2. log_ask = -log(1.2) ≈ -0.182.
    // Three edges sum to ≈ -0.547, a negative cycle, so arbitrage.
    CrossGraph g(3, {
        {0, 1, 1.0, 1.2},
        {1, 2, 1.0, 1.2},
        {2, 0, 1.0, 1.2},
    });

    SPFABuffer buf(g.V);
    bool cpu_result = BellmanFord::detectArbitrage(g.cpu, 0, buf);
    bool converged = launch_bellman_ford_full_v2(g.gpu, 0, g.gpu.num_vertices);
    auto gpu_edges = phase7::optimized::launch_arbitrage_detection_persistent(g.gpu, 0, &converged);

    EXPECT_TRUE(cpu_result);
    EXPECT_EQ(gpu_edges.size(), 3u);  // all 3 edges are part of the cycle
}

TEST(CrossValidation, SelfLoopArbitrage) {
    // 0 -> 0 with ask = 1.5 > 1. log_ask = -log(1.5) ≈ -0.405 (negative).
    CrossGraph g(1, {
        {0, 0, 1.0, 1.5},
    });

    SPFABuffer buf(g.V);
    bool cpu_result = BellmanFord::detectArbitrage(g.cpu, 0, buf);
    bool converged = launch_bellman_ford_full_v2(g.gpu, 0, g.gpu.num_vertices);
    auto gpu_edges = phase7::optimized::launch_arbitrage_detection_persistent(g.gpu, 0, &converged);

    EXPECT_TRUE(cpu_result);
    EXPECT_EQ(gpu_edges.size(), 1u);
}

TEST(CrossValidation, NoCycleOnDisconnectedComponents) {
    // Component A: 0 -> 1 (positive), Component B: 2 -> 3 (positive)
    // No cycle anywhere.
    CrossGraph g(4, {
        {0, 1, 1.0, 1.2},
        {2, 3, 1.0, 1.2},
    });

    SPFABuffer buf(g.V);
    bool cpu_result = BellmanFord::detectArbitrage(g.cpu, 0, buf);
    bool converged = launch_bellman_ford_full_v2(g.gpu, 0, g.gpu.num_vertices);
    auto gpu_edges = phase7::optimized::launch_arbitrage_detection_persistent(g.gpu, 0, &converged);

    EXPECT_FALSE(cpu_result);
    EXPECT_TRUE(gpu_edges.empty());
}

TEST(CrossValidation, CyclePlusExtraEdges) {
    // Triangle 0->1->2->0 (ask=1.2, arbitrage) plus an ISOLATED
    // component (3<->4) that the source cannot reach. The detector
    // must not flag edges that are unreachable from the source.
    // (The cascade only relaxes edges that the source can reach; any
    // edge starting from an unreachable vertex is skipped because
    // dist[unreachable] == DBL_MAX.)
    CrossGraph g(5, {
        // Cycle (reachable from source 0)
        {0, 1, 1.0, 1.2},
        {1, 2, 1.0, 1.2},
        {2, 0, 1.0, 1.2},
        // Isolated component (NOT reachable from source)
        {3, 4, 1.0, 1.1},
        {4, 3, 1.0, 1.1},
    });

    SPFABuffer buf(g.V);
    bool cpu_result = BellmanFord::detectArbitrage(g.cpu, 0, buf);
    bool converged = launch_bellman_ford_full_v2(g.gpu, 0, g.gpu.num_vertices);
    auto gpu_edges = phase7::optimized::launch_arbitrage_detection_persistent(g.gpu, 0, &converged);

    EXPECT_TRUE(cpu_result);
    // Only the 3 cycle edges should be flagged. The 2 isolated edges are NOT.
    EXPECT_EQ(gpu_edges.size(), 3u);
}

TEST(CrossValidation, IsolatedNodeWithNoEdges) {
    // 3 nodes, but only one edge. No cycle possible.
    CrossGraph g(3, {
        {0, 1, 1.0, 1.0},
    });

    SPFABuffer buf(g.V);
    bool cpu_result = BellmanFord::detectArbitrage(g.cpu, 0, buf);
    bool converged = launch_bellman_ford_full_v2(g.gpu, 0, g.gpu.num_vertices);
    auto gpu_edges = phase7::optimized::launch_arbitrage_detection_persistent(g.gpu, 0, &converged);

    EXPECT_FALSE(cpu_result);
    EXPECT_TRUE(gpu_edges.empty());
}
