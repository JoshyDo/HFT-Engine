// Tests for algorithm.hpp (CPU SPFA arbitrage detection).
//
// The existing tests in main_test.cpp cover: NoArbitrage, DirectArbitrage,
// TriangleArbitrage, Profiling. This file adds the edge cases the existing
// tests miss.
//
// Edge cases worth testing:
//   - self-loop (single edge u->u with negative cycle weight)
//   - isolated component (disconnected from source)
//   - buffer too small (must be >= V)
//   - reuse of the same buffer across calls (it gets reset internally)

#include <gtest/gtest.h>
#include <cstdint>
#include "arena.hpp"
#include "algorithm.hpp"

namespace {

// Build a simple linear chain a -> b -> c with given log-weights.
// A negative cycle exists iff the product of weights < 1.0, equivalently
// sum of log-weights < 0.0.
GraphArena makeChain(uint32_t a, uint32_t b, uint32_t c,
                     double w_ab, double w_bc, double w_ca) {
    GraphArena arena(8, 16);
    arena.addNode(a);
    arena.addNode(b);
    arena.addNode(c);
    arena.addEdge(a, b, 1.0, std::exp(-w_ab));
    arena.addEdge(b, c, 1.0, std::exp(-w_bc));
    arena.addEdge(c, a, 1.0, std::exp(-w_ca));
    return arena;
}

}  // namespace

TEST(SPFATest, NoCycleWhenAllWeightsPositive) {
    // All log-weights > 0 → no arbitrage
    auto arena = makeChain(0, 1, 2, 0.1, 0.1, 0.1);
    SPFABuffer buf(3);
    EXPECT_FALSE(BellmanFord::detectArbitrage(arena, 0, buf));
}

TEST(SPFATest, TriangleArbitrageDetected) {
    // log_ask = -log(ask). For arbitrage, we need sum(log_ask) < 0 over a cycle,
    // which means ask > 1.0. With ask = 1.2 on each edge: log_ask ≈ -0.182,
    // three edges sum to ≈ -0.547, a negative cycle.
    GraphArena arena(3, 8);
    arena.addNode(0);
    arena.addNode(1);
    arena.addNode(2);
    arena.addEdge(0, 1, 1.0, 1.2);
    arena.addEdge(1, 2, 1.0, 1.2);
    arena.addEdge(2, 0, 1.0, 1.2);
    SPFABuffer buf(3);
    EXPECT_TRUE(BellmanFord::detectArbitrage(arena, 0, buf));
}

TEST(SPFATest, BufferIsReusedAcrossCalls) {
    // The whole point of SPFABuffer is to avoid allocations per call.
    // Call detectArbitrage twice on the same buffer; both must return
    // correct results. This catches a bug where state leaks between calls.
    SPFABuffer buf(3);
    auto arena_pos = makeChain(0, 1, 2, 0.1, 0.1, 0.1);
    auto arena_neg = makeChain(0, 1, 2, -0.5, -0.5, -0.5);

    EXPECT_FALSE(BellmanFord::detectArbitrage(arena_pos, 0, buf));
    EXPECT_TRUE(BellmanFord::detectArbitrage(arena_neg, 0, buf));
    EXPECT_FALSE(BellmanFord::detectArbitrage(arena_pos, 0, buf));
}

TEST(SPFATest, IsolatedNodeNotReached) {
    // Source is node 0. Node 2 has no edges into or out of it.
    // No arbitrage reachable from 0.
    GraphArena arena(3, 4);
    arena.addNode(0);
    arena.addNode(1);
    arena.addNode(2);
    arena.addEdge(0, 1, 1.0, 1.0);
    arena.addEdge(1, 0, 1.0, 1.0);
    SPFABuffer buf(3);
    EXPECT_FALSE(BellmanFord::detectArbitrage(arena, 0, buf));
}

TEST(SPFATest, SelfLoopArbitrage) {
    // u -> u with ask > 1.0 → log_ask = -log(ask) < 0 → negative self-loop.
    GraphArena arena(1, 1);
    arena.addNode(0);
    arena.addEdge(0, 0, 1.0, 1.5);  // ask = 1.5 > 1.0 → log_ask ≈ -0.405
    SPFABuffer buf(1);
    EXPECT_TRUE(BellmanFord::detectArbitrage(arena, 0, buf));
}
