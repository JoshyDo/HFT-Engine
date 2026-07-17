// live_arbitrage_test.cpp - Unit Tests for tick_to_edge_update_kernel
//
// Tick -> CSR edge_weight Update pipeline dispatch.
// Schema A: node_id = Edge-Index. -log(bid) wird direkt geschrieben.
//
// Wir testen vier Faelle:
//   1. BasicUpdate: 10 Ticks mit bid=2.0 → alle 10 weights ≈ -0.693
//   2. ZeroBidSkipped: bid=0.0 → weight bleibt unveraendert (init = 42.0)
//   3. NegativeBidSkipped: bid=-1.0 → weight bleibt unveraendert
//   4. OutOfRangeNodeIdSkipped: node_id >= num_edges → skip
//
// Erwarteter Wert: -log(2.0) ≈ -0.6931471805599453
// Wir verwenden EXPECT_NEAR mit tolerance = 1e-9.
// =============================================================================

#include "live_arbitrage.hpp"

#include <gtest/gtest.h>

#include "market_tick.hpp"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int   kNumEdges        = 16;
constexpr float kSentinel        = 42.0f;                 // unmodified sentinel value for skip validation
constexpr float kExpectedLogInv2 = -0.6931471805599453f;  // -log(2.0)

void fill_sentinel(float* h_w, int n) {
    for (int i = 0; i < n; ++i) h_w[i] = kSentinel;
}

TEST(TickToEdgeUpdate, BasicUpdate) {
    // Initialize host memory structures.
    std::vector<float> h_w(kNumEdges);
    fill_sentinel(h_w.data(), kNumEdges);

    std::vector<int64_t> h_orig_to_csc(kNumEdges);
    for (int i = 0; i < kNumEdges; ++i) h_orig_to_csc[i] = i;

    std::vector<MarketTick> h_ticks(10);
    for (int i = 0; i < 10; ++i) {
        h_ticks[i].node_id = static_cast<std::uint32_t>(i);  // 0..9
        h_ticks[i].bid     = 2.0;
        h_ticks[i].ask     = 2.1;
        h_ticks[i].weight  = 1.0;
    }

    // Initialize device memory structures.
    float*      d_w           = nullptr;
    int64_t*    d_orig_to_csc = nullptr;
    MarketTick* d_ticks       = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, kNumEdges * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_orig_to_csc, kNumEdges * sizeof(int64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ticks, h_ticks.size() * sizeof(MarketTick)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_w, h_w.data(), kNumEdges * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_orig_to_csc, h_orig_to_csc.data(), kNumEdges * sizeof(int64_t), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_ticks, h_ticks.data(), h_ticks.size() * sizeof(MarketTick), cudaMemcpyHostToDevice),
              cudaSuccess);

    // Dispatch kernel.
    auto err =
        phase7::launch_tick_to_edge_update(d_ticks, static_cast<int>(h_ticks.size()), d_w, d_orig_to_csc, kNumEdges);
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Retrieve memory from device.
    ASSERT_EQ(cudaMemcpy(h_w.data(), d_w, kNumEdges * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    // indices 0..9: -log(2.0) ≈ -0.693
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(h_w[i], kExpectedLogInv2, 1e-9) << "edge[" << i << "] not updated correctly";
    }
    // indices 10..15: untouched by kernel dispatch.
    for (int i = 10; i < kNumEdges; ++i) {
        EXPECT_EQ(h_w[i], kSentinel) << "edge[" << i << "] should be untouched";
    }

    cudaFree(d_w);
    cudaFree(d_ticks);
    cudaFree(d_orig_to_csc);
}

TEST(TickToEdgeUpdate, ZeroBidSkipped) {
    std::vector<float> h_w(kNumEdges);
    fill_sentinel(h_w.data(), kNumEdges);

    std::vector<int64_t> h_orig_to_csc(kNumEdges);
    for (int i = 0; i < kNumEdges; ++i) h_orig_to_csc[i] = i;

    std::vector<MarketTick> h_ticks(1);
    h_ticks[0].node_id = 5u;
    h_ticks[0].bid     = 0.0;  // -log(0) = +inf -> discard
    h_ticks[0].ask     = 1.0;
    h_ticks[0].weight  = 1.0;

    float*      d_w           = nullptr;
    int64_t*    d_orig_to_csc = nullptr;
    MarketTick* d_ticks       = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, kNumEdges * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_orig_to_csc, kNumEdges * sizeof(int64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ticks, sizeof(MarketTick)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, h_w.data(), kNumEdges * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_orig_to_csc, h_orig_to_csc.data(), kNumEdges * sizeof(int64_t), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_ticks, h_ticks.data(), sizeof(MarketTick), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(phase7::launch_tick_to_edge_update(d_ticks, 1, d_w, d_orig_to_csc, kNumEdges), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_w.data(), d_w, kNumEdges * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < kNumEdges; ++i) {
        EXPECT_EQ(h_w[i], kSentinel) << "edge[" << i << "] mutated despite bid=0";
    }

    cudaFree(d_w);
    cudaFree(d_ticks);
    cudaFree(d_orig_to_csc);
}

TEST(TickToEdgeUpdate, NegativeBidSkipped) {
    std::vector<float> h_w(kNumEdges);
    fill_sentinel(h_w.data(), kNumEdges);

    std::vector<int64_t> h_orig_to_csc(kNumEdges);
    for (int i = 0; i < kNumEdges; ++i) h_orig_to_csc[i] = i;

    std::vector<MarketTick> h_ticks(1);
    h_ticks[0].node_id = 5u;
    h_ticks[0].bid     = -1.0;  // -log(neg) = NaN -> discard
    h_ticks[0].ask     = 1.0;
    h_ticks[0].weight  = 1.0;

    float*      d_w           = nullptr;
    int64_t*    d_orig_to_csc = nullptr;
    MarketTick* d_ticks       = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, kNumEdges * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_orig_to_csc, kNumEdges * sizeof(int64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ticks, sizeof(MarketTick)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, h_w.data(), kNumEdges * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_orig_to_csc, h_orig_to_csc.data(), kNumEdges * sizeof(int64_t), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_ticks, h_ticks.data(), sizeof(MarketTick), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(phase7::launch_tick_to_edge_update(d_ticks, 1, d_w, d_orig_to_csc, kNumEdges), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_w.data(), d_w, kNumEdges * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < kNumEdges; ++i) {
        EXPECT_EQ(h_w[i], kSentinel) << "edge[" << i << "] mutated despite bid<0";
    }

    cudaFree(d_w);
    cudaFree(d_ticks);
    cudaFree(d_orig_to_csc);
}

TEST(TickToEdgeUpdate, OutOfRangeNodeIdSkipped) {
    std::vector<float> h_w(kNumEdges);
    fill_sentinel(h_w.data(), kNumEdges);

    std::vector<int64_t> h_orig_to_csc(kNumEdges);
    for (int i = 0; i < kNumEdges; ++i) h_orig_to_csc[i] = i;

    // Three ticks: 2 valid, 1 out-of-bounds.
    std::vector<MarketTick> h_ticks(3);
    h_ticks[0].node_id = 0u;  // OK
    h_ticks[0].bid     = 2.0;
    h_ticks[1].node_id = 99999u;  // out of range
    h_ticks[1].bid     = 2.0;
    h_ticks[2].node_id = 1u;  // OK
    h_ticks[2].bid     = 2.0;
    for (int i = 0; i < 3; ++i) {
        h_ticks[i].ask    = 2.1;
        h_ticks[i].weight = 1.0;
    }

    float*      d_w           = nullptr;
    int64_t*    d_orig_to_csc = nullptr;
    MarketTick* d_ticks       = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, kNumEdges * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_orig_to_csc, kNumEdges * sizeof(int64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ticks, 3 * sizeof(MarketTick)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, h_w.data(), kNumEdges * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_orig_to_csc, h_orig_to_csc.data(), kNumEdges * sizeof(int64_t), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_ticks, h_ticks.data(), 3 * sizeof(MarketTick), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(phase7::launch_tick_to_edge_update(d_ticks, 3, d_w, d_orig_to_csc, kNumEdges), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_w.data(), d_w, kNumEdges * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    // Tick 0 → edge[0] = -log(2.0)
    EXPECT_NEAR(h_w[0], kExpectedLogInv2, 1e-9);
    // Tick 2 → edge[1] = -log(2.0)
    EXPECT_NEAR(h_w[1], kExpectedLogInv2, 1e-9);
    // Remaining elements are untouched.
    for (int i = 2; i < kNumEdges; ++i) {
        EXPECT_EQ(h_w[i], kSentinel) << "edge[" << i << "] should be untouched";
    }

    cudaFree(d_w);
    cudaFree(d_ticks);
    cudaFree(d_orig_to_csc);
}

TEST(TickToEdgeUpdate, EmptyBatchIsNoop) {
    std::vector<float> h_w(kNumEdges);
    fill_sentinel(h_w.data(), kNumEdges);

    std::vector<int64_t> h_orig_to_csc(kNumEdges);
    for (int i = 0; i < kNumEdges; ++i) h_orig_to_csc[i] = i;

    float*      d_w           = nullptr;
    int64_t*    d_orig_to_csc = nullptr;
    MarketTick* d_ticks       = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, kNumEdges * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_orig_to_csc, kNumEdges * sizeof(int64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ticks, sizeof(MarketTick)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, h_w.data(), kNumEdges * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_orig_to_csc, h_orig_to_csc.data(), kNumEdges * sizeof(int64_t), cudaMemcpyHostToDevice),
              cudaSuccess);

    // batch_size = 0 -> no-op, bypass kernel dispatch.
    ASSERT_EQ(phase7::launch_tick_to_edge_update(d_ticks, 0, d_w, d_orig_to_csc, kNumEdges), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_w.data(), d_w, kNumEdges * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < kNumEdges; ++i) {
        EXPECT_EQ(h_w[i], kSentinel);
    }

    cudaFree(d_w);
    cudaFree(d_orig_to_csc);
    cudaFree(d_ticks);
}

}  // namespace
