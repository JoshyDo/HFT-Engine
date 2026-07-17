// =============================================================================
// vram_updater_test.cpp - VRAM Batch-Updates Validation
// =============================================================================
//
// Validation matrix:
//   1. step_once() liest genau 1 Batch (256) bei vollem SPSC
//   2. step_once() liest < batch_size wenn SPSC nicht voll
//   3. Ring-buffer wrap-around (head > capacity, modulo arithmetic)
//   4. cudaMemGetInfo zeigt korrekte VRAM-Allokation
//   5. End-to-End: SPSC fuellen, run_for(1s), Counter pruefen
//
// Excluded bounds:
//   - VRAM-Ring explicit synchronization with GPU dispatches
//   - Latency-Messungen (gehoert in micro-benchmarks)
// =============================================================================

// Compiler diagnostic: gtest-port.h requires ctype declarations without
// explicit inclusion. Pushing cctype BEFORE gtest header.
#include <cctype>
#include "vram_updater.hpp"
#include "spsc_ringbuffer.hpp"
#include "market_tick.hpp"

#include <cctype>
#include <gtest/gtest.h>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>

namespace {

// Hilfsfunktion: cudaMemGetInfo in Bytes (free, total).
struct GPUMemInfo {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
};
GPUMemInfo get_gpu_mem() {
    GPUMemInfo m;
    cudaMemGetInfo(&m.free_bytes, &m.total_bytes);
    return m;
}

TEST(VRAMUpdater, BasicRoundTrip) {
    auto rb = std::make_shared<SPSCRingbuffer<MarketTick, 65536>>();

    // 100 Ticks in SPSC fuellen.
    for (int i = 0; i < 100; ++i) {
        MarketTick t{};
        t.node_id = static_cast<std::uint32_t>(i);
        t.bid     = 100.0 + static_cast<double>(i);
        t.ask     = 100.5 + static_cast<double>(i);
        t.weight  = 1.0;
        ASSERT_TRUE(rb->try_push(t));
    }

    phase7::VRAMUpdater updater(rb);
    const std::size_t pushed = updater.step_once();
    EXPECT_EQ(pushed, 100u);
    EXPECT_EQ(updater.ticks_pushed_to_vram(), 100u);
    EXPECT_EQ(updater.batches_pushed(), 1u);
}

TEST(VRAMUpdater, EmptySPCReturnsZero) {
    auto rb = std::make_shared<SPSCRingbuffer<MarketTick, 65536>>();
    phase7::VRAMUpdater updater(rb);
    const std::size_t pushed = updater.step_once();
    EXPECT_EQ(pushed, 0u);
}

TEST(VRAMUpdater, BatchBoundary) {
    // 1024 Ticks pushen, dann step_once sollte batch_size=256 lesen.
    auto rb = std::make_shared<SPSCRingbuffer<MarketTick, 65536>>();
    phase7::VRAMRingConfig cfg;
    cfg.batch_size = 256;
    for (int i = 0; i < 1024; ++i) {
        MarketTick t{};
        t.node_id = static_cast<std::uint32_t>(i);
        t.bid     = 1.0;
        t.ask     = 2.0;
        t.weight  = 1.0;
        ASSERT_TRUE(rb->try_push(t));
    }
    phase7::VRAMUpdater updater(rb, cfg);
    EXPECT_EQ(updater.step_once(), 256u);
    EXPECT_EQ(updater.step_once(), 256u);
    EXPECT_EQ(updater.step_once(), 256u);
    EXPECT_EQ(updater.step_once(), 256u);
    EXPECT_EQ(updater.step_once(), 0u);  // SPSC leer
    EXPECT_EQ(updater.ticks_pushed_to_vram(), 1024u);
    EXPECT_EQ(updater.batches_pushed(), 4u);
}

TEST(VRAMUpdater, VRAMAllocated) {
    auto rb = std::make_shared<SPSCRingbuffer<MarketTick, 65536>>();

    const auto before = get_gpu_mem();

    // Default-Config: 1M Ticks = 32 MB VRAM.
    {
        phase7::VRAMUpdater updater(rb);

        // Ingest 100 consecutive ticks and dispatch.
        for (int i = 0; i < 100; ++i) {
            MarketTick t{};
            t.bid = 1.0; t.ask = 2.0; t.weight = 1.0;
            ASSERT_TRUE(rb->try_push(t));
        }
        EXPECT_EQ(updater.step_once(), 100u);

        const auto during = get_gpu_mem();
        // VRAM-Verbrauch sollte >= 32 MB sein (32M Ticks * 32 Byte).
        const std::size_t used_delta = before.free_bytes - during.free_bytes;
        std::printf("[VRAM] used delta = %zu bytes (~ %.1f MB)\n",
                    used_delta, used_delta / 1024.0 / 1024.0);
        EXPECT_GE(used_delta, 32u * 1024u * 1024u)
            << "VRAM-Allocation zu klein: " << used_delta << " bytes";
    }
    // Post-destructor bounds: VRAM heap guarantees explicit zero-allocation release.
    const auto after = get_gpu_mem();
    // Memory footprint must revert to pre-initialization bounds (ignoring OS overhead)
    // cuda-Caching-Effekte, daher Toleranz).
    const std::size_t leaked = before.free_bytes - after.free_bytes;
    std::printf("[VRAM] leaked after dtor = %zu bytes\n", leaked);
    EXPECT_LT(leaked, 1u * 1024u * 1024u)
        << "VRAM-Leak nach Destruktor: " << leaked << " bytes";
}

TEST(VRAMUpdater, HeadWrapsAround) {
    // Minimal-capacity VRAM-Ring (4) evaluating cyclic boundary overruns.
    auto rb = std::make_shared<SPSCRingbuffer<MarketTick, 65536>>();
    phase7::VRAMRingConfig cfg;
    cfg.vram_capacity_ticks = 4;
    cfg.batch_size = 1;

    phase7::VRAMUpdater updater(rb, cfg);

    // 12 sequential ingestions + step_once. Head pointer migrates 0 -> 12,
    // VRAM-Ring wraps nach 4.
    for (int i = 0; i < 12; ++i) {
        MarketTick t{};
        t.node_id = static_cast<std::uint32_t>(i);
        t.bid     = 1.0; t.ask = 2.0; t.weight = 1.0;
        ASSERT_TRUE(rb->try_push(t));
        const std::size_t pushed = updater.step_once();
        EXPECT_EQ(pushed, 1u) << "iter " << i;
    }
    EXPECT_EQ(updater.ticks_pushed_to_vram(), 12u);
    EXPECT_EQ(updater.device_ring_head(), 12u);
    // 12 % 4 = 0 -> cyclic pointer returns to 0, validating
    // head-Tracker zaehlt linear (12), nicht modulo capacity.
}

TEST(VRAMUpdater, RunForConsumesTicks) {
    auto rb = std::make_shared<SPSCRingbuffer<MarketTick, 65536>>();

    // 10.000 Ticks im Voraus fuellen.
    for (int i = 0; i < 10'000; ++i) {
        MarketTick t{};
        t.node_id = static_cast<std::uint32_t>(i);
        t.bid     = 100.0 + i * 0.01;
        t.ask     = 100.5 + i * 0.01;
        t.weight  = 1.0;
        ASSERT_TRUE(rb->try_push(t));
    }

    phase7::VRAMUpdater updater(rb);
    updater.run();

    // 200ms spin-wait lock - consumer asynchronously coalesces 10k ticks.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    updater.stop();
    // Worker-Thread joined automatisch im Destruktor; explizit stoppen.

    std::printf("[RunFor] ticks_pushed=%llu, batches=%llu\n",
                static_cast<unsigned long long>(updater.ticks_pushed_to_vram()),
                static_cast<unsigned long long>(updater.batches_pushed()));

    EXPECT_GT(updater.ticks_pushed_to_vram(), 0u);
}

}  // namespace
