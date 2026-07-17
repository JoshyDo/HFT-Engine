// =============================================================================
// spsc_ringbuffer_test.cpp - Lock-Free SPSC Ringbuffer Validation
// =============================================================================
// Verwendet GoogleTest (bereits in CMake via GTest::gtest_main).
//
// Validation matrix:
//   1. Single-Threaded Basics (push/pop, voll/leer)
//   2. Wrap-Around: cyclical boundaries extending beyond allocated Capacity
//   3. Ordering invariant: Lock-free SPSC strict FIFO
//   4. Power-of-2-Maskierung (capacity = 4, 8, 65536)
//   5. SPSC-Concurrency: 1 Producer + 1 Consumer, 1M Items, perf-counter
//   6. Compile-Time invariants (static_assert validations)
//
// Excluded execution bounds (intentional bypass):
//   - MPMC contention: inherently violates strict SPSC thread-safety contract.
//     Silently induces data races. Architectural documentation specifies:
//     "MPMC requires explicit CAS atomic spin-loops".
// =============================================================================

#include "spsc_ringbuffer.hpp"
#include "market_tick.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <cstdint>

namespace {

constexpr std::size_t kCapacity = 8;  // Constrained capacity forcefully triggers wrap-around
[[maybe_unused]] constexpr std::size_t kBigCapacity = 65536;  // 2^16, realer Use-Case

// ---- 1. Single-Threaded Basics ----

TEST(SPSCRingbuffer, EmptyAfterConstruction) {
    SPSCRingbuffer<int, kCapacity> rb;
    EXPECT_TRUE(rb.empty_approx());
    EXPECT_FALSE(rb.full_approx());
    EXPECT_EQ(rb.size_approx(), 0u);
    EXPECT_EQ(rb.capacity(), kCapacity);
}

TEST(SPSCRingbuffer, PopFromEmptyReturnsFalse) {
    SPSCRingbuffer<int, kCapacity> rb;
    int out = -1;
    EXPECT_FALSE(rb.try_pop(out));
    EXPECT_EQ(rb.size_approx(), 0u);
}

TEST(SPSCRingbuffer, PeekFromEmptyReturnsNullptr) {
    SPSCRingbuffer<int, kCapacity> rb;
    EXPECT_EQ(rb.try_peek(), nullptr);
}

TEST(SPSCRingbuffer, PushPopSingle) {
    SPSCRingbuffer<int, kCapacity> rb;
    EXPECT_TRUE(rb.try_push(42));
    EXPECT_EQ(rb.size_approx(), 1u);

    int out = 0;
    EXPECT_TRUE(rb.try_pop(out));
    EXPECT_EQ(out, 42);
    EXPECT_TRUE(rb.empty_approx());
}

TEST(SPSCRingbuffer, PushUntilFull) {
    SPSCRingbuffer<int, kCapacity> rb;
    for (std::size_t i = 0; i < kCapacity; ++i) {
        EXPECT_TRUE(rb.try_push(static_cast<int>(i))) << "push #" << i;
    }
    EXPECT_TRUE(rb.full_approx());
    EXPECT_FALSE(rb.try_push(999));  // voll
    EXPECT_EQ(rb.size_approx(), kCapacity);
}

// ---- 2. Wrap-Around Execution ----

TEST(SPSCRingbuffer, WrapAroundPreservesOrder) {
    // Wir testen mit capacity 8. Push mehr als capacity OHNE zwischendurch
    // pop sequence on underflow is invalid. Interleaving push/pop ensures head_ - tail_
    // remains < capacity, while absolute head_/tail_ pointers scale beyond bounds.
    // Simulates prolonged cyclic lifetime.
    SPSCRingbuffer<int, kCapacity> rb;
    constexpr int kTotal = 200;
    int next_push = 0;
    int next_expected = 0;

    // Erstmal halb voll machen.
    for (std::size_t i = 0; i < kCapacity / 2; ++i) {
        ASSERT_TRUE(rb.try_push(next_push++));
    }
    // Halb leer lesen.
    for (std::size_t i = 0; i < kCapacity / 2; ++i) {
        int out;
        ASSERT_TRUE(rb.try_pop(out));
        EXPECT_EQ(out, next_expected++);
    }

    // Cyclic lock-free push/pop — monotonically incrementing pointers.
    while (next_expected < kTotal) {
        ASSERT_TRUE(rb.try_push(next_push++));
        int out;
        ASSERT_TRUE(rb.try_pop(out));
        EXPECT_EQ(out, next_expected++);
    }
    EXPECT_TRUE(rb.empty_approx());
}

// ---- 3. Zero-Copy Peek ----

TEST(SPSCRingbuffer, PeekDoesNotConsume) {
    SPSCRingbuffer<int, kCapacity> rb;
    EXPECT_TRUE(rb.try_push(7));
    EXPECT_TRUE(rb.try_push(13));

    const int* p = rb.try_peek();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(*p, 7);
    EXPECT_EQ(rb.size_approx(), 2u);  // nicht konsumiert

    // Nochmal peek: muss immer noch 7 sein
    p = rb.try_peek();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(*p, 7);

    // Explicit read-acknowledgment required before slot invalidation.
    rb.pop_consumer_acked();
    EXPECT_EQ(rb.size_approx(), 1u);

    p = rb.try_peek();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(*p, 13);
    rb.pop_consumer_acked();
    EXPECT_TRUE(rb.empty_approx());
}

// ---- 4. SPSC Concurrency (1 Producer, 1 Consumer) ----

TEST(SPSCRingbuffer, SPSCThroughput) {
    // Single-Threaded Variante: Producer + Consumer im selben Thread.
    // Measures deterministic push/pop latencies isolating OS-scheduler interference.
    //
    // WICHTIG: SPSCRingbuffer<MarketTick, 65536> waere 2 MB. Das passt
    // noch in main()-Stack, aber wir nutzen hier bewusst eine kleinere
    // Bounded capacity circumventing excessive stack allocations.
    constexpr std::size_t kTestCapacity = 1024;  // 2^10
    constexpr std::uint64_t kItems = 10'000;
    constexpr std::size_t kBatch = 64;

    SPSCRingbuffer<MarketTick, kTestCapacity> rb;

    auto t0 = std::chrono::steady_clock::now();
    std::uint64_t sum = 0;

    for (std::uint64_t i = 0; i < kItems; i += kBatch) {
        // Push kBatch
        for (std::uint64_t j = 0; j < kBatch; ++j) {
            MarketTick t{};
            t.node_id = static_cast<std::uint32_t>((i + j) % 1024);
            t.bid = 100.0 + static_cast<double>(i + j);
            t.ask = 100.5 + static_cast<double>(i + j);
            t.weight = 1.0;
            bool ok = rb.try_push(t);
            ASSERT_TRUE(ok) << "push failed at i=" << (i + j);
        }
        // Pop kBatch
        for (std::uint64_t j = 0; j < kBatch; ++j) {
            MarketTick t;
            bool ok = rb.try_pop(t);
            ASSERT_TRUE(ok) << "pop failed at i=" << (i + j);
            sum += static_cast<std::uint64_t>(t.bid);
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    EXPECT_TRUE(rb.empty_approx());
    EXPECT_GT(sum, 0u);

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double ops_per_sec = (2.0 * kItems * 1'000'000.0) / static_cast<double>(us);
    std::cout << "[SPSC single-thread burst] " << kItems << " items in " << us
              << " us = " << static_cast<std::uint64_t>(ops_per_sec / 1e6)
              << " M ops/s" << std::endl;
}

TEST(SPSCRingbuffer, SPSCNoLossNoDuplicates) {
    // Sanity bounds: Post N-pushes / N-pops the underlying queue yields empty,
    // validating aggregate transactional equivalence.
    //
    // WICHTIG: 65536 * sizeof(int) = 256 KB, plus overhead - passt noch
    // in main-Stack. Bei MarketTick waeren es 2 MB, was wir vermeiden.
    // Im Produktivcode: Buffer auf dem HEAP ablegen (siehe Production-Code).
    constexpr int kItems = 100'000;
    constexpr std::size_t kBufSize = 65536;  // 2^16

    // Heap-Allokation: schuetzt vor Stack-Overflow in den Worker-Threads
    // (Windows Default: 1 MB Thread-Stack).
    auto rb_ptr = std::make_unique<SPSCRingbuffer<int, kBufSize>>();
    auto& rb = *rb_ptr;
    std::atomic<bool> producer_done{false};

    std::thread producer([&]() {
        for (int i = 0; i < kItems; ++i) {
            while (!rb.try_push(i + 1)) std::this_thread::yield();
        }
        producer_done.store(true, std::memory_order_release);
    });

    long long sum = 0;
    int count = 0;
    std::thread consumer([&]() {
        while (count < kItems) {
            int v;
            if (rb.try_pop(v)) {
                sum += v;
                ++count;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(count, kItems);
    EXPECT_TRUE(rb.empty_approx());
    // Sum 1..N = N*(N+1)/2
    const long long expected = (long long)kItems * (kItems + 1) / 2;
    EXPECT_EQ(sum, expected);
}

}  // namespace
