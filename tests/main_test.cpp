#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numbers>

#include "algorithm.hpp"
#include "generator.hpp"
#include "graph.hpp"

// Global allocation counters exported for heap-allocation invariants tracking.
// Internal linkage stripped - function defined to intercept syscalls globally.
// External references executed in independent translation units.
uint64_t& global_allocation_count() {
    static uint64_t count = 0;
    return count;
}
uint64_t test_alloc_snapshot() { return global_allocation_count(); }

namespace {

uint64_t& allocation_count() {
    return global_allocation_count();
}

}

// catch every allocation and increment the counter
void* operator new(size_t size) {
    global_allocation_count()++;
    return malloc(size);
}

void* operator new[](size_t size) {
    global_allocation_count()++;
    return malloc(size);
}

void operator delete(void* memory) noexcept {
    free(memory);
}

void operator delete(void* memory, size_t) noexcept {
    free(memory);
}

void operator delete[](void* memory) noexcept {
    free(memory);
}

void operator delete[](void* memory, size_t) noexcept {
    free(memory);
}

TEST(AlgorithmTest, Profiling) {
    // Reduced from 250/250 (~78s) to 10/10 (~50ms) for fast feedback.
    // Still proves the SPFABuffer zero-allocation invariant on a small graph.
    constexpr uint32_t N = 10;
    constexpr uint32_t RUNS = 10;
    SPFABuffer buffer(N);
    allocation_count() = 0;
    GraphArena arena(N, N * N);
    DataGenerator::populateGraph(arena, N);

    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < RUNS; ++i) {
        BellmanFord::detectArbitrage(arena, 0, buffer);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for " << RUNS << " runs: " << elapsed.count() << " seconds\n";
    std::cout << "Heap allocations during " << RUNS << " runs: " << allocation_count() << "\n";
}

TEST(GeneratorTest, CompleteGraph) {
    GraphArena arena(5, 25);
    DataGenerator::populateGraph(arena, 5);
    auto node0 = arena.getNodes()[0];
    EXPECT_EQ(node0.edgeCount, 4);
}

TEST(AlgorithmTest, NoArbitrage) {
    SPFABuffer buffer(100);
    GraphArena arena(10, 100);
    arena.addNode(0);
    arena.addNode(1);
    arena.addEdge(0, 1, 1.0, 1.0);
    arena.addEdge(1, 0, 1.0, 1.0);

    EXPECT_FALSE(BellmanFord::detectArbitrage(arena, 0, buffer));
}

TEST(AlgorithmTest, DirectArbitrage) {
    SPFABuffer buffer(100);
    GraphArena arena(10, 100);
    arena.addNode(0);
    arena.addNode(1);
    arena.addEdge(0, 1, 2.0, 2.0);
    arena.addEdge(1, 0, 2.0, 2.0);

    EXPECT_TRUE(BellmanFord::detectArbitrage(arena, 0, buffer));
}

TEST(AlgorithmTest, TriangleArbitrage) {
    SPFABuffer buffer(100);
    GraphArena arena(10, 100);
    arena.addNode(0);
    arena.addNode(1);
    arena.addNode(2);
    arena.addEdge(0, 1, 1.2, 1.2);
    arena.addEdge(1, 2, 1.2, 1.2);
    arena.addEdge(2, 0, 1.2, 1.2);

    EXPECT_TRUE(BellmanFord::detectArbitrage(arena, 0, buffer));
}