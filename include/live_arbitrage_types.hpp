// =============================================================================
// live_arbitrage_types.hpp - Shared Types Between Host and Device
// =============================================================================
// Included by BOTH host (C++) and device (CUDA) compilation units
// (e.g., live_arbitrage.cu and live_arbitrage_test.cpp). Strict POD enforcement,
// no extern "C", no CUDA-exclusive types.
// =============================================================================
#pragma once

#include <cstdint>

#include "market_tick.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

namespace phase7 {

// 32-Byte cacheline-aligned Edge struct for coalesced VRAM access.
struct alignas(16) GraphEdge {
    std::uint32_t source;
    std::uint32_t target;
    double        expected_weight;
    double        tolerance;
};
static_assert(sizeof(GraphEdge) == 32, "GraphEdge must be exactly 32 bytes");

// 64-Byte cacheline-aligned Opportunity payload dispatched by the kernel.
struct alignas(32) Opportunity {
    std::uint64_t tick_head;
    std::uint32_t source;
    std::uint32_t target;
    double        observed_bid;
    double        observed_ask;
    double        expected_weight;
    double        profit_estimate;
    std::uint64_t kernel_tsc;
};
static_assert(sizeof(Opportunity) == 64, "Opportunity must be exactly 64 bytes");

}  // namespace phase7

#ifdef _MSC_VER
#pragma warning(pop)
#endif
