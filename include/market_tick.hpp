// =============================================================================
// market_tick.hpp - POD tick circulating through the SPSC ring buffer.
// =============================================================================
// Trivially copyable is MANDATORY for try_push(T&&) using memcpy. Non-trivial
// types (e.g., std::string) would trigger allocations on the hot-path,
// violating our latency constraints.
//
// Layout (32 bytes, 2 per 64-byte cache line, actual packing on MSVC):
//   [0..3]   node_id  : uint32_t - Vertex ID in the CSR graph
//   [4..7]   _pad0    : 4 bytes compiler padding (alignof(double) = 8)
//   [8..15]  bid      : double   - Bid price (Binance/Kraken)
//   [16..23] ask      : double   - Ask price
//   [24..31] weight   : double   - Confidence / Liquidity Score
// Total: 32 bytes. With 64-byte cache lines, exactly 2 ticks/line.
//
// Why alignas(16)?
//   AVX-256 demands 32-byte alignment for vmovaps, AVX-512 demands 64-byte.
//   simdjson utilizes AVX-512, thus 64-byte alignment. Our MarketTick is merely
//   32 bytes; 16-byte alignment is sufficient and guarantees the 8-byte doubles
//   remain unfragmented across cache lines.
// =============================================================================
#pragma once

#include <cstdint>
#include <type_traits>

struct alignas(16) MarketTick {
    std::uint32_t node_id{0};
    double        bid{0.0};
    double        ask{0.0};
    double        weight{0.0};
};

static_assert(sizeof(MarketTick) == 32,
              "MarketTick must be exactly 32 bytes: 4 (node_id) + 4 (pad) + "
              "8 (bid) + 8 (ask) + 8 (weight) = 32");
static_assert(alignof(MarketTick) >= 8, "MarketTick must be at least 8-byte aligned (alignof(double))");
static_assert(std::is_trivially_copyable_v<MarketTick>,
              "MarketTick must remain trivially copyable (no user dtor, "
              "no virtual functions, purely POD)");
