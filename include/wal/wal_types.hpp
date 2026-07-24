// =============================================================================
// wal_types.hpp - Normalized Data Structure for Write-Ahead Log
// =============================================================================
#pragma once

#include <cstdint>
#include <type_traits>

namespace phase7::wal {

enum class EventType : std::uint8_t {
    MARKET_TICK_BID = 0,
    MARKET_TICK_ASK = 1,
    TRADE_EXECUTION = 2,
    ORDER_SUBMIT    = 3,
    SYSTEM_STATE    = 4
};

// 64-byte aligned structure to perfectly fit an L1/L2 cache line.
struct alignas(64) WalEvent {
    std::uint64_t timestamp_ns;  // Epoch nanoseconds
    std::uint32_t node_id;       // CSR Graph Vertex ID
    EventType     type;
    std::uint8_t  flags;
    std::uint16_t pad0;          // Explicit padding
    
    // Union to overlay different event types without dynamic allocation
    union Payload {
        struct {
            double price;
            double amount;
        } tick;
        
        struct {
            std::uint64_t order_id;
            double        executed_price;
            double        executed_amount;
        } trade;
        
        std::uint8_t raw[48];    // 64 - 16 bytes overhead = 48 bytes payload
    } payload;
};

static_assert(sizeof(WalEvent) == 64, "WalEvent must exactly match a 64-byte cache line.");
static_assert(std::is_trivially_copyable_v<WalEvent>, "WalEvent must be purely POD for fast SPSC memcpy.");

} // namespace phase7::wal
