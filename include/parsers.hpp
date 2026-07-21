// =============================================================================
// parsers.hpp - Zero-Allocation Deserialization Policies
// =============================================================================
#pragma once

#include <simdjson.h>
#include <string_view>
#include "market_tick.hpp"
#include "stream_parser.hpp" // For phase7::parse_binance_ticker

namespace phase7 {

// Policy for JSON (Binance format) using simdjson ondemand.
struct BinanceJsonParser {
    simdjson::ondemand::parser parser;

    // Returns an integer status (0 for success).
    int parse(std::string_view raw, MarketTick& out, std::string_view& symbol_out) {
        auto err = phase7::parse_binance_ticker(raw, out, parser, symbol_out);
        return (err == simdjson::SUCCESS) ? 0 : 1;
    }
};

#pragma pack(push, 1)
// Mock SBE Message Header (Standard CME Globex style)
struct SbeMessageHeader {
    std::uint16_t blockLength;
    std::uint16_t templateId;
    std::uint16_t schemaId;
    std::uint16_t version;
};

// Mock SBE Market Data Tick Payload
struct SbeTickPayload {
    char          symbol[8]; // e.g. "BTCUSDT\0"
    double        bid;
    double        ask;
};
#pragma pack(pop)

// SBE Parser (Zero-Allocation, Zero-Copy)
struct SbeParser {
    int parse(std::string_view raw, MarketTick& out, std::string_view& symbol_out) {
        if (raw.size() < sizeof(SbeMessageHeader) + sizeof(SbeTickPayload)) {
            return 1; // Incorrect size
        }

        // Direct pointer cast for zero-copy access
        const auto* header = reinterpret_cast<const SbeMessageHeader*>(raw.data());
        
        // In a real scenario, we would switch on header->templateId
        if (header->templateId != 42) { // Arbitrary mock template ID
            return 1;
        }

        const auto* payload = reinterpret_cast<const SbeTickPayload*>(raw.data() + sizeof(SbeMessageHeader));

        // Create a string_view pointing directly into the raw buffer. No copies.
        // We find the null terminator or use the full 8 chars.
        size_t sym_len = 0;
        while (sym_len < 8 && payload->symbol[sym_len] != '\0') {
            ++sym_len;
        }
        symbol_out = std::string_view(payload->symbol, sym_len);

        out.bid = payload->bid;
        out.ask = payload->ask;
        
        const double spread = out.ask - out.bid;
        out.weight = (spread > 0.0) ? (1.0 / spread) : 1.0;

        return 0; // Success
    }
};

// Stub for Protocol Buffers parser
struct ProtobufParserStub {
    int parse([[maybe_unused]] std::string_view raw, [[maybe_unused]] MarketTick& out, [[maybe_unused]] std::string_view& symbol_out) {
        // Implement Protobuf decoding here (e.g., using nanopb for zero-alloc)
        return -1; // Not implemented
    }
};

} // namespace phase7
