// =============================================================================
// stream_parser.hpp - Zero-Allocation Binance/Kraken Ticker Parser
// =============================================================================
// PARSING PHILOSOPHY :
//
//   - simdjson On-Demand parses the JSON STREAM, avoiding intermediate ASTs.
//   - STRICT NO `dom::parser` usage (which allocates a Document per parse).
//   - `parser.iterate(ptr, len, cap)` mandates a buffer equipped with SIMD
//     padding (cap >= len + SIMDJSON_PADDING, defaulting to 64 bytes).
//   - `get_double()` on a `number` executes with zero allocations.
//   - `get_string()` returns a `std::string_view` avoiding heap copies.
//
// PRE-ALLOCATION:
//   We allocate a `simdjson::ondemand::parser` EXACTLY ONCE per Engine instance.
//   Per tick, we utilize a thread-local stack buffer (or the buffer provided
//   by Beast in Step 4). ZERO malloc/free invocations per tick.
//
// VERTEX MAPPING:
//   Symbol (e.g., "BTCUSDT") -> node_id via 64-bit FNV-1a. Hash collisions
//   are acceptable (in a 100k-vertex domain, the collision probability
//   with 20-50 active symbols approaches zero).
// =============================================================================
#pragma once

#include <simdjson.h>

#include <charconv>
#include <cstdint>
#include <string_view>
#include <system_error>
#include <utility>

#include "market_tick.hpp"

namespace phase7 {

// FNV-1a 64-Bit Hash: computationally cheap, deterministic, zero-alloc.
// Maps symbol strings (typically 6-12 ASCII bytes) directly to a uint32_t node_id.
[[nodiscard]] inline std::uint32_t fnv1a_32(std::string_view s) noexcept {
    std::uint32_t hash = 2166136261u;
    for (char c : s) {
        hash ^= static_cast<std::uint8_t>(c);
        hash *= 16777619u;
    }
    return hash;
}

// Parses ONE Binance ticker JSON and populates 'out'.
// json: raw network buffer (not null-terminated, length = json.size()).
//       MUST reside within a buffer >= json.size() + SIMDJSON_PADDING.
// parser: persistent instance, maintained by caller (NOT local to this function).
//
// ZERO-ALLOC: strictly prohibits heap allocations inside this function. The parser
// internally utilizes tape structures allocated EXACTLY ONCE.
//
// : 4-argument variant featuring symbol_out. Callers receive the
// extracted symbol string (e.g., "BTCUSDT" or "btcusdt" for the combined
// stream "btcusdt@ticker") to query an LUT. The string_view
// references the 'json' buffer, which must outlive the function invocation.
[[nodiscard]] inline simdjson::error_code parse_binance_ticker(std::string_view            json,
                                                               MarketTick&                 out,
                                                               simdjson::ondemand::parser& parser,
                                                               std::string_view&           symbol_out) noexcept {
    // iterate() strictly requires length + capacity (due to SIMD padding).
    // The caller assumes responsibility for guaranteeing the buffer contains
    // at least json.size() + SIMDJSON_PADDING bytes.
    simdjson::ondemand::document doc;
    auto err = parser.iterate(json.data(), json.size(), json.size() + simdjson::SIMDJSON_PADDING).get(doc);
    if (err) return err;

    simdjson::ondemand::object root;
    err = doc.get_object().get(root);
    if (err) return err;

    // =====================================================================
    // Binance transmits via two primary formats:
    //   1) Single-Stream:  {"e":"24hrTicker","E":...,"s":"BTCUSDT","b":"...","a":"..."}
    //   2) Combined-Stream: {"stream":"btcusdt@ticker","data":{"e":"...","s":"BTCUSDT",...}}
    //
    // We must deterministically identify the active schema. For Combined-Streams,
    // bid/ask values reside entirely within the "data" payload, NOT top-level.
    //
    // DETECTION: Combined formats possess both "stream" and "data". If "stream"
    // is successfully parsed, we repoint the root to "data".
    // =====================================================================
    simdjson::ondemand::object& active_root = root;
    std::string_view            symbol_str;

    // Attempt: Combined-Stream Format verification
    simdjson::ondemand::value  stream_val;
    simdjson::ondemand::object data_obj;
    if (root["stream"].get(stream_val) == simdjson::SUCCESS) {
        std::string_view stream_name;
        if (stream_val.get_string().get(stream_name) == simdjson::SUCCESS) {
            // Stream format: "btcusdt@ticker" -> Extracts symbol "btcusdt".
            const auto at_pos = stream_name.find('@');
            symbol_str        = (at_pos == std::string_view::npos) ? stream_name : stream_name.substr(0, at_pos);
        }
        // Transition active root to the "data" sub-object.
        if (root["data"].get_object().get(data_obj) == simdjson::SUCCESS) {
            active_root = data_obj;  // subsequent bid/ask extractions target this payload
        }
    }

    // Fallback / Single-Stream: Extract symbol directly from "s".
    if (symbol_str.empty()) {
        if (active_root["s"].get_string().get(symbol_str) != simdjson::SUCCESS) {
            return simdjson::INCORRECT_TYPE;
        }
    }

    // symbol_str now contains the Binance symbol (e.g., "BTCUSDT" or
    // "btcusdt" for Combined). The CALLER assumes responsibility for translating
    // this to a numeric node_id (via LUT). We retain the FNV-1a hash
    // exclusively as a fallback mechanism if an LUT is bypassed.
    symbol_out  = symbol_str;  // : empowers caller to execute LUT-lookup
    out.node_id = fnv1a_32(symbol_str);

    // bid extraction (Binance transmits bid/ask as JSON-STRINGS, not Numbers)
    std::string_view bid_str;
    err = active_root["b"].get_string().get(bid_str);
    if (err) return err;
    auto bid_p = std::from_chars(bid_str.data(), bid_str.data() + bid_str.size(), out.bid);
    if (bid_p.ec != std::errc{}) return simdjson::INCORRECT_TYPE;

    // ask extraction
    std::string_view ask_str;
    err = active_root["a"].get_string().get(ask_str);
    if (err) return err;
    auto ask_p = std::from_chars(ask_str.data(), ask_str.data() + ask_str.size(), out.ask);
    if (ask_p.ec != std::errc{}) return simdjson::INCORRECT_TYPE;

    // weight: Default 1.0. The Binance ticker provides no direct 'weight'.
    // We derive the score from spread proximity: weight = 1.0 / (ask - bid).
    // Tighter spreads = increased liquidity = higher algorithmic score.
    const double spread = out.ask - out.bid;
    out.weight          = (spread > 0.0) ? (1.0 / spread) : 1.0;

    return simdjson::SUCCESS;
}

// Convenience overload dropping symbol_out (legacy execution path, ).
// Defined inline POST 4-arg variant; otherwise the 3-arg overload fails
// to resolve the underlying 4-arg implementation.
[[nodiscard]] inline simdjson::error_code parse_binance_ticker(std::string_view            json,
                                                               MarketTick&                 out,
                                                               simdjson::ondemand::parser& parser) noexcept {
    std::string_view sym_dummy;
    return parse_binance_ticker(json, out, parser, sym_dummy);
}

}  // namespace phase7
