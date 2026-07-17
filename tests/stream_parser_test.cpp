// =============================================================================
// stream_parser_test.cpp - Zero-Allocation Parsing Validation
// =============================================================================
//
// HEAP-COUNTER-TEST:
//   main_test.cpp definiert global::operator new/delete und schreibt in
//   global_allocation_count() (exposed in tests/main_test.cpp). Wir
//   nutzen test_alloc_snapshot() von dort, um den Parser-Hot-Path auf
//   Allokationen zu pruefen.
// =============================================================================

#include "stream_parser.hpp"
#include "market_tick.hpp"

#include <gtest/gtest.h>
#include <simdjson.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Aus main_test.cpp (exposed).
uint64_t test_alloc_snapshot();

namespace {

struct PaddedBuffer {
    static constexpr std::size_t kPad = simdjson::SIMDJSON_PADDING;  // 64
    std::vector<char> data;  // size() = json_len + kPad

    explicit PaddedBuffer(std::string_view json) {
        data.resize(json.size() + kPad);
        std::memcpy(data.data(), json.data(), json.size());
        std::memset(data.data() + json.size(), 0, kPad);
    }
    [[nodiscard]] const char* ptr() const noexcept { return data.data(); }
    [[nodiscard]] std::size_t size() const noexcept {
        return data.size() - kPad;
    }
    [[nodiscard]] std::size_t capacity() const noexcept { return data.size(); }
};

constexpr std::string_view kBinanceBtcTicker = R"({"e":"24hrTicker","E":1234567890,"s":"BTCUSDT","b":"42500.10","a":"42500.50","h":"42800.00","l":"42100.00","v":"12345.67","q":"525000000.00"})";

TEST(StreamParser, BinanceBasicTicker) {
    simdjson::ondemand::parser parser;
    PaddedBuffer buf(kBinanceBtcTicker);

    MarketTick tick{};
    auto err = phase7::parse_binance_ticker(
        std::string_view(buf.ptr(), buf.size()), tick, parser);
    ASSERT_EQ(err, simdjson::SUCCESS) << simdjson::error_message(err);

    EXPECT_GT(tick.node_id, 0u);
    EXPECT_DOUBLE_EQ(tick.bid, 42500.10);
    EXPECT_DOUBLE_EQ(tick.ask, 42500.50);
    EXPECT_GT(tick.weight, 0.0);
}

TEST(StreamParser, DeterministicNodeId) {
    EXPECT_EQ(phase7::fnv1a_32("BTCUSDT"), phase7::fnv1a_32("BTCUSDT"));
    EXPECT_NE(phase7::fnv1a_32("BTCUSDT"), phase7::fnv1a_32("ETHUSDT"));
    EXPECT_NE(phase7::fnv1a_32("BTCUSDT"), 0u);
}

TEST(StreamParser, DifferentSymbolsDifferentIds) {
    auto a = phase7::fnv1a_32("XBT/USD");
    auto b = phase7::fnv1a_32("BTCUSDT");
    auto c = phase7::fnv1a_32("ETHUSDT");
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    EXPECT_NE(a, c);
}

TEST(StreamParser, MalformedJsonReturnsError) {
    simdjson::ondemand::parser parser;
    constexpr std::string_view bad = R"({"s":"BTCUSDT","b":"oops")";
    PaddedBuffer buf(bad);

    MarketTick tick{};
    auto err = phase7::parse_binance_ticker(
        std::string_view(buf.ptr(), buf.size()), tick, parser);
    EXPECT_NE(err, simdjson::SUCCESS);
}

TEST(StreamParser, MissingBidFieldReturnsError) {
    simdjson::ondemand::parser parser;
    constexpr std::string_view no_bid = R"({"s":"BTCUSDT","a":"42500.50"})";
    PaddedBuffer buf(no_bid);

    MarketTick tick{};
    auto err = phase7::parse_binance_ticker(
        std::string_view(buf.ptr(), buf.size()), tick, parser);
    EXPECT_NE(err, simdjson::SUCCESS);
}

TEST(StreamParser, MultipleConsecutiveParsesStable) {
    simdjson::ondemand::parser parser;
    PaddedBuffer buf(kBinanceBtcTicker);

    for (int i = 0; i < 10'000; ++i) {
        MarketTick tick{};
        auto err = phase7::parse_binance_ticker(
            std::string_view(buf.ptr(), buf.size()), tick, parser);
        ASSERT_EQ(err, simdjson::SUCCESS) << "at i=" << i;
        ASSERT_GT(tick.node_id, 0u);
        ASSERT_GT(tick.bid, 0.0);
    }
}

TEST(StreamParser, HeapAllocationSnapshot) {
    simdjson::ondemand::parser parser;
    PaddedBuffer buf(kBinanceBtcTicker);

    // Initial parsing dispatch to trigger framework setup (one-time allocation).
    {
        MarketTick tick{};
        auto err = phase7::parse_binance_ticker(
            std::string_view(buf.ptr(), buf.size()), tick, parser);
        ASSERT_EQ(err, simdjson::SUCCESS);
    }

    const uint64_t before = test_alloc_snapshot();
    for (int i = 0; i < 1'000; ++i) {
        MarketTick tick{};
        auto err = phase7::parse_binance_ticker(
            std::string_view(buf.ptr(), buf.size()), tick, parser);
        ASSERT_EQ(err, simdjson::SUCCESS);
    }
    const uint64_t after = test_alloc_snapshot();

    const uint64_t delta = after - before;
    std::cout << "[Parser hot path] 1000 ticks | heap alloc delta = "
              << delta << std::endl;
    EXPECT_EQ(delta, 0u) << "Parser allokiert im Hot-Path! delta=" << delta;
}

// =============================================================================
// Multi-Symbol validation (Combined-Stream-Format + symbol_out)
// =============================================================================

constexpr std::string_view kBinanceCombinedEth = R"({"stream":"ethusdt@ticker","data":{"e":"24hrTicker","E":1234567890,"s":"ETHUSDT","b":"3000.10","a":"3000.50"}})";

TEST(StreamParser, CombinedStreamExtractsSymbol) {
    simdjson::ondemand::parser parser;
    PaddedBuffer buf(kBinanceCombinedEth);

    MarketTick tick{};
    std::string_view sym_out{};
    auto err = phase7::parse_binance_ticker(
        std::string_view(buf.ptr(), buf.size()), tick, parser, sym_out);
    ASSERT_EQ(err, simdjson::SUCCESS) << simdjson::error_message(err);

    // symbol_out must isolate the stream nomenclature excluding @-suffix ("ethusdt").
    EXPECT_EQ(sym_out, "ethusdt");
    // node_id resolution via FNV-1a hash of "ethusdt" (fallback sequence).
    EXPECT_EQ(tick.node_id, phase7::fnv1a_32("ethusdt"));
    // bid/ask muessen korrekt aus "data" extrahiert sein.
    EXPECT_DOUBLE_EQ(tick.bid, 3000.10);
    EXPECT_DOUBLE_EQ(tick.ask, 3000.50);
}

TEST(StreamParser, SingleStreamStillWorks) {
    simdjson::ondemand::parser parser;
    PaddedBuffer buf(kBinanceBtcTicker);

    MarketTick tick{};
    std::string_view sym_out{};
    auto err = phase7::parse_binance_ticker(
        std::string_view(buf.ptr(), buf.size()), tick, parser, sym_out);
    ASSERT_EQ(err, simdjson::SUCCESS);

    // Single-stream payload resolution designates "BTCUSDT" (extracted from "s"-field).
    EXPECT_EQ(sym_out, "BTCUSDT");
    EXPECT_EQ(tick.node_id, phase7::fnv1a_32("BTCUSDT"));
    EXPECT_DOUBLE_EQ(tick.bid, 42500.10);
}

TEST(StreamParser, CombinedStreamNoAtSuffix) {
    // Edge case: stream = "ethusdt" ohne @-Suffix.
    constexpr std::string_view kNoSuffix = R"({"stream":"ethusdt","data":{"s":"ETHUSDT","b":"1.0","a":"1.1"}})";
    simdjson::ondemand::parser parser;
    PaddedBuffer buf(kNoSuffix);

    MarketTick tick{};
    std::string_view sym_out{};
    auto err = phase7::parse_binance_ticker(
        std::string_view(buf.ptr(), buf.size()), tick, parser, sym_out);
    ASSERT_EQ(err, simdjson::SUCCESS);
    EXPECT_EQ(sym_out, "ethusdt");
}

}  // namespace
