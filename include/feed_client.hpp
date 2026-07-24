// =============================================================================
// feed_client.hpp - CRTP Base for Network Transports
// =============================================================================
//
// ARCHITECTURE :
//   - TransportBase is a CRTP template class. Derived classes handle network
//     I/O (WebSocket, UDP, TCP).
//   - ParserPolicy handles the zero-allocation parsing of raw bytes to MarketTick.
//   - Ensures NO virtual functions or vtable lookups on the hot path.
// =============================================================================
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "market_tick.hpp"
#include "spsc_ringbuffer.hpp"
#include "wal/wal_writer.hpp"

namespace phase7 {

template <typename Derived, typename ParserPolicy>
class TransportBase {
public:
    using RingBuffer = SPSCRingbuffer<MarketTick, 65536>;

    struct Config {
        std::string host = "stream.binance.com";
        std::string port = "9443";
        std::string target = "/ws/btcusdt@ticker";
        std::uint32_t max_reconnect_attempts = 0;
        std::chrono::milliseconds reconnect_delay{1000};
        std::uint32_t node_id_override = UINT32_MAX;
        std::unordered_map<std::string, std::uint32_t> symbol_to_node_id;
        int io_core_id = -1;
    };

    TransportBase(Config cfg, std::shared_ptr<RingBuffer> ringbuffer, std::shared_ptr<wal::WalWriter::RingBuffer> wal_ringbuffer = nullptr)
        : cfg_(std::move(cfg)), ringbuffer_(std::move(ringbuffer)), wal_ringbuffer_(std::move(wal_ringbuffer)) {
        
        // LUT lowercase pre-population
        if (!cfg_.symbol_to_node_id.empty()) {
            std::vector<std::pair<std::string, std::uint32_t>> lowers;
            lowers.reserve(cfg_.symbol_to_node_id.size());
            for (const auto& [sym, nid] : cfg_.symbol_to_node_id) {
                bool is_lower_already = true;
                for (char c : sym) {
                    if (c >= 'A' && c <= 'Z') {
                        is_lower_already = false;
                        break;
                    }
                }
                if (!is_lower_already) {
                    std::string lower;
                    lower.reserve(sym.size());
                    for (char c : sym) {
                        lower.push_back(static_cast<char>((c >= 'A' && c <= 'Z') ? (c + 32) : c));
                    }
                    if (lower != sym) {
                        lowers.emplace_back(std::move(lower), nid);
                    }
                }
            }
            for (auto& [k, v] : lowers) {
                cfg_.symbol_to_node_id.emplace(std::move(k), v);
            }
        }
    }

    void run() {
        static_cast<Derived*>(this)->do_run();
    }

    void stop() {
        static_cast<Derived*>(this)->do_stop();
    }

    // Diagnostics
    [[nodiscard]] std::uint64_t ticks_pushed() const noexcept { return ticks_pushed_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t parse_errors() const noexcept { return parse_errors_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t drops_full() const noexcept { return drops_full_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t unknown_symbol_drops() const noexcept {
        return unknown_symbol_drops_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool is_running() const noexcept { return running_.load(std::memory_order_acquire); }

    // Ingestion Latency Probe
    static constexpr std::size_t kIngestBuckets = 6;
    [[nodiscard]] std::uint64_t ingest_count() const noexcept { return ingest_count_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t ingest_max_ns() const noexcept { return ingest_max_ns_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t ingest_sum_ns() const noexcept { return ingest_sum_ns_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t ingest_bucket(std::size_t i) const noexcept {
        return (i < kIngestBuckets) ? ingest_hist_[i].load(std::memory_order_relaxed) : 0;
    }
    [[nodiscard]] double ingest_mean_ns() const noexcept {
        const auto c = ingest_count();
        return c ? static_cast<double>(ingest_sum_ns()) / c : 0.0;
    }

protected:
    // Called by Derived transport when a raw network message is received.
    void on_message(std::string_view raw) {
        const auto t0 = std::chrono::high_resolution_clock::now();

        MarketTick tick{};
        std::string_view symbol_view{};
        
        auto err = parser_.parse(raw, tick, symbol_view);
        
        if (err == 0 /* SUCCESS */) {
            if (cfg_.node_id_override != UINT32_MAX) {
                tick.node_id = cfg_.node_id_override;
            } else if (!cfg_.symbol_to_node_id.empty()) {
                const std::string key(symbol_view);
                const auto it = cfg_.symbol_to_node_id.find(key);
                if (it == cfg_.symbol_to_node_id.end()) {
                    unknown_symbol_drops_.fetch_add(1, std::memory_order_relaxed);
                    goto skip_push;
                }
                tick.node_id = it->second;
            }
            
            if (ringbuffer_->try_push(tick)) {
                ticks_pushed_.fetch_add(1, std::memory_order_relaxed);
            } else {
                drops_full_.fetch_add(1, std::memory_order_relaxed);
            }

            if (wal_ringbuffer_) {
                wal::WalEvent we{};
                we.timestamp_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::system_clock::now().time_since_epoch()).count());
                we.node_id = tick.node_id;
                we.type = wal::EventType::MARKET_TICK_BID; // or ask depending on parser, assuming bid for now
                we.flags = 0;
                we.payload.tick.price = tick.bid > 0 ? tick.bid : tick.ask;
                we.payload.tick.amount = tick.weight;
                (void)wal_ringbuffer_->try_push(we);
            }
        } else {
            parse_errors_.fetch_add(1, std::memory_order_relaxed);
        }

    skip_push:;
        if (err == 0) {
            const auto t1 = std::chrono::high_resolution_clock::now();
            const std::uint64_t ns =
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

            std::size_t b = 0;
            std::uint64_t n = ns;
            while (n >= 10 && b + 1 < kIngestBuckets) {
                n /= 10;
                ++b;
            }
            if (b >= kIngestBuckets) b = kIngestBuckets - 1;

            ingest_hist_[b].fetch_add(1, std::memory_order_relaxed);
            ingest_count_.fetch_add(1, std::memory_order_relaxed);
            ingest_sum_ns_.fetch_add(ns, std::memory_order_relaxed);

            std::uint64_t cur_max = ingest_max_ns_.load(std::memory_order_relaxed);
            while (ns > cur_max && !ingest_max_ns_.compare_exchange_weak(cur_max, ns, std::memory_order_relaxed)) {}
        }
    }

    Config cfg_;
    std::shared_ptr<RingBuffer> ringbuffer_;
    std::shared_ptr<wal::WalWriter::RingBuffer> wal_ringbuffer_;
    ParserPolicy parser_;

    std::atomic<bool> running_{false};
    std::uint32_t reconnect_attempts_ = 0;
    
    std::atomic<std::uint64_t> ticks_pushed_{0};
    std::atomic<std::uint64_t> parse_errors_{0};
    std::atomic<std::uint64_t> drops_full_{0};
    std::atomic<std::uint64_t> unknown_symbol_drops_{0};

    std::atomic<std::uint64_t> ingest_count_{0};
    std::atomic<std::uint64_t> ingest_sum_ns_{0};
    std::atomic<std::uint64_t> ingest_max_ns_{0};
    std::array<std::atomic<std::uint64_t>, kIngestBuckets> ingest_hist_{};
};

}  // namespace phase7
