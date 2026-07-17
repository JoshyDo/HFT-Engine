// =============================================================================
// ws_client.hpp - Asynchronous Boost.Beast WebSocket Client
// =============================================================================
//
// ARCHITECTURE :
//   - A Beast WebSocket stream on an asio::io_context.
//   - Connect to Binance WSS endpoint: wss://stream.binance.com:9443/ws/btcusdt@ticker
//   - Reads messages into a std::string buffer (asio mutable_buffer).
//   - Per Message: phase7::parse_binance_ticker() -> try_push() to the SPSC Ringbuffer.
//   - On error or stream completion: async_reconnect.
//
// THREAD MODEL:
//   - ONE asio thread (asio::io_context) executes the Beast I/O.
//   - Parser also executes on this thread (zero-alloc, no multithreading required).
//   - Producer (Beast thread) -> SPSC -> Consumer (Host Control Loop, Step 5).
//
// BACKPRESSURE:
//   - SPSC is bounded (capacity 65536, 2 MB). If buffer is full:
//     - Option A: Drop tick (log, continue parsing) - HFT realistic
//     - Option B: Pause network (asio write via beast::websocket::pause_read())
//   - Currently: Drop tick + counter. If drops > 1%: alert.
// =============================================================================
#pragma once

#include <simdjson.h>

#include <array>
#include <atomic>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "market_tick.hpp"
#include "stream_parser.hpp"
#include "spsc_ringbuffer.hpp"
#include "thread_pinning.hpp"

namespace phase7 {

namespace beast     = boost::beast;
namespace websocket = beast::websocket;
namespace net       = boost::asio;
namespace ssl       = boost::asio::ssl;

class WebSocketTickerClient {
public:
    using RingBuffer   = SPSCRingbuffer<MarketTick, 65536>;
    using TickCallback = std::function<void(const MarketTick&)>;

    struct Config {
        std::string host   = "stream.binance.com";
        std::string port   = "9443";
        std::string target = "/ws/btcusdt@ticker";
        // Reconnect delay following a connection loss.
        std::chrono::milliseconds reconnect_delay{1000};
        // Maximum reconnect attempts per cycle before aborting.
        // 0 = infinite.
        std::uint32_t max_reconnect_attempts = 0;
        // : If != UINT32_MAX, the node_id parsed from JSON
        // (FNV-1a hash of the symbol) is substituted by this static value.
        // Necessary because the hash for "BTCUSDT", for example, is 2166198,
        // which would cause an OOB write at V=100k.
        // SUPERSEDED by symbol_to_node_id (+) when multi-symbol
        // is active. Both cannot be populated simultaneously.
        std::uint32_t node_id_override = UINT32_MAX;
        // : Symbol -> node_id LUT. If NOT empty, the symbol_str
        // parsed from JSON (e.g., "BTCUSDT" or "btcusdt" for combined
        // streams) is mapped here. Key is case-insensitive (both UPPER
        // and lower case are supported -  pre-populated).
        // node_id_override takes precedence if != UINT32_MAX.
        std::unordered_map<std::string, std::uint32_t> symbol_to_node_id;
        // : Pin io_thread_ to a logical core ID.
        // -1 = no pinning. Caller should ideally select 2 (discrete CCD).
        int io_core_id = -1;
    };

    WebSocketTickerClient(Config cfg, std::shared_ptr<RingBuffer> ringbuffer)
        : cfg_(std::move(cfg)),
          ringbuffer_(std::move(ringbuffer)),
          parser_(std::make_unique<simdjson::ondemand::parser>()),
          resolver_(net::make_strand(ioc_)),
          ws_(net::make_strand(ioc_), ssl_ctx_),
          io_core_id_(cfg.io_core_id) {
        // : LUT lowercase pre-population.
        //
        // Prior to this patch, the hot-path executed two lookups (one per
        // case variant) and a 6-12 byte ASCII loop in the miss-path with
        // manual lowercase conversion per tick. This incurred ~80-150ns
        // overhead per tick. With 8 symbols at 8 ticks/s, this is negligible
        // for aggregate latency, BUT we mandate an ABSOLUTE MINIMAL hot-path.
        //
        // Solution: Augment the LUT during construction with a secondary
        // lowercase entry per symbol. Example:
        //   cfg.symbol_to_node_id = {{"BTCUSDT", 0}, {"ETHUSDT", 1}, ...}
        //   becomes:
        //   {{"BTCUSDT", 0}, {"btcusdt", 0}, {"ETHUSDT", 1}, {"ethusdt", 1}}
        // This reduces the hot-path to EXACTLY ONE unordered_map<string,uint32_t>
        // lookup utilizing the binary string from the JSON (copied as-is
        // to a std::string).
        //
        // CRITICAL: This executes exactly ONCE per constructor (~10 symbols,
        // <1us overhead), entirely off the tick hot-path.
        if (!cfg_.symbol_to_node_id.empty()) {
            // Extract existing keys, append lower variants.
            // Requires a discrete vector to prevent iterator invalidation
            // during mutation.
            std::vector<std::pair<std::string, std::uint32_t>> lowers;
            lowers.reserve(cfg_.symbol_to_node_id.size());
            for (const auto& [sym, nid] : cfg_.symbol_to_node_id) {
                // If the key is not strictly lowercase, append variant.
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
                    // If lower != sym (e.g., Sym == "BTCUSDT"
                    // and lower "btcusdt"), append.
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

    ~WebSocketTickerClient() {
        stop();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }

    WebSocketTickerClient(const WebSocketTickerClient&)            = delete;
    WebSocketTickerClient& operator=(const WebSocketTickerClient&) = delete;

    // Initiates the asynchronous connect + read loop on an internal thread.
    // Idempotent: consecutive calls are no-ops.
    void run() {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) {
            return;  // already running
        }
        do_connect();
        const int core = io_core_id_;
        io_thread_     = std::thread([this, core]() {
            // : Pinning the Beast execution loop. SSL handshake
            // and parsing significantly benefit from L1 cache hit rates on
            // a dedicated core. Pinned to an ID disjoint from VRAMUpdater.
            if (core >= 0) {
                if (!::phase7::pin_current_thread_to_core(core)) {
                    std::fprintf(stderr, "[WSClient] WARNING: pinning to core %d failed\n", core);
                }
            }
            ioc_.run();
        });
    }

    // Terminates the client (EXTERNAL, thread-safe).
    void stop() {
        if (!running_.exchange(false)) return;
        // Best-effort: terminate Beast strand, then ioc_.stop()
        net::post(ioc_, [this]() {
            beast::error_code ec;
            ws_.close(websocket::close_code::normal, ec);
            ioc_.stop();
        });
    }

    // Diagnostics.
    [[nodiscard]] std::uint64_t ticks_pushed() const noexcept { return ticks_pushed_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t parse_errors() const noexcept { return parse_errors_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t drops_full() const noexcept { return drops_full_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t unknown_symbol_drops() const noexcept {
        return unknown_symbol_drops_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool is_running() const noexcept { return running_.load(std::memory_order_acquire); }

    // =====================================================================
    // : Ingestion Latency Probe (Histogram + Max + Mean).
    // ---------------------------------------------------------------------
    // Bucket boundaries in NANOSECONDS (logarithmic, base 10 factor):
    //   [0]:       <  1us
    //   [1]:       < 10us
    //   [2]:       <100us
    //   [3]:       <  1ms
    //   [4]:       < 10ms
    //   [5]:       >=10ms
    // Total: 6 Buckets. index = floor(log10(ns)) + 1, clamped to 5.
    // =====================================================================
    static constexpr std::size_t kIngestBuckets = 6;

    [[nodiscard]] std::uint64_t ingest_count() const noexcept { return ingest_count_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t ingest_max_ns() const noexcept {
        return ingest_max_ns_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t ingest_sum_ns() const noexcept {
        return ingest_sum_ns_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t ingest_bucket(std::size_t i) const noexcept {
        return (i < kIngestBuckets) ? ingest_hist_[i].load(std::memory_order_relaxed) : 0;
    }
    // Mean in NANOSECONDS (0 if no samples exist).
    [[nodiscard]] double ingest_mean_ns() const noexcept {
        const auto c = ingest_count();
        return c ? static_cast<double>(ingest_sum_ns()) / c : 0.0;
    }

private:
    // ---- SSL Configuration ----
    // ---- Internal State ----
    // WARNING: C++ member initialization order complies with the
    // CLASS DECLARATION order, NOT the initializer list order.
    // ioc_ and ssl_ctx_ MUST be declared PRIOR to ws_ because
    // ws_ relies on them via const-ref.
    net::io_context                                         ioc_;
    ssl::context                                            ssl_ctx_{ssl::context::tls_client};
    Config                                                  cfg_;
    std::shared_ptr<RingBuffer>                             ringbuffer_;
    std::unique_ptr<simdjson::ondemand::parser>             parser_;
    beast::flat_buffer                                      buffer_;  // ~16 KB internal, acceptable limit
    net::ip::tcp::resolver                                  resolver_;
    websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws_;
    std::thread                                             io_thread_;
    std::atomic<bool>                                       running_{false};
    // : Core pin target for the io_thread_ Beast loop.
    int                        io_core_id_         = -1;
    std::uint32_t              reconnect_attempts_ = 0;
    std::atomic<std::uint64_t> ticks_pushed_{0};
    std::atomic<std::uint64_t> parse_errors_{0};
    std::atomic<std::uint64_t> drops_full_{0};
    // : Ticks referencing a symbol absent from the LUT.
    std::atomic<std::uint64_t> unknown_symbol_drops_{0};
    // : Ingestion latency telemetry. Entirely atomic, lock-free.
    std::atomic<std::uint64_t>                             ingest_count_{0};
    std::atomic<std::uint64_t>                             ingest_sum_ns_{0};
    std::atomic<std::uint64_t>                             ingest_max_ns_{0};
    std::array<std::atomic<std::uint64_t>, kIngestBuckets> ingest_hist_{};

    // ---- Connect Logic ----
    void do_connect() {
        // SSL verify_none as  Mock; set to peer in Production.
        ws_.next_layer().native_handle();  // no-op, placeholder

        resolver_.async_resolve(
            cfg_.host, cfg_.port, beast::bind_front_handler(&WebSocketTickerClient::on_resolve, this));
    }

    void on_resolve(beast::error_code ec, net::ip::tcp::resolver::results_type results) {
        if (ec) {
            return schedule_reconnect(ec);
        }
        // SSL handshake + WebSocket handshake.
        beast::get_lowest_layer(ws_).expires_after(std::chrono::seconds(30));
        beast::get_lowest_layer(ws_).async_connect(results,
                                                   beast::bind_front_handler(&WebSocketTickerClient::on_connect, this));
    }

    void on_connect(beast::error_code ec, net::ip::tcp::resolver::results_type::endpoint_type) {
        if (ec) {
            return schedule_reconnect(ec);
        }
        // WebSocket stream configuration.
        ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
        ws_.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req) { req.set(beast::http::field::user_agent, "HFT-Engine/Phase7"); }));

        // SSL Handshake.
        ws_.next_layer().async_handshake(ssl::stream_base::client,
                                         beast::bind_front_handler(&WebSocketTickerClient::on_ssl_handshake, this));
    }

    void on_ssl_handshake(beast::error_code ec) {
        if (ec) {
            return schedule_reconnect(ec);
        }
        // WebSocket Handshake.
        ws_.async_handshake(
            cfg_.host, cfg_.target, beast::bind_front_handler(&WebSocketTickerClient::on_handshake, this));
    }

    void on_handshake(beast::error_code ec) {
        if (ec) {
            return schedule_reconnect(ec);
        }
        // Connected! Reset reconnect counter, initiate read loop.
        reconnect_attempts_ = 0;
        do_read();
    }

    // ---- Read Loop ----
    void do_read() {
        // We do not drain the buffer - Beast accumulates. We process
        // sequentially per message. flat_buffer allocates internally
        // on the heap beyond ~16KB, an acceptable trade-off (off hot-path).
        buffer_.consume(buffer_.size());
        ws_.async_read(buffer_, beast::bind_front_handler(&WebSocketTickerClient::on_read, this));
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);

        if (ec) {
            if (ec == websocket::error::closed) {
                return;  // standard termination
            }
            return schedule_reconnect(ec);
        }

        // We hold ONE distinct message. The Beast buffer encapsulates precisely
        // this payload. We parse and push the result to the SPSC Ringbuffer.
        std::string_view json{static_cast<const char*>(buffer_.data().data()), buffer_.data().size()};

        // : Profiling Parse + Lookup + Push. high_resolution_clock
        // wraps QueryPerformanceCounter on Windows (ns resolution). We benchmark
        // ONLY the happy-path (push successful OR dropped); parse_errors are
        // excluded from latency calculations to prevent mean distortion.
        const auto t0 = std::chrono::high_resolution_clock::now();

        MarketTick       tick{};
        std::string_view symbol_view{};  // : for LUT lookup
        auto             err = parse_binance_ticker(json, tick, *parser_, symbol_view);
        if (err == simdjson::SUCCESS) {
            // : legacy single-symbol override (e.g., BTCUSDT -> 0).
            if (cfg_.node_id_override != UINT32_MAX) {
                tick.node_id = cfg_.node_id_override;
            }
            // : Multi-Symbol LUT. If populated, we execute
            // a case-insensitive symbol lookup. The LUT is pre-populated
            // with lowercase key variants during constructor init
            // , distilling the hot-path to EXACTLY ONE
            // unordered_map::find operation.
            else if (!cfg_.symbol_to_node_id.empty()) {
                // std::string allocation is the sole requisite overhead here
                // (~24 bytes SSO for Symbol <=15 char, zero heap allocations),
                // but mandatory since heterogeneous lookup via std::string_view
                // fails without transparent Hash/Equal functors and we preclude
                // injecting extraneous dependencies.
                const std::string key(symbol_view);
                const auto        it = cfg_.symbol_to_node_id.find(key);
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
        } else {
            parse_errors_.fetch_add(1, std::memory_order_relaxed);
        }
    skip_push:;

        // Histogram update: exclusive to successfully parsed ticks.
        // trade-off: try_push=false (Drop) qualifies, as SPSC::try_push
        // injects inherent latency; we mandate visibility into backpressure events.
        if (err == simdjson::SUCCESS) {
            const auto          t1 = std::chrono::high_resolution_clock::now();
            const std::uint64_t ns =
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

            // Bucket index: log10(ns), clamped to [0, kIngestBuckets-1].
            // 0-999 ns → bucket 0 (<1us)
            // 1k-9.9k → 1 (<10us)
            // 10k-99.9k → 2 (<100us)
            // ...etc
            std::size_t   b = 0;
            std::uint64_t n = ns;
            while (n >= 10 && b + 1 < kIngestBuckets) {
                n /= 10;
                ++b;
            }
            // n>=1: min. 1 digit. b ∈ {0..4}. If ns >= 100000 (=10^5),
            // b=5. If ns >= 1000000, b=6 → clamps to 5.
            if (b >= kIngestBuckets) b = kIngestBuckets - 1;

            ingest_hist_[b].fetch_add(1, std::memory_order_relaxed);
            ingest_count_.fetch_add(1, std::memory_order_relaxed);
            ingest_sum_ns_.fetch_add(ns, std::memory_order_relaxed);

            // Max tracking via atomic CAS loop. Lock-free, retry on contention.
            std::uint64_t cur_max = ingest_max_ns_.load(std::memory_order_relaxed);
            while (ns > cur_max && !ingest_max_ns_.compare_exchange_weak(cur_max, ns, std::memory_order_relaxed)) {
                // cur_max invalidated, re-evaluate
            }
        }

        // Poll subsequent payload.
        if (running_.load(std::memory_order_acquire)) {
            do_read();
        }
    }

    // ---- Reconnect Logic ----
    void schedule_reconnect([[maybe_unused]] const beast::error_code& ec) {
        if (!running_.load(std::memory_order_acquire)) return;

        if (cfg_.max_reconnect_attempts != 0 && reconnect_attempts_ >= cfg_.max_reconnect_attempts) {
            running_.store(false, std::memory_order_release);
            ioc_.stop();
            return;
        }
        ++reconnect_attempts_;

        // Clean SSL stream termination.
        beast::error_code shutdown_ec;
        beast::get_lowest_layer(ws_).socket().shutdown(net::ip::tcp::socket::shutdown_send, shutdown_ec);

        // Schedule delayed reconnection.
        net::steady_timer timer(ioc_, std::chrono::steady_clock::now() + cfg_.reconnect_delay);
        timer.async_wait([this](beast::error_code) {
            if (running_.load(std::memory_order_acquire)) {
                do_connect();
            }
        });
    }
};

}  // namespace phase7
