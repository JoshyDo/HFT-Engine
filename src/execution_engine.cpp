// =============================================================================
// execution_engine.cpp - Zero-Allocation Order Execution & Routing (Phase 9.1)
// =============================================================================

#include "execution_engine.hpp"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#ifndef _MM_PAUSE_DEFINED_HFT
#define _MM_PAUSE_DEFINED_HFT
static inline void _mm_pause() {
    __asm__ __volatile__("yield" ::: "memory");
}
#endif
#else
#ifndef _MM_PAUSE_DEFINED_HFT
#define _MM_PAUSE_DEFINED_HFT
static inline void _mm_pause() {}
#endif
#endif

#include <algorithm>
#include <charconv>
#include <cstring>
#include <iostream>

namespace phase7 {

ExecutionEngine::ExecutionEngine(const CSRGraph& g, const std::string& host, const std::string& port) : graph(g) {
    // Initialize the JSON template for all 3 workers
    const char* tmpl =
        "{\"symbol\":\"           \",\"side\":\"    \",\"type\":\"MARKET\",\"quantity\":\"          \"}\n";
    for (int i = 0; i < 3; ++i) {
        std::memcpy(order_payload[i], tmpl, std::strlen(tmpl) + 1);
    }

    // Start 3 worker threads
    for (int i = 0; i < 3; ++i) {
        workers.emplace_back(&ExecutionEngine::worker_thread, this, i);
    }

    try {
        connect(host, port);
    } catch (const std::exception& e) {
        std::cerr << "[ExecutionEngine] Connection failed: " << e.what() << "\n";
    }
}

ExecutionEngine::~ExecutionEngine() {
    stop_workers = true;
    cv.notify_all();
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    if (connected && stream) {
        boost::system::error_code ec;
        stream->shutdown(ec);
    }
}

void ExecutionEngine::connect(const std::string& host, const std::string& port) {
    // Basic DNS resolution and TLS connect setup
    boost::asio::ip::tcp::resolver resolver(ioc);
    auto const                     results = resolver.resolve(host, port);

    stream = std::make_unique<boost::beast::ssl_stream<boost::beast::tcp_stream>>(ioc, ctx);

    // Set SNI Hostname (many hosts need this to handshake successfully)
    if (!SSL_set_tlsext_host_name(stream->native_handle(), host.c_str())) {
        boost::system::error_code ec{static_cast<int>(::ERR_get_error()), boost::asio::error::get_ssl_category()};
        throw boost::system::system_error{ec};
    }

    boost::beast::get_lowest_layer(*stream).connect(results);
    stream->handshake(boost::asio::ssl::stream_base::client);
    connected = true;
    std::cout << "[ExecutionEngine] Connected to " << host << ":" << port << "\n";
}

void ExecutionEngine::execute_cycles(const ArbitrageOpportunity* opp) {
    if (opp->path_length > 0) {
        cycle_length = std::min(opp->path_length, 10);
        for (int i = 0; i < cycle_length; ++i) {
            cycle_buffer[i] = opp->edge_indices[i];
        }
        format_and_dispatch();
    }
}

namespace {
// Helper to statically map node IDs to 11-byte space-padded Binance symbols
// 0: USDT, 1: BTC, 2: ETH, 3: SOL, 4: XRP, 5: DOGE
const char* get_symbol(int u, int v) {
    int min_val = std::min(u, v);
    int max_val = std::max(u, v);

    if (min_val == 0) {
        if (max_val == 1) return "BTCUSDT    ";
        if (max_val == 2) return "ETHUSDT    ";
        if (max_val == 3) return "SOLUSDT    ";
        if (max_val == 4) return "XRPUSDT    ";
        if (max_val == 5) return "DOGEUSDT   ";
    } else if (min_val == 1) {
        if (max_val == 2) return "ETHBTC     ";
        if (max_val == 3) return "SOLBTC     ";
        if (max_val == 4) return "XRPBTC     ";
        if (max_val == 5) return "DOGEBTC    ";
    }
    return "UNKNOWN    ";
}
}  // namespace

void ExecutionEngine::worker_thread(int worker_id) {
    while (!stop_workers) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this, worker_id] { return stop_workers || pending_tasks > worker_id; });

        if (stop_workers) break;

        int64_t e = edges_to_dispatch[worker_id];
        lock.unlock();

        // 1. Find source and target node
        auto it = std::upper_bound(graph.row_offsets.begin(), graph.row_offsets.end(), e);
        int  u  = static_cast<int>(std::distance(graph.row_offsets.begin(), it) - 1);
        int  v  = graph.col_indices[e];

        // 2. Generate symbol representation via static mapping (11 bytes padding)
        const char* symbol_str = get_symbol(u, v);

        // 3. Pointer-arithmetic Payload Formatting
        char* payload = order_payload[worker_id];

        std::memcpy(payload + 11, symbol_str, 11);

        const char* side = (u < v) ? "BUY " : "SELL";
        std::memcpy(payload + 32, side, 4);

        const char* qty = "0.0010    ";
        std::memcpy(payload + 66, qty, 10);

        // 4. Dispatch via TCP/TLS
        // In real execution, each thread would have its own SSL stream.
        // For Phase 12 demonstration, we print concurrently.
        std::cout << "[Parallel Worker " << worker_id << "] Dispatching Order: " << payload;

        if (connected && worker_id == 0) {
            // Only worker 0 writes to the single shared TLS stream in this demo
            // to avoid TLS framing corruption, since we don't have 3 sockets yet.
            boost::asio::write(*stream, boost::asio::buffer(payload, std::strlen(payload)));
        }

        active_tasks--;
    }
}

void ExecutionEngine::format_and_dispatch() {
    // Phase 12: Parallel Dispatch
    // We expect 3 edges in a standard crypto triangular arbitrage cycle
    int num_to_dispatch = std::min(cycle_length, 3);

    for (int i = 0; i < num_to_dispatch; ++i) {
        // Reverse order is not strictly necessary anymore since they are dispatched in parallel,
        // but we'll feed them in order of execution.
        edges_to_dispatch[i] = cycle_buffer[cycle_length - 1 - i];
    }

    active_tasks = num_to_dispatch;
    {
        std::lock_guard<std::mutex> lock(mtx);
        pending_tasks = num_to_dispatch;
    }
    cv.notify_all();

    // Busy-wait for workers to finish formatting and dispatching
    while (active_tasks > 0) {
        _mm_pause();
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        pending_tasks = 0;
    }
}

}  // namespace phase7
