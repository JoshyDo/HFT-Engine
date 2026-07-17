// =============================================================================
// execution_engine.hpp - Zero-Allocation Order Execution & Routing (Phase 9.1)
// =============================================================================
#pragma once

#include <array>
#include <atomic>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "csr_graph.hpp"
#include "device_graph.hpp"

namespace phase7 {

class ExecutionEngine {
private:
    boost::asio::io_context                                             ioc;
    boost::asio::ssl::context                                           ctx{boost::asio::ssl::context::tlsv12_client};
    std::unique_ptr<boost::beast::ssl_stream<boost::beast::tcp_stream>> stream;
    bool                                                                connected = false;

    // Zero-allocation buffer for tracing the cycle (e.g. max 32 edges in a crypto cycle)
    std::array<int64_t, 32> cycle_buffer;
    int                     cycle_length = 0;

    // Pre-formatted JSON order template.
    // Format: {"symbol":"           ","side":"    ","type":"MARKET","quantity":"          "}
    char order_payload[3][128];  // One buffer per parallel worker

    // Thread Pool for Parallel Dispatch
    std::vector<std::thread> workers;
    std::atomic<bool>        stop_workers{false};
    std::atomic<int>         active_tasks{0};

    // Synchronization
    std::mutex              mtx;
    std::condition_variable cv;
    int                     pending_tasks = 0;

    // Cycle Data for Workers
    int64_t edges_to_dispatch[3];  // Max 3 hops in parallel

    // Reference to graph topology to resolve edge IDs to nodes/symbols
    const CSRGraph& graph;

    void connect(const std::string& host, const std::string& port);
    void format_and_dispatch();
    void worker_thread(int worker_id);

public:
    ExecutionEngine(const CSRGraph& g, const std::string& host, const std::string& port);
    ~ExecutionEngine();

    // Runs extraction and triggers execution if a cycle is found
    void execute_cycles(const ArbitrageOpportunity* opp);
};

}  // namespace phase7
