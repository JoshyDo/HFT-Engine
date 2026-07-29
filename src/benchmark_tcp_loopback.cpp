#include <boost/asio.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/use_awaitable.hpp>

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <thread>
#include <cstring>

#include "../include/parsers.hpp"
#include "../include/thread_pinning.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#else
#include <chrono>
static inline uint64_t __rdtsc() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
#endif

namespace net = boost::asio;

struct alignas(16) DummyOrder {
    char symbol[8];
    double price;
};

// Benchmark Configuration
constexpr int kIterations = 100'000;
constexpr int kExchangeCore = 2; // distinct from EngineCore
constexpr int kEngineCore = 4;   // distinct from ExchangeCore
constexpr short kPort = 9999;

// Dual Metrics
std::vector<uint64_t> core_logic_cycles;
std::vector<uint64_t> t2t_cycles;

void MockExchange() {
    phase7::pin_current_thread_to_core(kExchangeCore);

    net::io_context ioc;
    net::ip::tcp::acceptor acceptor(ioc, net::ip::tcp::endpoint(net::ip::tcp::v4(), kPort));
    
    net::ip::tcp::socket socket(ioc);
    acceptor.accept(socket);
    
    // Disable Nagle's algorithm for low latency
    socket.set_option(net::ip::tcp::no_delay(true));

    phase7::SbeMessageHeader header{sizeof(phase7::SbeTickPayload), 42, 1, 1};
    phase7::SbeTickPayload payload{"BTCUSDT", 50000.0, 50001.0};
    
    char send_buf[sizeof(header) + sizeof(payload)];
    std::memcpy(send_buf, &header, sizeof(header));
    std::memcpy(send_buf + sizeof(header), &payload, sizeof(payload));
    
    DummyOrder response;
    
    for (int i = 0; i < kIterations; ++i) {
        // Publish SBE Tick
        net::write(socket, net::buffer(send_buf, sizeof(send_buf)));
        
        // Synchronously await order dispatch from Engine
        boost::system::error_code ec;
        size_t n = net::read(socket, net::buffer(&response, sizeof(response)), ec);
        if (ec || n != sizeof(response)) break;
    }
}

net::awaitable<void> loopback_engine(net::io_context& ioc) {
    net::ip::tcp::socket socket(ioc);
    
    // Connect
    co_await socket.async_connect(net::ip::tcp::endpoint(net::ip::address::from_string("127.0.0.1"), kPort), net::use_awaitable);
    socket.set_option(net::ip::tcp::no_delay(true));

    phase7::SbeParser parser;
    double best_bid = 0;
    [[maybe_unused]] double best_ask = 0;

    char read_buf[64];
    DummyOrder outbound_order;
    
    core_logic_cycles.reserve(kIterations);
    t2t_cycles.reserve(kIterations);

    for (int i = 0; i < kIterations; ++i) {
        // 1. Await Market Data
        co_await net::async_read(socket, net::buffer(read_buf, sizeof(phase7::SbeMessageHeader) + sizeof(phase7::SbeTickPayload)), net::use_awaitable);
        
        // --- START PROFILING ---
        uint64_t t0 = __rdtsc();
        
        MarketTick tick;
        std::string_view sym;
        parser.parse(std::string_view(read_buf, sizeof(read_buf)), tick, sym);
        
        // 2. Update L2 Book
        best_bid = tick.bid;
        best_ask = tick.ask;
        
        // 3. Format Order
        std::memcpy(outbound_order.symbol, sym.data(), std::min<size_t>(8, sym.size()));
        outbound_order.price = best_bid; // aggressive hit
        
        uint64_t t1 = __rdtsc();
        
        // 4. Dispatch Order
        co_await net::async_write(socket, net::buffer(&outbound_order, sizeof(outbound_order)), net::use_awaitable);
        
        uint64_t t2 = __rdtsc();
        // --- END PROFILING ---
        
        // Warmup exclusion (first 1000)
        if (i > 1000) {
            core_logic_cycles.push_back(t1 - t0);
            t2t_cycles.push_back(t2 - t0);
        }
    }
    
    ioc.stop();
}

int main() {
    std::cout << "[Benchmark] TCP Loopback Sub-10us Profiling\n";
    std::cout << "[Benchmark] Pinning Exchange->Core " << kExchangeCore << " and Engine->Core " << kEngineCore << "\n";
    
    // Spin up Publisher
    std::thread exchange_thread(MockExchange);
    
    // Spin up Engine
    std::thread engine_thread([]() {
        phase7::pin_current_thread_to_core(kEngineCore);
        net::io_context ioc;
        net::co_spawn(ioc, loopback_engine(ioc), net::detached);
        ioc.run();
    });
    
    exchange_thread.join();
    engine_thread.join();
    
    auto print_metrics = [](const std::string& label, std::vector<uint64_t>& cycles) {
        std::sort(cycles.begin(), cycles.end());
        uint64_t median = cycles[cycles.size() / 2];
        uint64_t p99 = cycles[cycles.size() * 99 / 100];
        
        // Roughly assume 3-5 GHz for display
        double us_estimate_median = median / 3000.0;
        double us_estimate_p99 = p99 / 3000.0;
        
        std::cout << "--- " << label << " ---\n";
        std::cout << "Median: " << median << " CPU Cycles (~" << us_estimate_median << " us)\n";
        std::cout << "99th %: " << p99 << " CPU Cycles (~" << us_estimate_p99 << " us)\n\n";
    };
    
    std::cout << "\nResults (" << core_logic_cycles.size() << " samples after warmup):\n";
    print_metrics("Core Logic (Parse -> Book -> Order)", core_logic_cycles);
    print_metrics("Software T2T (Core Logic + async_write dispatch)", t2t_cycles);
    
    return 0;
}
