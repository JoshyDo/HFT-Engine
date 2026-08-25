#include <boost/asio.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <thread>
#include <cstring>
#include <liburing.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netinet/ip.h>

#ifndef IORING_SETUP_COOP_TASKRUN
#define IORING_SETUP_COOP_TASKRUN (1U << 8)
#endif
#ifndef IORING_SETUP_SINGLE_ISSUER
#define IORING_SETUP_SINGLE_ISSUER (1U << 12)
#endif
#ifndef IORING_SETUP_DEFER_TASKRUN
#define IORING_SETUP_DEFER_TASKRUN (1U << 13)
#endif

#include "../include/parsers.hpp"
#include "../include/thread_pinning.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
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

constexpr int kIterations = 100'000;
constexpr int kExchangeCore = 2;
constexpr int kEngineCore = 4;
constexpr short kPort = 12347;

std::vector<uint64_t> core_logic_cycles;
std::vector<uint64_t> t2t_cycles;

void setup_io_uring(struct io_uring *ring, int sq_core) {
    struct io_uring_params params;
    memset(&params, 0, sizeof(params));
    params.flags = IORING_SETUP_SQPOLL | IORING_SETUP_SQ_AFF | IORING_SETUP_SINGLE_ISSUER;
    params.sq_thread_idle = 2000; // milliseconds
    params.sq_thread_cpu = sq_core;

    int ret = io_uring_queue_init_params(128, ring, &params);
    if (ret < 0) {
        std::cerr << "io_uring_queue_init_params failed: " << strerror(-ret) << "\n";
        exit(1);
    }
}

void MockExchange() {
    phase7::pin_current_thread_to_core(kExchangeCore);
    std::cout << "[MockExchange] Started and pinned to core " << kExchangeCore << ".\n";

    net::io_context ioc;
    net::ip::tcp::endpoint endpoint(net::ip::tcp::v4(), kPort);
    net::ip::tcp::acceptor acceptor(ioc);
    acceptor.open(endpoint.protocol());
    acceptor.set_option(net::socket_base::reuse_address(true));
    typedef net::detail::socket_option::boolean<SOL_SOCKET, SO_REUSEPORT> reuse_port;
    acceptor.set_option(reuse_port(true));
    acceptor.bind(endpoint);
    acceptor.listen();
    
    net::ip::tcp::socket socket(ioc);
    std::cout << "[MockExchange] Waiting for Engine...\n";
    acceptor.accept(socket);
    std::cout << "[MockExchange] Engine connected.\n";
    
    int fd = socket.native_handle();
    socket.set_option(net::ip::tcp::no_delay(true));

    int tos = IPTOS_LOWDELAY;
    setsockopt(fd, IPPROTO_IP, IP_TOS, &tos, sizeof(tos));
    int quickack = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &quickack, sizeof(quickack));
    int usecs = 50;
    setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &usecs, sizeof(usecs));

    struct io_uring ring;
    setup_io_uring(&ring, 3); // Core 3 for SQPOLL

    phase7::SbeMessageHeader header{sizeof(phase7::SbeTickPayload), 42, 1, 1};
    phase7::SbeTickPayload payload{"BTCUSDT", 50000.0, 50001.0};
    
    char send_buf[sizeof(header) + sizeof(payload)];
    std::memcpy(send_buf, &header, sizeof(header));
    std::memcpy(send_buf + sizeof(header), &payload, sizeof(payload));
    
    DummyOrder response;
    
    std::cout << "[MockExchange] Starting loop...\n";
    for (int i = 0; i < kIterations; ++i) {
        // Send
        if (i < 2) std::cout << "[MockExchange] Iter " << i << ": Submitting Send...\n";
        struct io_uring_sqe *sqe_send = io_uring_get_sqe(&ring);
        io_uring_prep_send(sqe_send, fd, send_buf, sizeof(send_buf), MSG_NOSIGNAL);
        io_uring_submit(&ring);
        
        struct io_uring_cqe *cqe;
        if (i < 2) std::cout << "[MockExchange] Iter " << i << ": Polling CQE for Send...\n";
        while (io_uring_peek_cqe(&ring, &cqe) == -EAGAIN) {
            _mm_pause();
        }
        if (cqe->res < 0) { std::cerr << "Exchange Send Error: " << strerror(-cqe->res) << "\n"; exit(1); }
        io_uring_cqe_seen(&ring, cqe);
        if (i < 2) std::cout << "[MockExchange] Iter " << i << ": Send complete.\n";

        // Recv
        if (i < 2) std::cout << "[MockExchange] Iter " << i << ": Submitting Recv...\n";
        struct io_uring_sqe *sqe_recv = io_uring_get_sqe(&ring);
        io_uring_prep_recv(sqe_recv, fd, &response, sizeof(response), 0);
        io_uring_submit(&ring);
        
        if (i < 2) std::cout << "[MockExchange] Iter " << i << ": Polling CQE for Recv...\n";
        while (io_uring_peek_cqe(&ring, &cqe) == -EAGAIN) {
            _mm_pause();
        }
        if (cqe->res <= 0) { std::cerr << "Exchange Recv Error/Closed: " << cqe->res << "\n"; exit(1); }
        io_uring_cqe_seen(&ring, cqe);
        if (i < 2) std::cout << "[MockExchange] Iter " << i << ": Recv complete.\n";
        
        int qa = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &qa, sizeof(qa));
    }
}

void loopback_engine(net::io_context& ioc) {
    phase7::pin_current_thread_to_core(kEngineCore);
    std::cout << "[Engine] Started and pinned to core " << kEngineCore << ".\n";

    net::ip::tcp::socket socket(ioc);
    socket.connect(net::ip::tcp::endpoint(net::ip::address::from_string("127.0.0.1"), kPort));
    std::cout << "[Engine] Connected to Exchange.\n";
    
    int fd = socket.native_handle();
    socket.set_option(net::ip::tcp::no_delay(true));

    int tos = IPTOS_LOWDELAY;
    setsockopt(fd, IPPROTO_IP, IP_TOS, &tos, sizeof(tos));
    int quickack = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &quickack, sizeof(quickack));
    int usecs = 50;
    setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &usecs, sizeof(usecs));

    struct io_uring ring;
    setup_io_uring(&ring, 5); // Core 5 for SQPOLL

    phase7::SbeParser parser;
    double best_bid = 0;
    [[maybe_unused]] double best_ask = 0;

    char read_buf[64];
    DummyOrder outbound_order;
    
    core_logic_cycles.reserve(kIterations);
    t2t_cycles.reserve(kIterations);
    
    size_t expected_read_size = sizeof(phase7::SbeMessageHeader) + sizeof(phase7::SbeTickPayload);

    std::cout << "[Engine] Starting loop...\n";
    for (int i = 0; i < kIterations; ++i) {
        // Recv
        if (i < 2) std::cout << "[Engine] Iter " << i << ": Submitting Recv...\n";
        struct io_uring_sqe *sqe_recv = io_uring_get_sqe(&ring);
        io_uring_prep_recv(sqe_recv, fd, read_buf, expected_read_size, 0);
        io_uring_submit(&ring);
        
        struct io_uring_cqe *cqe;
        if (i < 2) std::cout << "[Engine] Iter " << i << ": Polling CQE for Recv...\n";
        while (io_uring_peek_cqe(&ring, &cqe) == -EAGAIN) {
            _mm_pause();
        }
        if (cqe->res <= 0) { std::cerr << "Engine Recv Error/Closed: " << cqe->res << "\n"; exit(1); }
        io_uring_cqe_seen(&ring, cqe);
        if (i < 2) std::cout << "[Engine] Iter " << i << ": Recv complete.\n";
        
        uint64_t t0 = __rdtsc();
        
        MarketTick tick;
        std::string_view sym;
        parser.parse(std::string_view(read_buf, sizeof(read_buf)), tick, sym);
        
        best_bid = tick.bid;
        best_ask = tick.ask;
        
        std::memcpy(outbound_order.symbol, sym.data(), std::min<size_t>(8, sym.size()));
        outbound_order.price = best_bid;
        
        uint64_t t1 = __rdtsc();
        
        // Send
        if (i < 2) std::cout << "[Engine] Iter " << i << ": Submitting Send...\n";
        struct io_uring_sqe *sqe_send = io_uring_get_sqe(&ring);
        io_uring_prep_send(sqe_send, fd, &outbound_order, sizeof(outbound_order), MSG_NOSIGNAL);
        io_uring_submit(&ring);
        
        if (i < 2) std::cout << "[Engine] Iter " << i << ": Polling CQE for Send...\n";
        while (io_uring_peek_cqe(&ring, &cqe) == -EAGAIN) {
            _mm_pause();
        }
        if (cqe->res < 0) { std::cerr << "Engine Send Error: " << strerror(-cqe->res) << "\n"; exit(1); }
        io_uring_cqe_seen(&ring, cqe);
        if (i < 2) std::cout << "[Engine] Iter " << i << ": Send complete.\n";
        
        uint64_t t2 = __rdtsc();
        
        if (i > 1000) {
            core_logic_cycles.push_back(t1 - t0);
            t2t_cycles.push_back(t2 - t0);
        }

        int qa = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &qa, sizeof(qa));
    }
}

int main() {
    std::cout << std::unitbuf;
    std::cout << "[Benchmark] TCP Loopback Sub-10us Profiling (io_uring SQPOLL)\n";
    
    std::thread exchange_thread(MockExchange);
    
    std::thread engine_thread([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        net::io_context ioc;
        loopback_engine(ioc);
    });
    
    exchange_thread.join();
    engine_thread.join();
    
    auto print_metrics = [](const std::string& label, std::vector<uint64_t>& cycles) {
        std::sort(cycles.begin(), cycles.end());
        uint64_t median = cycles[cycles.size() / 2];
        uint64_t p99 = cycles[cycles.size() * 99 / 100];
        
        double us_estimate_median = median / 3000.0;
        double us_estimate_p99 = p99 / 3000.0;
        
        std::cout << "--- " << label << " ---\n";
        std::cout << "Median: " << median << " CPU Cycles (~" << us_estimate_median << " us)\n";
        std::cout << "99th %: " << p99 << " CPU Cycles (~" << us_estimate_p99 << " us)\n\n";
    };
    
    std::cout << "\nResults (" << core_logic_cycles.size() << " samples after warmup):\n";
    print_metrics("Core Logic (Parse -> Book -> Order)", core_logic_cycles);
    print_metrics("Software T2T (Core Logic + SQPOLL submit)", t2t_cycles);
    
    return 0;
}
