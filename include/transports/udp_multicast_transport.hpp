// =============================================================================
// udp_multicast_transport.hpp - Asynchronous Boost.Asio C++20 Coroutine UDP
// =============================================================================
#pragma once

#include <boost/asio.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/ip/udp.hpp>
#include <boost/asio/ip/multicast.hpp>

#include <memory>
#include <string_view>
#include <thread>
#include <atomic>
#include <iostream>
#include <system_error>

#include "../feed_client.hpp"
#include "../thread_pinning.hpp"

namespace phase7 {

namespace net = boost::asio;

template <typename ParserPolicy>
class UdpMulticastTransport : public TransportBase<UdpMulticastTransport<ParserPolicy>, ParserPolicy> {
public:
    using Base = TransportBase<UdpMulticastTransport<ParserPolicy>, ParserPolicy>;
    using Config = typename Base::Config;
    using RingBuffer = typename Base::RingBuffer;

    UdpMulticastTransport(Config cfg, std::shared_ptr<RingBuffer> ringbuffer, std::shared_ptr<wal::WalWriter::RingBuffer> wal_ringbuffer = nullptr)
        : Base(std::move(cfg), std::move(ringbuffer), std::move(wal_ringbuffer)), socket_(ioc_) {
    }

    ~UdpMulticastTransport() {
        do_stop();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }

    void do_run() {
        bool expected = false;
        if (!this->running_.compare_exchange_strong(expected, true)) {
            return;
        }

        setup_socket();

        net::co_spawn(ioc_, read_loop(), net::detached);

        const int core = this->cfg_.io_core_id;
        io_thread_ = std::thread([this, core]() {
            if (core >= 0) {
                if (!::phase7::pin_current_thread_to_core(core)) {
                    std::fprintf(stderr, "[UdpTransport] WARNING: pinning to core %d failed\n", core);
                }
            }
            ioc_.run();
        });
    }

    void do_stop() {
        if (!this->running_.exchange(false)) return;
        
        net::post(ioc_, [this]() {
            boost::system::error_code ec;
            socket_.close(ec);
            ioc_.stop();
        });
    }
    
private:
    net::io_context ioc_;
    net::ip::udp::socket socket_;
    std::thread io_thread_;

    void setup_socket() {
        boost::system::error_code ec;
        
        // Open UDP socket
        socket_.open(net::ip::udp::v4(), ec);
        if (ec) throw std::system_error(ec, "open");

        // Reuse address
        socket_.set_option(net::socket_base::reuse_address(true), ec);
        if (ec) throw std::system_error(ec, "reuse_address");

        // Maximize receive buffer (e.g., 16 MB)
        socket_.set_option(net::socket_base::receive_buffer_size(16 * 1024 * 1024), ec);

        // Bind
        net::ip::udp::endpoint listen_endpoint(net::ip::address_v4::any(), std::stoi(this->cfg_.port));
        socket_.bind(listen_endpoint, ec);
        if (ec) throw std::system_error(ec, "bind");

        // Join multicast group
        auto multicast_address = net::ip::make_address(this->cfg_.host, ec);
        if (!ec && multicast_address.is_multicast()) {
            socket_.set_option(net::ip::multicast::join_group(multicast_address.to_v4()), ec);
            if (ec) {
                std::fprintf(stderr, "[UdpTransport] WARNING: Failed to join multicast group %s\n", this->cfg_.host.c_str());
            }
        }
    }

    net::awaitable<void> read_loop() {
        char buffer[65536];
        
        while (this->running_.load(std::memory_order_acquire)) {
            boost::system::error_code ec;
            net::ip::udp::endpoint sender_endpoint;
            
            std::size_t n = co_await socket_.async_receive_from(
                net::buffer(buffer), sender_endpoint, net::redirect_error(net::use_awaitable, ec));
                
            if (ec) {
                if (ec == net::error::operation_aborted) break;
                continue; // Ignore UDP read errors and keep listening
            }
            
            if (n > 0) {
                std::string_view raw(buffer, n);
                this->on_message(raw);
            }
        }
    }
};

} // namespace phase7
