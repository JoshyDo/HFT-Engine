// =============================================================================
// tcp_transport.hpp - Asynchronous Boost.Asio C++20 Coroutine TCP Transport
// =============================================================================
#pragma once

#include <boost/asio.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <memory>
#include <string_view>
#include <thread>
#include <atomic>
#include <chrono>

#include "../feed_client.hpp"
#include "../thread_pinning.hpp"

namespace phase7 {

namespace net = boost::asio;

template <typename ParserPolicy>
class TcpTransport : public TransportBase<TcpTransport<ParserPolicy>, ParserPolicy> {
public:
    using Base = TransportBase<TcpTransport<ParserPolicy>, ParserPolicy>;
    using Config = typename Base::Config;
    using RingBuffer = typename Base::RingBuffer;

    TcpTransport(Config cfg, std::shared_ptr<RingBuffer> ringbuffer, std::shared_ptr<wal::WalWriter::RingBuffer> wal_ringbuffer = nullptr)
        : Base(std::move(cfg), std::move(ringbuffer), std::move(wal_ringbuffer)),
          socket_(ioc_) {
    }

    ~TcpTransport() {
        do_stop();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }

    TcpTransport(const TcpTransport&) = delete;
    TcpTransport& operator=(const TcpTransport&) = delete;

    void do_run() {
        bool expected = false;
        if (!this->running_.compare_exchange_strong(expected, true)) {
            return;
        }

        net::co_spawn(ioc_, connect_and_read_loop(), net::detached);

        const int core = this->cfg_.io_core_id;
        io_thread_ = std::thread([this, core]() {
            if (core >= 0) {
                if (!::phase7::pin_current_thread_to_core(core)) {
                    std::fprintf(stderr, "[TcpTransport] WARNING: pinning to core %d failed\n", core);
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
    net::ip::tcp::socket socket_;
    std::thread io_thread_;

    net::awaitable<void> connect_and_read_loop() {
        auto executor = co_await net::this_coro::executor;
        net::ip::tcp::resolver resolver(executor);

        while (this->running_.load(std::memory_order_acquire)) {
            boost::system::error_code ec;

            // Resolve
            auto results = co_await resolver.async_resolve(this->cfg_.host, this->cfg_.port, net::redirect_error(net::use_awaitable, ec));
            if (ec) {
                co_await handle_reconnect(ec);
                continue;
            }

            // Connect
            co_await net::async_connect(socket_, results, net::redirect_error(net::use_awaitable, ec));
            if (ec) {
                co_await handle_reconnect(ec);
                continue;
            }

            this->reconnect_attempts_ = 0;

            // Read Loop
            char buffer[65536];
            while (this->running_.load(std::memory_order_acquire)) {
                std::size_t n = co_await socket_.async_read_some(net::buffer(buffer), net::redirect_error(net::use_awaitable, ec));
                
                if (ec) {
                    if (ec == net::error::eof || ec == net::error::connection_reset) {
                        break; // Will reconnect
                    }
                    break; // Other error, also reconnect
                }

                if (n > 0) {
                    std::string_view raw(buffer, n);
                    this->on_message(raw); // Base CRTP parses & queues
                }
            }

            if (this->running_.load(std::memory_order_acquire)) {
                co_await handle_reconnect(ec);
            }
        }
    }

    net::awaitable<void> handle_reconnect(const boost::system::error_code& /*ec*/) {
        if (!this->running_.load(std::memory_order_acquire)) {
            co_return;
        }

        if (this->cfg_.max_reconnect_attempts != 0 && this->reconnect_attempts_ >= this->cfg_.max_reconnect_attempts) {
            this->running_.store(false, std::memory_order_release);
            ioc_.stop();
            co_return;
        }
        
        ++this->reconnect_attempts_;
        
        boost::system::error_code ignored_ec;
        socket_.close(ignored_ec);

        net::steady_timer timer(co_await net::this_coro::executor, this->cfg_.reconnect_delay);
        boost::system::error_code timer_ec;
        co_await timer.async_wait(net::redirect_error(net::use_awaitable, timer_ec));
    }
};

} // namespace phase7
