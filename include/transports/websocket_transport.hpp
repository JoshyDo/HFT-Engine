// =============================================================================
// websocket_transport.hpp - Asynchronous Boost.Beast WebSocket Coroutine Transport
// =============================================================================
#pragma once

#include <boost/asio.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <thread>
#include <memory>
#include <string>
#include <string_view>
#include <atomic>

#include "../feed_client.hpp"
#include "../thread_pinning.hpp"

namespace phase7 {

namespace beast     = boost::beast;
namespace websocket = beast::websocket;
namespace net       = boost::asio;
namespace ssl       = boost::asio::ssl;

template <typename ParserPolicy>
class WebSocketTransport : public TransportBase<WebSocketTransport<ParserPolicy>, ParserPolicy> {
public:
    using Base = TransportBase<WebSocketTransport<ParserPolicy>, ParserPolicy>;
    using Config = typename Base::Config;
    using RingBuffer = typename Base::RingBuffer;

    WebSocketTransport(Config cfg, std::shared_ptr<RingBuffer> ringbuffer, std::shared_ptr<wal::WalWriter::RingBuffer> wal_ringbuffer = nullptr)
        : Base(std::move(cfg), std::move(ringbuffer), std::move(wal_ringbuffer)),
          ws_(net::make_strand(ioc_), ssl_ctx_) {
    }

    ~WebSocketTransport() {
        do_stop();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }

    WebSocketTransport(const WebSocketTransport&) = delete;
    WebSocketTransport& operator=(const WebSocketTransport&) = delete;

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
                    std::fprintf(stderr, "[WSClient] WARNING: pinning to core %d failed\n", core);
                }
            }
            ioc_.run();
        });
    }

    void do_stop() {
        if (!this->running_.exchange(false)) return;
        net::post(ioc_, [this]() {
            beast::error_code ec;
            ws_.close(websocket::close_code::normal, ec);
            ioc_.stop();
        });
    }

private:
    net::io_context ioc_;
    ssl::context ssl_ctx_{ssl::context::tls_client};
    beast::flat_buffer buffer_;
    websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws_;
    std::thread io_thread_;

    net::awaitable<void> connect_and_read_loop() {
        auto executor = co_await net::this_coro::executor;
        net::ip::tcp::resolver resolver(executor);

        while (this->running_.load(std::memory_order_acquire)) {
            beast::error_code ec;
            
            ws_.next_layer().native_handle();
            
            // Resolve
            auto results = co_await resolver.async_resolve(this->cfg_.host, this->cfg_.port, net::redirect_error(net::use_awaitable, ec));
            if (ec) {
                co_await handle_reconnect(ec);
                continue;
            }

            // Connect
            beast::get_lowest_layer(ws_).expires_after(std::chrono::seconds(30));
            co_await beast::get_lowest_layer(ws_).async_connect(results, net::redirect_error(net::use_awaitable, ec));
            if (ec) {
                co_await handle_reconnect(ec);
                continue;
            }

            // Options
            ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
            ws_.set_option(websocket::stream_base::decorator(
                [](websocket::request_type& req) { req.set(beast::http::field::user_agent, "HFT-Engine/Phase7"); }));

            // SSL Handshake
            co_await ws_.next_layer().async_handshake(ssl::stream_base::client, net::redirect_error(net::use_awaitable, ec));
            if (ec) {
                co_await handle_reconnect(ec);
                continue;
            }

            // WS Handshake
            co_await ws_.async_handshake(this->cfg_.host, this->cfg_.target, net::redirect_error(net::use_awaitable, ec));
            if (ec) {
                co_await handle_reconnect(ec);
                continue;
            }

            this->reconnect_attempts_ = 0;

            // Read Loop
            while (this->running_.load(std::memory_order_acquire)) {
                buffer_.consume(buffer_.size());
                
                co_await ws_.async_read(buffer_, net::redirect_error(net::use_awaitable, ec));
                
                if (ec) {
                    if (ec == websocket::error::closed) {
                        break;
                    }
                    break;
                }

                std::string_view raw{static_cast<const char*>(buffer_.data().data()), buffer_.data().size()};
                this->on_message(raw);
            }

            if (this->running_.load(std::memory_order_acquire)) {
                co_await handle_reconnect(ec);
            }
        }
    }

    net::awaitable<void> handle_reconnect(const beast::error_code& /*ec*/) {
        if (!this->running_.load(std::memory_order_acquire)) co_return;

        if (this->cfg_.max_reconnect_attempts != 0 && this->reconnect_attempts_ >= this->cfg_.max_reconnect_attempts) {
            this->running_.store(false, std::memory_order_release);
            ioc_.stop();
            co_return;
        }
        
        ++this->reconnect_attempts_;

        beast::error_code shutdown_ec;
        beast::get_lowest_layer(ws_).socket().shutdown(net::ip::tcp::socket::shutdown_send, shutdown_ec);

        net::steady_timer timer(co_await net::this_coro::executor, this->cfg_.reconnect_delay);
        beast::error_code timer_ec;
        co_await timer.async_wait(net::redirect_error(net::use_awaitable, timer_ec));
    }
};

}  // namespace phase7
