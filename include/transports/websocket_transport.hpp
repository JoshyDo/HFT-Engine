// =============================================================================
// websocket_transport.hpp - Asynchronous Boost.Beast WebSocket Client Transport
// =============================================================================
#pragma once

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <thread>
#include <memory>
#include <string>
#include <string_view>

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

    WebSocketTransport(Config cfg, std::shared_ptr<RingBuffer> ringbuffer)
        : Base(std::move(cfg), std::move(ringbuffer)),
          resolver_(net::make_strand(ioc_)),
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
        do_connect();
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
    net::ip::tcp::resolver resolver_;
    websocket::stream<beast::ssl_stream<beast::tcp_stream>> ws_;
    std::thread io_thread_;

    void do_connect() {
        ws_.next_layer().native_handle();
        resolver_.async_resolve(
            this->cfg_.host, this->cfg_.port, beast::bind_front_handler(&WebSocketTransport::on_resolve, this));
    }

    void on_resolve(beast::error_code ec, net::ip::tcp::resolver::results_type results) {
        if (ec) return schedule_reconnect(ec);
        beast::get_lowest_layer(ws_).expires_after(std::chrono::seconds(30));
        beast::get_lowest_layer(ws_).async_connect(results,
                                                   beast::bind_front_handler(&WebSocketTransport::on_connect, this));
    }

    void on_connect(beast::error_code ec, net::ip::tcp::resolver::results_type::endpoint_type) {
        if (ec) return schedule_reconnect(ec);
        ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
        ws_.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req) { req.set(beast::http::field::user_agent, "HFT-Engine/Phase7"); }));

        ws_.next_layer().async_handshake(ssl::stream_base::client,
                                         beast::bind_front_handler(&WebSocketTransport::on_ssl_handshake, this));
    }

    void on_ssl_handshake(beast::error_code ec) {
        if (ec) return schedule_reconnect(ec);
        ws_.async_handshake(
            this->cfg_.host, this->cfg_.target, beast::bind_front_handler(&WebSocketTransport::on_handshake, this));
    }

    void on_handshake(beast::error_code ec) {
        if (ec) return schedule_reconnect(ec);
        this->reconnect_attempts_ = 0;
        do_read();
    }

    void do_read() {
        buffer_.consume(buffer_.size());
        ws_.async_read(buffer_, beast::bind_front_handler(&WebSocketTransport::on_read, this));
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);

        if (ec) {
            if (ec == websocket::error::closed) return;
            return schedule_reconnect(ec);
        }

        std::string_view raw{static_cast<const char*>(buffer_.data().data()), buffer_.data().size()};
        
        // --- DELEGATE TO BASE CLASS VIA CRTP ---
        this->on_message(raw);

        if (this->running_.load(std::memory_order_acquire)) {
            do_read();
        }
    }

    void schedule_reconnect([[maybe_unused]] const beast::error_code& ec) {
        if (!this->running_.load(std::memory_order_acquire)) return;

        if (this->cfg_.max_reconnect_attempts != 0 && this->reconnect_attempts_ >= this->cfg_.max_reconnect_attempts) {
            this->running_.store(false, std::memory_order_release);
            ioc_.stop();
            return;
        }
        ++this->reconnect_attempts_;

        beast::error_code shutdown_ec;
        beast::get_lowest_layer(ws_).socket().shutdown(net::ip::tcp::socket::shutdown_send, shutdown_ec);

        net::steady_timer timer(ioc_, std::chrono::steady_clock::now() + this->cfg_.reconnect_delay);
        timer.async_wait([this](beast::error_code) {
            if (this->running_.load(std::memory_order_acquire)) {
                do_connect();
            }
        });
    }
};

}  // namespace phase7
