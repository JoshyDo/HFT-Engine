// =============================================================================
// tcp_transport.hpp - Stub for TCP Transport
// =============================================================================
#pragma once

#include <memory>
#include <string_view>
#include "../feed_client.hpp"

namespace phase7 {

template <typename ParserPolicy>
class TcpTransport : public TransportBase<TcpTransport<ParserPolicy>, ParserPolicy> {
public:
    using Base = TransportBase<TcpTransport<ParserPolicy>, ParserPolicy>;
    using Config = typename Base::Config;
    using RingBuffer = typename Base::RingBuffer;

    TcpTransport(Config cfg, std::shared_ptr<RingBuffer> ringbuffer)
        : Base(std::move(cfg), std::move(ringbuffer)) {
    }

    void do_run() {
        // Implement TCP socket connect and async_read loop here
    }

    void do_stop() {
        // Implement graceful shutdown
    }

private:
    void on_read(std::string_view raw) {
        // Call base class method
        this->on_message(raw);
    }
};

} // namespace phase7
