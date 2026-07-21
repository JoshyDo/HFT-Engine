// =============================================================================
// udp_multicast_transport.hpp - Native UDP Multicast Transport (kqueue/epoll)
// =============================================================================
#pragma once

#include <memory>
#include <string_view>
#include <system_error>
#include <vector>
#include <iostream>
#include <thread>
#include <atomic>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

#if defined(__linux__)
#include <sys/epoll.h>
#elif defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/event.h>
#endif

#include "../feed_client.hpp"
#include "../thread_pinning.hpp"

namespace phase7 {

template <typename ParserPolicy>
class UdpMulticastTransport : public TransportBase<UdpMulticastTransport<ParserPolicy>, ParserPolicy> {
public:
    using Base = TransportBase<UdpMulticastTransport<ParserPolicy>, ParserPolicy>;
    using Config = typename Base::Config;
    using RingBuffer = typename Base::RingBuffer;

    UdpMulticastTransport(Config cfg, std::shared_ptr<RingBuffer> ringbuffer)
        : Base(std::move(cfg), std::move(ringbuffer)), fd_(-1), epfd_(-1) {
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

        const int core = this->cfg_.io_core_id;
        io_thread_ = std::thread([this, core]() {
            if (core >= 0) {
                if (!::phase7::pin_current_thread_to_core(core)) {
                    std::fprintf(stderr, "[UdpTransport] WARNING: pinning to core %d failed\n", core);
                }
            }
            event_loop();
        });
    }

    void do_stop() {
        if (!this->running_.exchange(false)) return;
        
        // Break the event loop by closing the multiplexer and socket
        if (epfd_ >= 0) {
            close(epfd_);
            epfd_ = -1;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }
    
private:
    int fd_;
    int epfd_;
    std::thread io_thread_;

    void setup_socket() {
        fd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "socket");

        int reuse = 1;
        if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
            throw std::system_error(errno, std::generic_category(), "setsockopt SO_REUSEADDR");
        }

#ifdef SO_REUSEPORT
        if (setsockopt(fd_, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse)) < 0) {
            // Non-fatal
        }
#endif

        // Maximize receive buffer (e.g., 16 MB) to prevent packet loss
        int rcvbuf = 16 * 1024 * 1024;
        setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

        // Non-blocking
        int flags = fcntl(fd_, F_GETFL, 0);
        fcntl(fd_, F_SETFL, flags | O_NONBLOCK);

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(std::stoi(this->cfg_.port));
        addr.sin_addr.s_addr = htonl(INADDR_ANY);

        if (bind(fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            throw std::system_error(errno, std::generic_category(), "bind");
        }

        // Join multicast group
        struct ip_mreq mreq{};
        mreq.imr_multiaddr.s_addr = inet_addr(this->cfg_.host.c_str());
        mreq.imr_interface.s_addr = htonl(INADDR_ANY);
        if (setsockopt(fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
            // If it's not a multicast IP, this will fail. We just ignore for generic UDP usage.
            // In a real environment, you might want to log this.
        }

#if defined(__linux__)
        epfd_ = epoll_create1(0);
        if (epfd_ < 0) throw std::system_error(errno, std::generic_category(), "epoll_create1");

        struct epoll_event ev{};
        ev.events = EPOLLIN | EPOLLET; // Edge-triggered
        ev.data.fd = fd_;
        if (epoll_ctl(epfd_, EPOLL_CTL_ADD, fd_, &ev) < 0) {
            throw std::system_error(errno, std::generic_category(), "epoll_ctl");
        }
#elif defined(__APPLE__) || defined(__FreeBSD__)
        epfd_ = kqueue();
        if (epfd_ < 0) throw std::system_error(errno, std::generic_category(), "kqueue");

        struct kevent kev;
        EV_SET(&kev, fd_, EVFILT_READ, EV_ADD | EV_CLEAR, 0, 0, NULL);
        if (kevent(epfd_, &kev, 1, NULL, 0, NULL) < 0) {
            throw std::system_error(errno, std::generic_category(), "kevent add");
        }
#else
        throw std::runtime_error("Unsupported OS for native UDP transport");
#endif
    }

    void event_loop() {
        // Stack buffer for receiving packets (Max UDP payload ~65k)
        char buffer[65536];

#if defined(__linux__)
        struct epoll_event events[16];
        while (this->running_.load(std::memory_order_acquire)) {
            int n = epoll_wait(epfd_, events, 16, 100);
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            
            for (int i = 0; i < n; ++i) {
                if (events[i].data.fd == fd_) {
                    drain_socket(buffer, sizeof(buffer));
                }
            }
        }
#elif defined(__APPLE__) || defined(__FreeBSD__)
        struct kevent events[16];
        struct timespec timeout = {0, 100000000}; // 100ms
        while (this->running_.load(std::memory_order_acquire)) {
            int n = kevent(epfd_, NULL, 0, events, 16, &timeout);
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            
            for (int i = 0; i < n; ++i) {
                if (events[i].ident == static_cast<uintptr_t>(fd_)) {
                    drain_socket(buffer, sizeof(buffer));
                }
            }
        }
#endif
    }

    void drain_socket(char* buffer, size_t max_len) {
        while (true) {
            ssize_t bytes = recvfrom(fd_, buffer, max_len, 0, nullptr, nullptr);
            if (bytes < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break; // Fully drained
                }
                break;
            }
            if (bytes > 0) {
                std::string_view raw(buffer, bytes);
                this->on_message(raw);
            }
        }
    }
};

} // namespace phase7
