// =============================================================================
// wal_writer.hpp - Background I/O Thread for Lock-Free Write-Ahead Log
// =============================================================================
#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <memory>
#include <span>

#include "wal_types.hpp"
#include "../spsc_ringbuffer.hpp"

namespace phase7::wal {

class WalWriter {
public:
    using RingBuffer = ::SPSCRingbuffer<WalEvent, 1048576>;

    WalWriter(std::string filepath, std::shared_ptr<RingBuffer> ringbuffer);
    ~WalWriter();

    // Prevent copies
    WalWriter(const WalWriter&) = delete;
    WalWriter& operator=(const WalWriter&) = delete;

    // Starts the background thread pinned to the specified core
    void start(int core_id = -1);

    // Signals the thread to flush and stop
    void stop();

    // Diagnostics
    [[nodiscard]] std::uint64_t events_written() const noexcept;
    [[nodiscard]] std::uint64_t bytes_written() const noexcept;

private:
    void consumer_loop();
    void expand_file_mapping();
    void flush_to_disk();
    void prefault_memory(void* ptr, std::size_t size);

    std::string filepath_;
    std::shared_ptr<RingBuffer> ringbuffer_;
    
    std::thread io_thread_;
    std::atomic<bool> running_{false};
    
    // I/O State
    int fd_{-1};
    std::uint8_t* mapped_ptr_{nullptr};
    std::size_t current_file_size_{0};
    std::size_t write_offset_{0};
    std::size_t last_sync_offset_{0};
    
    std::atomic<std::uint64_t> events_written_{0};
    
    // 1 GB chunks for fallocate
    static constexpr std::size_t kChunkSize = 1024 * 1024 * 1024; 
};

} // namespace phase7::wal
