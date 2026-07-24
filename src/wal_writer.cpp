// =============================================================================
// wal_writer.cpp - Implementation of Background I/O Thread for Lock-Free WAL
// =============================================================================
#include "../include/wal/wal_writer.hpp"
#include "../include/thread_pinning.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <system_error>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#else
static inline void _mm_pause() {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#endif
}
#endif

namespace phase7::wal {

WalWriter::WalWriter(std::string filepath, std::shared_ptr<RingBuffer> ringbuffer)
    : filepath_(std::move(filepath)), ringbuffer_(std::move(ringbuffer)) {
    
    fd_ = ::open(filepath_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        throw std::system_error(errno, std::generic_category(), "Failed to open WAL file");
    }
    
    // Get current size
    struct stat st;
    if (::fstat(fd_, &st) < 0) {
        throw std::system_error(errno, std::generic_category(), "Failed to stat WAL file");
    }
    
    current_file_size_ = st.st_size;
    write_offset_ = current_file_size_;
    last_sync_offset_ = write_offset_;
    
    if (current_file_size_ > 0) {
        mapped_ptr_ = static_cast<std::uint8_t*>(::mmap(nullptr, current_file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
        if (mapped_ptr_ == MAP_FAILED) {
            mapped_ptr_ = nullptr;
            throw std::system_error(errno, std::generic_category(), "Failed to mmap existing WAL file");
        }
    }
}

WalWriter::~WalWriter() {
    stop();
    if (mapped_ptr_) {
        ::munmap(mapped_ptr_, current_file_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

void WalWriter::start(int core_id) {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return;
    }
    
    io_thread_ = std::thread([this, core_id]() {
        if (core_id >= 0) {
            if (!::phase7::pin_current_thread_to_core(core_id)) {
                std::fprintf(stderr, "[WalWriter] WARNING: pinning to core %d failed\n", core_id);
            }
        }
        consumer_loop();
    });
}

void WalWriter::stop() {
    if (!running_.exchange(false)) return;
    
    if (io_thread_.joinable()) {
        io_thread_.join();
    }
    flush_to_disk();
}

void WalWriter::expand_file_mapping() {
    std::size_t new_size = current_file_size_ + kChunkSize;
    
#if defined(__APPLE__)
    fstore_t store{};
    store.fst_flags = F_ALLOCATECONTIG;
    store.fst_posmode = F_PEOFPOSMODE;
    store.fst_offset = 0;
    store.fst_length = static_cast<off_t>(kChunkSize);
    if (fcntl(fd_, F_PREALLOCATE, &store) == -1) {
        store.fst_flags = F_ALLOCATEALL;
        if (fcntl(fd_, F_PREALLOCATE, &store) == -1) {
            // fallback to ftruncate
            if (::ftruncate(fd_, new_size) < 0) {
                throw std::system_error(errno, std::generic_category(), "ftruncate failed on expansion");
            }
        }
    }
    if (::ftruncate(fd_, new_size) < 0) {
         throw std::system_error(errno, std::generic_category(), "ftruncate failed on expansion");
    }
#else
    if (posix_fallocate(fd_, current_file_size_, kChunkSize) != 0) {
        if (::ftruncate(fd_, new_size) < 0) {
            throw std::system_error(errno, std::generic_category(), "ftruncate failed on expansion");
        }
    }
#endif

    // Remap
    std::uint8_t* new_ptr;
#if defined(__linux__)
    if (mapped_ptr_) {
        new_ptr = static_cast<std::uint8_t*>(::mremap(mapped_ptr_, current_file_size_, new_size, MREMAP_MAYMOVE));
        if (new_ptr == MAP_FAILED) {
            throw std::system_error(errno, std::generic_category(), "mremap failed");
        }
    } else {
        new_ptr = static_cast<std::uint8_t*>(::mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
        if (new_ptr == MAP_FAILED) {
            throw std::system_error(errno, std::generic_category(), "mmap failed");
        }
    }
#else
    // macOS / BSD fallback
    if (mapped_ptr_) {
        ::munmap(mapped_ptr_, current_file_size_);
    }
    new_ptr = static_cast<std::uint8_t*>(::mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (new_ptr == MAP_FAILED) {
        throw std::system_error(errno, std::generic_category(), "mmap failed on expansion");
    }
#endif

    mapped_ptr_ = new_ptr;
    
    // Pre-fault new memory
    prefault_memory(mapped_ptr_ + current_file_size_, kChunkSize);
    
    current_file_size_ = new_size;
}

void WalWriter::prefault_memory(void* ptr, std::size_t size) {
    volatile std::uint8_t* p = static_cast<volatile std::uint8_t*>(ptr);
    for (std::size_t i = 0; i < size; i += 4096) {
        p[i] = 0;
    }
}

void WalWriter::flush_to_disk() {
    if (mapped_ptr_ && write_offset_ > last_sync_offset_) {
        std::size_t size_to_sync = write_offset_ - last_sync_offset_;
        // align to page boundary for msync (4KB typical, but we just use start pointer)
        std::size_t page_offset = last_sync_offset_ & ~(4096 - 1);
        std::size_t sync_size = size_to_sync + (last_sync_offset_ - page_offset);
        
        ::msync(mapped_ptr_ + page_offset, sync_size, MS_ASYNC);
        last_sync_offset_ = write_offset_;
    }
}

void WalWriter::consumer_loop() {
    using namespace std::chrono;
    
    if (current_file_size_ == 0 || write_offset_ + sizeof(WalEvent) > current_file_size_) {
        expand_file_mapping();
    }

    auto last_flush_time = steady_clock::now();

    while (running_.load(std::memory_order_acquire)) {
        const WalEvent* event_ptr = ringbuffer_->try_peek();
        
        if (!event_ptr) {
            _mm_pause(); // Backoff
            
            auto now = steady_clock::now();
            if (duration_cast<milliseconds>(now - last_flush_time).count() >= 10) {
                flush_to_disk();
                last_flush_time = now;
            }
            continue;
        }

        // Check capacity
        if (write_offset_ + sizeof(WalEvent) > current_file_size_) {
            expand_file_mapping();
        }

        // Copy event to WAL
        std::memcpy(mapped_ptr_ + write_offset_, event_ptr, sizeof(WalEvent));
        write_offset_ += sizeof(WalEvent);
        events_written_.fetch_add(1, std::memory_order_relaxed);

        bool critical = (event_ptr->type == EventType::TRADE_EXECUTION || 
                         event_ptr->type == EventType::ORDER_SUBMIT);

        // Ack
        ringbuffer_->pop_consumer_acked();

        if (critical) {
            flush_to_disk();
            last_flush_time = steady_clock::now();
        }
    }
}

std::uint64_t WalWriter::events_written() const noexcept {
    return events_written_.load(std::memory_order_relaxed);
}

std::uint64_t WalWriter::bytes_written() const noexcept {
    return write_offset_;
}

} // namespace phase7::wal
