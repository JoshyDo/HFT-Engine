// =============================================================================
// spsc_ringbuffer.hpp - Single-Producer/Single-Consumer Lock-Free Ringbuffer
// =============================================================================
//
// Reference Algorithm: Dmitry Vyukov's bounded SPSC queue (2010).
// Industry standard in HFT and game engines.
//
// TWO atomic counters:
//   head_ : Position where the producer WRITES next.
//   tail_ : Position where the consumer READS next.
//
// Content Invariant: Slot [tail_ .. head_) contains valid items.
//   head_ == tail_  => EMPTY
//   head_ - tail_ == capacity  => FULL
//
// MEMORY ORDERING LOGIC (the core implementation):
// -----------------------------------------------------------------------------
//
//  PRODUCER writes item to Slot[head_ % capacity]:
//    1. Load tail_ with ACQUIRE: ensures the producer sees the consumer's
//       latest RELEASE (tail_.store(release)) and thus the "freeing" of
//       the target slot. Without ACQUIRE, it could write into a slot
//       still being read by the consumer -> DATA RACE.
//
//    2. Write slot content with RELAXED: Producer is the sole writer
//       of this slot (SPSC invariant). The consumer only reads AFTER
//       it executes head_.load(acquire) (see Consumer-3), and that
//       ACQUIRE pairs with the RELEASE on head_.store (Step 4).
//       RELAXED for the slot itself is therefore sufficient.
//
//    3. Load head_ (own counter) with RELAXED: Producer is the sole
//       reader+writer of head_. NO other entity writes to head_.
//       The other entity reads head_ with ACQUIRE.
//
//    4. Store head_ with RELEASE: synchronizes with the consumer's
//       ACQUIRE (head_.load(acquire)), making the slot content
//       visible to the consumer.
//
//  CONSUMER reads item from Slot[tail_ % capacity]:
//    1. Load head_ with ACQUIRE: pairs with the Producer's RELEASE. Guarantees
//       that the written slot content is visible in the local cache upon
//       successful ACQUIRE.
//
//    2. Read slot content with RELAXED: Consumer is the sole reader.
//       Visibility is guaranteed by the Step-1 ACQUIRE.
//
//    3. Load tail_ (own counter) with RELAXED: Consumer is the sole
//       reader+writer of tail_. The other entity reads with ACQUIRE (see
//       Producer-1).
//
//    4. Store tail_ with RELEASE: pairs with the Producer's ACQUIRE
//       (tail_.load(acquire) in Producer-1), signaling "Slot is free".
//
//  WHY RELAXED is permitted for the "own" counter:
//    On x86, all loads/stores without explicit fences are inherently
//    sequentially consistent. On ARM/POWER, it would be incorrect. However:
//    we ONLY load head_ locally, write head_, and the OTHER entity reads
//    head_ with ACQUIRE. The synchronization between entities occurs VIA
//    the RELEASE/ACQUIRE pairing, not via our local counter load.
//
// CACHE-LINE ALIGNMENT (prevents false sharing):
//    head_ and tail_ are 64-byte aligned. If the Producer core holds
//    the head_ cache line (exclusively) and the Consumer core holds the
//    tail_ cache line, no "bouncing" occurs between cores.
//    Modifications by one core do not invalidate the other's line
//    (MESI protocol).
//
// POWER-OF-2 CAPACITY:
//    capacity MUST be 2^k so mask_ = capacity-1 can utilize AND instead of MOD.
//    AND requires 1-2 cycles, MOD requires 20-40 cycles. At millions of push/pop
//    operations per second, this separates the "hot-path" from a "bottleneck".
//
// THREAD-SAFETY GUARANTEE:
//    This implementation is ONLY thread-safe for exactly 1 Producer
//    and exactly 1 Consumer. MPMC requires CAS loops.
// =============================================================================
#pragma once

// C4324: Structure padded due to alignment specifier. Intentional
// for false-sharing prevention (head_/tail_ on separate cache lines).
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

template <typename T, std::size_t Capacity>
    requires(std::has_single_bit(Capacity) && Capacity >= 2)
class SPSCRingbuffer {
    static_assert(std::is_trivially_copyable_v<T>,
                  "SPSCRingbuffer requires a trivially_copyable T (prevents "
                  "allocations in the hot-path, guarantees memcpy safety).");

    // Mask: capacity-1, functions ONLY because Capacity is a power-of-2.
    static constexpr std::size_t kMask = Capacity - 1;

    // Dedicated cache-line buffer (64 bytes) prevents head_ and
    // tail_ from residing on the same cache line.
    alignas(64) std::atomic<std::uint64_t> head_{0};  // Producer writes
    alignas(64) std::atomic<std::uint64_t> tail_{0};  // Consumer writes

    // Slots: we utilize aligned_storage to prevent the compiler from
    // invoking a default constructor for T (T might not be default-constructible).
    // We treat the bytes as raw storage.
    //
    // NO alignas(64) here: if sizeof(T) < 64, this would massively
    // waste memory. If sizeof(T) >= 64, alignas(64) could be applied,
    // but that's an optimization - SPSC correctness relies on head_/tail_,
    // not slot alignment.
    alignas(alignof(T)) std::byte storage_[Capacity * sizeof(T)];

    // ---- Utility Functions ----
    [[nodiscard]] T* slot_ptr(std::uint64_t idx) noexcept {
        return std::launder(reinterpret_cast<T*>(storage_ + (static_cast<std::size_t>(idx) & kMask) * sizeof(T)));
    }

    [[nodiscard]] const T* slot_ptr(std::uint64_t idx) const noexcept {
        return std::launder(reinterpret_cast<const T*>(storage_ + (static_cast<std::size_t>(idx) & kMask) * sizeof(T)));
    }

public:
    SPSCRingbuffer() noexcept                        = default;
    SPSCRingbuffer(const SPSCRingbuffer&)            = delete;
    SPSCRingbuffer& operator=(const SPSCRingbuffer&) = delete;
    SPSCRingbuffer(SPSCRingbuffer&&)                 = delete;
    SPSCRingbuffer& operator=(SPSCRingbuffer&&)      = delete;

    // ---- Producer API ----

    // Attempts to write an item into the buffer.
    // Returns: true = success, false = buffer full.
    //
    // HOT-PATH: zero allocations, zero syscalls, zero locks.
    // Worst-case: 1 atomic load (tail) + 1 atomic store (head) + memcpy.
    [[nodiscard]] bool try_push(const T& item) noexcept {
        const std::uint64_t head = head_.load(std::memory_order_relaxed);
        const std::uint64_t tail = tail_.load(std::memory_order_acquire);

        if (head - tail >= Capacity) [[unlikely]] {
            return false;  // full
        }

        // Write slot. RELAXED, because visibility is guaranteed by
        // the head_.store(release) below.
        std::memcpy(slot_ptr(head), &item, sizeof(T));

        // RELEASE: synchronizes with Consumer's head_.load(acquire).
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_push(T&& item) noexcept {
        const std::uint64_t head = head_.load(std::memory_order_relaxed);
        const std::uint64_t tail = tail_.load(std::memory_order_acquire);

        if (head - tail >= Capacity) [[unlikely]] {
            return false;
        }

        std::memcpy(slot_ptr(head), &item, sizeof(T));
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // ---- Consumer API ----

    // Attempts to read an item and write it to 'out'.
    // Returns: true = success, false = buffer empty.
    [[nodiscard]] bool try_pop(T& out) noexcept {
        const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
        const std::uint64_t head = head_.load(std::memory_order_acquire);

        if (tail == head) [[unlikely]] {
            return false;  // empty
        }

        // Read slot. RELAXED, because ACQUIRE above guarantees
        // visibility (paired with Producer's head_.store(release)).
        std::memcpy(&out, slot_ptr(tail), sizeof(T));

        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // Zero-Copy Peek: yields a pointer to the next item without copying.
    // Returns nullptr if empty. USABLE ONLY if the item is consumed
    // via pop_consumer_acked() following peek(). Otherwise, the slot
    // remains reserved and blocks the Producer.
    [[nodiscard]] const T* try_peek() const noexcept {
        const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
        const std::uint64_t head = head_.load(std::memory_order_acquire);

        if (tail == head) [[unlikely]] {
            return nullptr;
        }
        return slot_ptr(tail);
    }

    [[nodiscard]] T* try_peek() noexcept {
        const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
        const std::uint64_t head = head_.load(std::memory_order_acquire);

        if (tail == head) [[unlikely]] {
            return nullptr;
        }
        return slot_ptr(tail);
    }

    // Consumer acknowledges consumption following try_peek(). Increments tail_.
    void pop_consumer_acked() noexcept {
        const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
        tail_.store(tail + 1, std::memory_order_release);
    }

    // ---- Diagnostics (NOT for Hot-Path) ----
    [[nodiscard]] std::size_t capacity() const noexcept { return Capacity; }

    // Approximation: may return stale values due to non-atomic read.
    // Strictly for telemetry / testing.
    [[nodiscard]] std::size_t size_approx() const noexcept {
        const std::uint64_t h = head_.load(std::memory_order_acquire);
        const std::uint64_t t = tail_.load(std::memory_order_acquire);
        return static_cast<std::size_t>(h - t);
    }

    [[nodiscard]] bool empty_approx() const noexcept {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool full_approx() const noexcept {
        return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire) >= Capacity;
    }
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif
