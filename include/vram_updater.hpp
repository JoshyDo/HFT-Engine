// =============================================================================
// vram_updater.hpp - SPSC -> CUDA VRAM Pipeline (Phase 7.5)
// =============================================================================
//
// ARCHITECTURE:
//   WebSocket-Thread (Beast) -> SPSC-Ringbuffer -> VRAMUpdater-Thread -> VRAM
//
//   [Beast]   try_push(MarketTick)  // Producer
//   [SPSC]    65536 Slots, 2 MB
//   [VRAM-Up] try_pop() in Batches  // Consumer
//   [CUDA]    cudaMemcpyAsync(MarketTick*, d_buffer, sizeof(MarketTick) * N)
//
// HOT-PATH OPTIMIZATIONS:
//   1. PINNED HOST MEMORY: cudaMallocHost instead of malloc. Pageable memory
//      forces CUDA to use an internal staging buffer per Memcpy
//      (silent ~30% latency penalty).
//   2. BATCHING: 256 ticks per cudaMemcpyAsync. Smaller = PCIe roundtrip
//      too expensive. Larger = worse latency for individual updates.
//   3. ASYNC: cudaMemcpyAsync with a dedicated stream. We DO NOT block
//      after each Memcpy (cudaDeviceSynchronize would stall the pipeline).
//   4. ZERO-COPY READ: try_peek() + pop_consumer_acked() - we DO NOT copy
//      into the staging buffer, but use the SPSC storage directly as the source.
//      Requirement: SPSC memory is 32-byte aligned (see slot_ptr) and cudaMemcpy
//      permits it.
//   5. DROP-ON-PRESSURE: If VRAM push exceeds X ms, we drop ticks.
//      Stale data is preferable to a complete pipeline stall.
//
// VRAM LAYOUT:
//   We write ticks to a VRAM ring buffer (d_capacity ticks).
//   When the VRAM ring is full, we overwrite the oldest slot (Circular).
//   The GPU kernel reads from this VRAM ring (Phase 8).
//   This class handles the allocation and exposure of the VRAM buffer.
// =============================================================================
#pragma once

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#else
typedef int cudaError_t;
typedef int cudaStream_t;
typedef int cudaEvent_t;
constexpr int cudaSuccess = 0;
constexpr int cudaMemcpyHostToDevice = 1;
constexpr int cudaEventDisableTiming = 0;
inline const char* cudaGetErrorString(int) { return "CUDA not available"; }
inline int cudaMalloc(void**, size_t) { return 0; }
inline int cudaMallocHost(void**, size_t) { return 0; }
inline int cudaFree(void*) { return 0; }
inline int cudaFreeHost(void*) { return 0; }
inline int cudaStreamDestroy(int) { return 0; }
inline int cudaEventCreateWithFlags(int*, int) { return 0; }
inline int cudaEventDestroy(int) { return 0; }
inline int cudaMemcpyAsync(void*, const void*, size_t, int, int) { return 0; }
inline int cudaEventRecord(int, int) { return 0; }
inline int cudaStreamWaitEvent(int, int, int) { return 0; }
inline int cudaMemGetInfo(size_t* free, size_t* total) { if(free)*free=1024*1024; if(total)*total=1024*1024; return 0; }
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#ifndef _MM_PAUSE_DEFINED_HFT
#define _MM_PAUSE_DEFINED_HFT
static inline void _mm_pause() {
    __asm__ __volatile__("yield" ::: "memory");
}
#endif
#else
#ifndef _MM_PAUSE_DEFINED_HFT
#define _MM_PAUSE_DEFINED_HFT
static inline void _mm_pause() {}
#endif
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "market_tick.hpp"
#include "spsc_ringbuffer.hpp"
#include "thread_pinning.hpp"

namespace phase7 {

// VRAM Ring configuration. Default: 1M ticks (32 MB VRAM).
struct VRAMRingConfig {
    std::size_t  vram_capacity_ticks = 1u << 20;  // 1M Ticks = 32 MB
    std::size_t  batch_size          = 4096;      // Hard-Cap: max ticks per Memcpy
    cudaStream_t stream              = 0;         // 0 = default stream
    // Policy when VRAM ring is full: overwrite (true) or drop (false).
    bool overwrite_on_full = true;
    // =====================================================================
    // Phase 8.8.4: PCIe Coalescing via High-Watermark + Timeout.
    // ---------------------------------------------------------------------
    // Prior to Phase 8.8.4, we triggered a cudaMemcpyAsync per batch (256 ticks
    // = 8 KB). However, with 8 symbols, we average only 0.5-1 ticks/s per symbol,
    // resulting in 4-8 ticks/s aggregated. With Batch=256, the updater waited
    // 25-60s on average to fill a batch.
    // This led to STAGGERED SUBMISSIONS where the CUDA driver initiated a fresh
    // DMA path per Memcpy — causing PCIe submission latency spikes up to 5 ms
    // (measured Phase 8.8.3 ab875final: pcie_max_ns_ = 5.37 ms).
    //
    // Solution: HIW (high_watermark) + TIMEOUT (coalesce_timeout).
    //   - Once >= hiw ticks are in SPSC: trigger memcpy IMMEDIATELY.
    //   - If 1 ms elapses without reaching hiw: trigger memcpy with the
    //     accumulated ticks anyway (Latency Cap).
    //   - hiw = 128 (4 KB), timeout = 1 ms. This matches ~128 ticks/s
    //     max throughput. Since we observe 4-8 ticks/s, the timeout path
    //     dominates, reducing API overhead by 10x+.
    // =====================================================================
    // Diagnostics: Memcpy trigger count (Counter for A/B testing).
    // WARNING: atomic members are NON-COPYABLE. We expose them via methods
    // instead of config fields. The counters reside on the VRAMUpdater instance.
    std::size_t               high_watermark_ticks = 128;  // 128 * 32 B = 4 KB
    std::chrono::microseconds coalesce_timeout{1000};      // 1 ms
    // Phase 8.8.4: Pinning the consumer loop to a logical core ID.
    // -1 = no pinning (legacy/test mode).
    // The caller typically passes 4 (isolated CCD away from Main/WS).
    int worker_core_id = -1;
};

class VRAMUpdater {
public:
    using RingBuffer = SPSCRingbuffer<MarketTick, 65536>;

    VRAMUpdater(std::shared_ptr<RingBuffer> rb, VRAMRingConfig cfg = {})
        : rb_(std::move(rb)), cfg_(cfg), vram_slots_(cfg.vram_capacity_ticks), worker_core_id_(cfg.worker_core_id) {
        // Allocate VRAM ring. 32-byte alignment is guaranteed by
        // cudaMalloc (CUDA spec).
        const cudaError_t e1 =
            cudaMalloc(reinterpret_cast<void**>(&d_ring_), cfg_.vram_capacity_ticks * sizeof(MarketTick));
        if (e1 != cudaSuccess) {
            std::fprintf(stderr, "[VRAMUpdater] cudaMalloc failed: %s\n", cudaGetErrorString(e1));
            throw std::runtime_error("VRAM allocation failed");
        }

        // NOTE Phase 8.8.4 final: Dedicated non-default stream
        // YIELDS NO BENEFIT, as cudaDeviceSynchronize in the main thread
        // waits on the default stream — which does NOT alleviate the updater
        // stream, but serializes the driver lock anyway.
        // We therefore use the default stream (cfg_.stream == 0) and
        // leverage PCIe coalescing logic (HiW + Timeout) to reduce
        // submit frequency. The coalesce counters prove this is effective
        // (1 Memcpy per 1-2 ticks).

        // Host staging buffer (PINNED) for the Memcpy source.
        // We DO NOT use SPSC storage directly because cudaMemcpyAsync
        // implicitly creates a staging buffer on "pageable -> device" paths.
        // With pinned host memory, the path leverages direct DMA.
        const cudaError_t e2 =
            cudaMallocHost(reinterpret_cast<void**>(&h_staging_), cfg_.batch_size * sizeof(MarketTick));
        if (e2 != cudaSuccess) {
            std::fprintf(stderr, "[VRAMUpdater] cudaMallocHost failed: %s\n", cudaGetErrorString(e2));
            cudaFree(d_ring_);
            if (owns_stream_) cudaStreamDestroy(internal_stream_);
            throw std::runtime_error("Pinned host allocation failed");
        }

        // Phase 8.8.4: Per-Memcpy Event Tracking. Instead of cudaDeviceSync
        // (which blocks ALL streams), we expose one cudaEvent per cudaMemcpyAsync.
        // The main thread can then await the specific Memcpy using wait_for_completion(N),
        // without blocking the updater stream.
        //
        // Ring buffer: We preemptively allocate 64 events and recycle them.
        // 64 is sufficient for >64 ms pipeline depth (even at 1 Memcpy/ms
        // = 64 ms lookahead), which is well beyond our requirements.
        constexpr std::size_t kEventRingSize = 64;
        event_ring_.resize(kEventRingSize);
        for (std::size_t i = 0; i < kEventRingSize; ++i) {
            cudaEventCreateWithFlags(&event_ring_[i], cudaEventDisableTiming);
        }
        event_ring_mask_ = kEventRingSize - 1;
        event_ring_head_.store(0, std::memory_order_relaxed);

        vram_head_ = 0;  // next VRAM slot to be written
        stop_flag_.store(false, std::memory_order_release);
    }

    // Helper: returns the effective stream. With Phase 8.8.4 final,
    // we ALWAYS utilize the default stream (no dedicated stream) because
    // lock contention between non-default and default caused PCIe spikes
    // up to 80 ms.
    [[nodiscard]] cudaStream_t stream_() const noexcept {
        return 0;  // default stream
    }

    ~VRAMUpdater() {
        stop();
        if (worker_.joinable()) {
            worker_.join();
        }
        if (d_ring_) cudaFree(d_ring_);
        if (h_staging_) cudaFreeHost(h_staging_);
        for (auto& e : event_ring_) {
            if (e) cudaEventDestroy(e);
        }
        if (owns_stream_ && internal_stream_) {
            cudaStreamDestroy(internal_stream_);
        }
    }

    VRAMUpdater(const VRAMUpdater&)            = delete;
    VRAMUpdater& operator=(const VRAMUpdater&) = delete;

    // Starts the consumer loop on an internal background thread.
    void run() {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) {
            return;
        }
        const int core = worker_core_id_;
        worker_        = std::thread([this, core]() {
            // Phase 8.8.4: Thread pinning immediately AFTER thread launch.
            // Core IDs are logical CPUs (including HT siblings). The caller
            // sets worker_core_id_ via Config; -1 disables pinning
            // (default for tests / single-core).
            if (core >= 0) {
                const bool ok = ::phase7::pin_current_thread_to_core(core);
                if (!ok) {
                    std::fprintf(stderr, "[VRAMUpdater] WARNING: pinning to core %d failed\n", core);
                }
            }
            this->loop();
        });
    }

    // Halts the execution loop and joins the background thread.
    void stop() {
        if (!running_.exchange(false)) return;
        stop_flag_.store(true, std::memory_order_release);
    }

    // Diagnostik.
    [[nodiscard]] std::uint64_t ticks_pushed_to_vram() const noexcept {
        return ticks_pushed_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t batches_pushed() const noexcept { return batches_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t drops_overwrite() const noexcept {
        return drops_overwrite_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t drops_backpressure() const noexcept {
        return drops_backpressure_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] MarketTick*   device_ring_ptr() noexcept { return d_ring_; }
    [[nodiscard]] std::size_t   device_ring_capacity() const noexcept { return cfg_.vram_capacity_ticks; }
    [[nodiscard]] std::uint64_t device_ring_head() const noexcept { return vram_head_.load(std::memory_order_acquire); }

    // Phase 8.5.2: Access the pinned host staging buffer for the latest batch.
    // IMPORTANT: The pointer addresses cfg_.batch_size MarketTick slots. Only
    // the first `n` slots (returned by step_once()) hold valid data from
    // the most recent Memcpy. The buffer is overwritten on the next step_once() call.
    [[nodiscard]] MarketTick* host_staging_ptr() noexcept { return h_staging_; }
    [[nodiscard]] std::size_t host_staging_capacity() const noexcept { return cfg_.batch_size; }

    // =====================================================================
    // Phase 8.2.2: PCIe API Submission Latency (Histogram + Max + Mean).
    // ---------------------------------------------------------------------
    // Measures elapsed time BEFORE the first cudaMemcpyAsync until AFTER the last.
    // This captures the DRIVER latency (pushing cudaMemcpy into the queue).
    // The TRUE PCIe transfer latency (DMA across the bus) is measured later
    // using cudaEvents (Phase 8.2.3).
    //
    // Buckets (logarithmic, nanoseconds): identical to the ingestion probe.
    // =====================================================================
    static constexpr std::size_t kPcieBuckets = 6;

    [[nodiscard]] std::uint64_t pcie_count() const noexcept { return pcie_count_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t pcie_max_ns() const noexcept { return pcie_max_ns_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t pcie_sum_ns() const noexcept { return pcie_sum_ns_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t pcie_bucket(std::size_t i) const noexcept {
        return (i < kPcieBuckets) ? pcie_hist_[i].load(std::memory_order_relaxed) : 0;
    }
    [[nodiscard]] double pcie_mean_ns() const noexcept {
        const auto c = pcie_count();
        return c ? static_cast<double>(pcie_sum_ns()) / c : 0.0;
    }
    // Bytes-Throughput-Stats. n_ticks * 32B per Memcpy.
    [[nodiscard]] std::uint64_t pcie_bytes_total() const noexcept {
        return pcie_bytes_.load(std::memory_order_relaxed);
    }

    // Phase 8.8.4: Fine-grained Stream Ordering (non-blocking).
    //
    // Issues cudaStreamWaitEvent on the default stream targeting the
    // most recent cudaMemcpyAsync. This allows the default stream
    // (BF / Update-Kernel) to recognize the Updater Memcpy as completed,
    // WITHOUT initiating a block here.
    //
    // This is the architecturally rigorous approach:
    //   - Non-blocking (eliminates PCIe spikes)
    //   - Non-locking (avoids Stream Lock Contention)
    //   - CUDA internally serializes streams automatically
    void order_after_pending([[maybe_unused]] std::uint64_t ticks_count) {
        (void)ticks_count;
        if (event_ring_.empty()) return;
        const std::uint64_t head = event_ring_head_.load(std::memory_order_acquire);
        if (head == 0) return;
        const std::uint64_t idx = (head - 1) & event_ring_mask_;
        // cudaStreamWaitEvent is non-blocking. It merely registers
        // a dependency; the default stream (0) will automatically wait
        // at the next launch until the event is signaled.
        cudaStreamWaitEvent(0, event_ring_[idx], 0);
    }

    // Synchronous variant (legacy): reads ALL currently available ticks
    // up to batch_size and writes them to VRAM. Invoked internally by
    // loop() with Watermark+Timeout logic (NOT in the hot path).
    // Returns: Number of ticks flushed to VRAM (0 if empty).
    std::size_t step_once() { return flush_now(0, cfg_.batch_size); }

    // Phase 8.8.4: New drain method with explicit capacity limits. Invoked
    // by both step_once() (legacy, drain all) and loop() under HiW/Timeout logic.
    // max_to_collect: Hard cap (safeguard against severe bursts). 0 = unlimited
    //                 up to cfg_.batch_size.
    // Returns: Number of flushed ticks (0 if SPSC is empty).
    std::size_t flush_now(std::size_t min_to_collect, std::size_t max_to_collect) {
        (void)min_to_collect;  // siehe Drain-Kommentar unten
        if (h_staging_ == nullptr || d_ring_ == nullptr) return 0;
        if (max_to_collect == 0) max_to_collect = cfg_.batch_size;
        if (max_to_collect > cfg_.batch_size) max_to_collect = cfg_.batch_size;

        // 1. Drain SPSC up to max_to_collect or until empty.
        // We ALWAYS drain up to max because the SPSC API lacks
        // push-back. min_to_collect is thus ignored (kept for API
        // symmetry). step_once() and loop() invoke this with min=0.
        std::size_t n = 0;
        for (; n < max_to_collect; ++n) {
            const MarketTick* const t = rb_->try_peek();
            if (t == nullptr) [[unlikely]]
                break;
            std::memcpy(h_staging_ + n, t, sizeof(MarketTick));
            rb_->pop_consumer_acked();
        }
        if (n == 0) return 0;

        // 2. cudaMemcpyAsync to the VRAM ring. We start at vram_head_
        //    and wrap around modulo capacity.
        const std::uint64_t head = vram_head_.load(std::memory_order_relaxed);
        const std::size_t   cap  = cfg_.vram_capacity_ticks;

        // If Batch > remaining slots before wrap: split the memcpy.
        const std::size_t chunk_a = (std::min)(n, cap - static_cast<std::size_t>(head % cap));
        const std::size_t chunk_b = n - chunk_a;

        MarketTick* dst_a = d_ring_ + (head % cap);

        // Phase 8.2.2: PCIe API Submission Timer.
        // Measures cudaMemcpyAsync latency (driver overhead), NOT the
        // true PCIe transfer latency. cudaMemcpyAsync is async, hence
        // the measured duration purely reflects the "submission" cost.
        const auto pcie_t0 = std::chrono::high_resolution_clock::now();

        if (chunk_a > 0) {
            const cudaError_t e =
                cudaMemcpyAsync(dst_a, h_staging_, chunk_a * sizeof(MarketTick), cudaMemcpyHostToDevice, stream_());
            if (e != cudaSuccess) [[unlikely]] {
                std::fprintf(stderr, "[VRAMUpdater] memcpy A failed: %s\n", cudaGetErrorString(e));
                drops_backpressure_.fetch_add(n, std::memory_order_relaxed);
                return 0;
            }
        }
        if (chunk_b > 0) {
            const cudaError_t e = cudaMemcpyAsync(
                d_ring_, h_staging_ + chunk_a, chunk_b * sizeof(MarketTick), cudaMemcpyHostToDevice, stream_());
            if (e != cudaSuccess) [[unlikely]] {
                std::fprintf(stderr, "[VRAMUpdater] memcpy B failed: %s\n", cudaGetErrorString(e));
                drops_backpressure_.fetch_add(chunk_b, std::memory_order_relaxed);
            }
        }

        // Phase 8.8.4: Record per-Memcpy event. IMPORTANT: We
        // record the event AFTER the PCIe timer (see below), because
        // cudaEventRecord acts as a stream fence and would inflate
        // the measured PCIe value. Placing the event sync outside
        // the measurement keeps the histogram statistics pure.
        const std::uint64_t event_idx = event_ring_head_.fetch_add(1, std::memory_order_acq_rel) & event_ring_mask_;

        const auto          pcie_t1 = std::chrono::high_resolution_clock::now();
        const std::uint64_t pcie_ns =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(pcie_t1 - pcie_t0).count());

        // Histogram update: identical logic to ingestion.
        std::size_t   b  = 0;
        std::uint64_t nn = pcie_ns;
        while (nn >= 10 && b + 1 < kPcieBuckets) {
            nn /= 10;
            ++b;
        }
        if (b >= kPcieBuckets) b = kPcieBuckets - 1;
        pcie_hist_[b].fetch_add(1, std::memory_order_relaxed);
        pcie_count_.fetch_add(1, std::memory_order_relaxed);
        pcie_sum_ns_.fetch_add(pcie_ns, std::memory_order_relaxed);
        pcie_bytes_.fetch_add(n * sizeof(MarketTick), std::memory_order_relaxed);
        // Max tracking using atomic CAS loop.
        std::uint64_t cur_max = pcie_max_ns_.load(std::memory_order_relaxed);
        while (pcie_ns > cur_max && !pcie_max_ns_.compare_exchange_weak(cur_max, pcie_ns, std::memory_order_relaxed)) {
        }

        // cudaEventRecord AFTER the PCIe timing. The event is
        // still recorded BEFORE vram_head_.store to maintain
        // correct stream ordering. We utilize the default stream
        // (stream_() == 0); the main thread can await it using
        // cudaStreamWaitEvent.
        cudaEventRecord(event_ring_[event_idx], stream_());

        vram_head_.store(head + n, std::memory_order_release);
        ticks_pushed_.fetch_add(n, std::memory_order_relaxed);
        batches_.fetch_add(1, std::memory_order_relaxed);
        return n;
    }

private:
    void loop() {
        // Phase 8.8.4: HiW + Timeout Coalescing.
        //
        // Design: We sleep in 50us slices until ONE of two triggers
        // fires:
        //   - HiW: SPSC contains >= high_watermark_ticks ticks
        //   - Timeout: >= coalesce_timeout elapsed since "last_window_start"
        //              AND at least 1 tick is available
        //
        // IMPORTANT: We must perform PEEK-only size checks without
        // draining. Otherwise, the probe-drain disrupts the coalescing
        // (each 50us slice would extract 1-2 ticks, leaving the SPSC
        // empty when the actual timeout event occurs).
        //
        // The SPSC API lacks a native size() method, but we can
        // measure this via read/write difference
        // — see rb_->size_approx() (in header).
        using clock                = std::chrono::steady_clock;
        [[maybe_unused]] constexpr auto kSliceSleep = std::chrono::microseconds(50);

        const std::size_t hiw     = cfg_.high_watermark_ticks;
        const auto        timeout = cfg_.coalesce_timeout;

        // 1. Wait for the first tick (blocking slice). We only start
        //    the coalesce window once at least 1 tick is present.
        while (running_.load(std::memory_order_acquire) && !has_any_pending()) {
            _mm_pause();
        }
        if (!running_.load(std::memory_order_acquire)) return;
        auto t_window_start = clock::now();

        // 2. Coalesce loop. Probe-size per slice, trigger on HiW
        //    or Timeout.
        while (running_.load(std::memory_order_acquire)) {
            const std::size_t pending = pending_count();

            // --- Trigger 1: HiW Reached ---
            if (pending >= hiw) {
                const std::size_t pushed = flush_now(0, cfg_.batch_size);
                if (pushed > 0) {
                    coalesce_early_flushes_.fetch_add(1, std::memory_order_relaxed);
                }
                t_window_start = clock::now();  // Window reset
                continue;
            }

            // --- Trigger 2: Timeout Elapsed + min. 1 Tick ---
            const auto now     = clock::now();
            const auto elapsed = now - t_window_start;
            if (elapsed >= timeout && pending > 0) [[likely]] {
                const std::size_t pushed = flush_now(0, cfg_.batch_size);
                if (pushed > 0) {
                    coalesce_timeout_flushes_.fetch_add(1, std::memory_order_relaxed);
                }
                t_window_start = clock::now();
            } else {
                // Wait for the next tick arrival or timeout.
                // We utilize _mm_pause() for ultra-low latency busy-waiting.
                _mm_pause();
            }
        }
    }

    // ---------------------------------------------------------------------
    // PEEK-only size check. SPSC offers size_approx() (lock-free, may
    // return stale values — but adequate for coalescing decisions, as
    // we only test a "minimum N" condition).
    // ---------------------------------------------------------------------
    std::size_t pending_count() const noexcept {
        if (!rb_) return 0;
        return rb_->size_approx();
    }
    bool has_any_pending() const noexcept { return pending_count() > 0; }

    // ---- State ----
    std::shared_ptr<RingBuffer> rb_;
    VRAMRingConfig              cfg_;
    MarketTick*                 d_ring_    = nullptr;  // Device pointer (VRAM)
    MarketTick*                 h_staging_ = nullptr;  // Pinned host staging
    std::atomic<std::uint64_t>  vram_head_{0};
    std::atomic<std::uint64_t>  ticks_pushed_{0};
    std::atomic<std::uint64_t>  batches_{0};
    std::atomic<std::uint64_t>  drops_overwrite_{0};
    std::atomic<std::uint64_t>  drops_backpressure_{0};
    // Phase 8.2.2: PCIe API Submission Latency. Measured on the updater thread.
    std::atomic<std::uint64_t>                           pcie_count_{0};
    std::atomic<std::uint64_t>                           pcie_sum_ns_{0};
    std::atomic<std::uint64_t>                           pcie_max_ns_{0};
    std::atomic<std::uint64_t>                           pcie_bytes_{0};
    std::array<std::atomic<std::uint64_t>, kPcieBuckets> pcie_hist_{};
    std::atomic<bool>                                    running_{false};
    std::atomic<bool>                                    stop_flag_{false};
    std::thread                                          worker_;
    std::vector<MarketTick>                              vram_slots_;  // size marker only, no payload

    // Phase 8.8.4: CPU Thread Pinning. Logical core ID for the
    // worker. -1 = no pinning (default for tests). The caller
    // assigns this via Config.
    int worker_core_id_ = -1;
    // Phase 8.8.4: PCIe coalescing diagnostic counters (atomic, hence
    // located here rather than in Config).
    std::atomic<std::uint64_t> coalesce_early_flushes_{0};    // HiW triggered
    std::atomic<std::uint64_t> coalesce_timeout_flushes_{0};  // Timeout triggered
    // Phase 8.8.4: Per-Memcpy event ring for fine-grained stream sync.
    // We allocate a cudaEvent_t for each Memcpy, recorded into the stream.
    // wait_for_completion(n) blocks on the event for the n-th Memcpy.
    std::vector<cudaEvent_t>   event_ring_;
    std::uint64_t              event_ring_mask_ = 0;
    std::atomic<std::uint64_t> event_ring_head_{0};
    // Phase 8.8.4: Dedicated non-default stream, if the caller
    // specifies 0. Mitigates lock contention with other stream users
    // (e.g. main.cpp pipeline). owns_stream_ == true dictates
    // resource cleanup during destruction.
    cudaStream_t internal_stream_ = 0;
    bool         owns_stream_     = false;

public:
    // Diagnostic getters for coalescing.
    [[nodiscard]] std::uint64_t coalesce_early_flushes() const noexcept {
        return coalesce_early_flushes_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t coalesce_timeout_flushes() const noexcept {
        return coalesce_timeout_flushes_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t coalesce_total_flushes() const noexcept {
        return coalesce_early_flushes() + coalesce_timeout_flushes();
    }
};

}  // namespace phase7
