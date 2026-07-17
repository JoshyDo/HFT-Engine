// =============================================================================
// thread_pinning.hpp - CPU Thread Pinning Helpers (Phase 8.8.4)
// =============================================================================
//
// RATIONALE FOR THREAD PINNING:
// Standard OS desktop schedulers frequently migrate threads across logical
// CPUs. Every migration incurs an L1/L2 cache invalidation penalty (10-50us
// stall) and potential cross-CCD QPI latencies. At high tick ingestion rates,
// every stall directly impairs execution determinism.
//
// DEPLOYMENT STRATEGY:
//   - Core 0 (or Core 1 with HT sibling): Reserved for OS/Interrupts.
//   - WS-Client (Boost.Beast): Core 2 (isolated CCD preferred).
//   - VRAM-Updater: Core 4 (isolated CCD preferred).
//   - Main / BF-Scheduler: Runs on Core 1 (sibling of Core 0).
// Affinity masks are advisory; the OS kernel retains the ability to migrate
// if absolutely necessary, but we capture the 90% cache-hit baseline.
//
// ARCHITECTURAL DECISION (HEADER-ONLY VS .CPP):
// These functions were previously inlined. However, any translation unit (TU)
// including this header subsequently required <windows.h> and specific macros.
// Unit tests frequently bypassed windows_lean.hpp, causing compilation errors.
// Consequently, only declarations remain here; implementations reside within
// src/thread_pinning.cpp.
// =============================================================================
#pragma once

namespace phase7 {

#ifdef _WIN32
// Pins the CURRENT executing thread to a specific logical processor.
// Core IDs are zero-indexed and typically encompass logical CPUs
// (including HT siblings).
//
// Returns: true on success, false if the core does not exist
// or the affinity mask is invalid.
bool pin_current_thread_to_core(int core_id) noexcept;

// Retrieves the total number of logical processors available in the system.
// Returns 0 if unavailable.
unsigned int logical_processor_count() noexcept;
#else
// POSIX fallback: typically pthread_setaffinity_np. Currently an inline stub.
inline bool pin_current_thread_to_core(int) noexcept {
    return false;
}
inline unsigned int logical_processor_count() noexcept {
    return 0;
}
#endif

}  // namespace phase7
