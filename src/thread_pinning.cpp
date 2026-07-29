// =============================================================================
// thread_pinning.cpp - Thread-pinning helpers implementation
// =============================================================================
//
// This is the sole translation unit (TU) fully including <windows.h>.
// This mitigates header ordering conflicts with windows_lean.hpp, which defines
// WIN32_LEAN_AND_MEAN and excludes several Windows structures.
// =============================================================================
#include "thread_pinning.hpp"

#ifdef _WIN32
// Architecture macros MUST be set BEFORE any Windows headers are included
// (otherwise <winnt.h> emits a "No Target Architecture" error).
#ifndef _AMD64_
#if defined(_M_AMD64) || defined(__x86_64__)
#define _AMD64_ 1
#endif
#endif

// Full include: We intentionally bypass WIN32_LEAN_AND_MEAN here because
// our thread affinity functions (GetCurrentThread, SetThreadAffinityMask)
// and SYSTEM_INFO depend on structures otherwise excluded.
#include <windows.h>

namespace phase7 {

bool pin_current_thread_to_core(int core_id) noexcept {
    if (core_id < 0) return false;
    if (core_id >= 64) return false;  // DWORD_PTR = 64-bit
    const DWORD_PTR mask   = static_cast<DWORD_PTR>(1) << core_id;
    const HANDLE    thread = ::GetCurrentThread();
    const DWORD_PTR prev   = ::SetThreadAffinityMask(thread, mask);
    return prev != 0;
}

unsigned int logical_processor_count() noexcept {
    ::SYSTEM_INFO si;
    ::ZeroMemory(&si, sizeof(si));
    ::GetSystemInfo(&si);
    return static_cast<unsigned int>(si.dwNumberOfProcessors);
}

}  // namespace phase7
#else
#include <pthread.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/thread_policy.h>
#endif

namespace phase7 {

bool pin_current_thread_to_core(int core_id) noexcept {
    if (core_id < 0) return false;
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();
    return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) == 0;
#elif defined(__APPLE__)
    thread_affinity_policy_data_t policy = { core_id };
    thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
    return thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                             (thread_policy_t)&policy,
                             THREAD_AFFINITY_POLICY_COUNT) == KERN_SUCCESS;
#else
    return false;
#endif
}

unsigned int logical_processor_count() noexcept {
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return nprocs > 0 ? static_cast<unsigned int>(nprocs) : 0;
}

}  // namespace phase7
#endif  // _WIN32
