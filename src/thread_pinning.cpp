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
#endif  // _WIN32
