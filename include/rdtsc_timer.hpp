#pragma once
#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
// Fallback for ARM Macs so it compiles, though rdtsc is x86 specific
// We can use generic timer or just mock it to compile. 
// For real benchmarking on ARM, we'd use mrs cntvct_el0.
#include <chrono>
#endif
#include <cstdint>

class RdtscTimer {
public:
    inline void start() {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_lfence(); // Verhindert, dass spätere Befehle vorgezogen werden
        start_ts = __rdtsc();
#else
        start_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
    }

    inline void stop() {
#if defined(__x86_64__) || defined(_M_X64)
        unsigned int aux;
        stop_ts = __rdtscp(&aux); // Wartet auf Abschluss aller vorherigen Befehle
        _mm_lfence();
#else
        stop_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
    }

    inline uint64_t elapsed_cycles() const {
        return stop_ts - start_ts;
    }

private:
    uint64_t start_ts = 0;
    uint64_t stop_ts = 0;
};
