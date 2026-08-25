# High-Frequency Trading (HFT) TCP Loopback Engine

An ultra-low latency C++20 HFT arbitrage engine designed to push the absolute physical limits of the standard Linux `AF_INET` TCP stack.

This repository serves as a performance whitepaper and benchmarking suite, documenting the journey to achieve deterministic sub-3µs Tick-to-Trade (T2T) latency without relying on specialized networking hardware (NICs) or third-party kernel bypass libraries.

## Architecture & Performance Journey

The core objective was to build a deterministic, zero-allocation trading loop and measure the true cost of network parsing and order generation.

### 1. The Core Logic: 12 Nanoseconds (38 Cycles)
The heart of the engine is a tightly optimized, branch-less C++ processing pipeline.
- **Zero-Copy SBE Parsing**: Directly parsing SBE (Simple Binary Encoding) market data ticks from the socket buffer.
- **L2 Orderbook Updates**: O(1) in-place book updates.
- **Order Generation**: Aggressive IOC order generation using SIMD-friendly structs.

**Result**: The median time from byte-buffer parse to outgoing order structure generation is **12 ns (38 CPU cycles)**. The software logic operates physically at the L1 cache instruction limit.

### 2. Baseline: Standard Linux TCP (~6.3 µs)
Initial tests using standard blocking `send()`/`recv()` calls yielded a T2T latency of ~6.3 µs. While fast for standard web applications, the syscall context switching (Ring-3 to Ring-0) and `ksoftirqd` scheduling overhead introduce unacceptable jitter for High-Frequency Trading.

### 3. The `LD_PRELOAD` AF_UNIX Anomaly (~0.76 µs)
To bypass the TCP stack, an `LD_PRELOAD` shared library was written to transparently intercept `AF_INET` socket calls and route them through `AF_UNIX` Domain Sockets (Shared Memory IPC). 
While this dropped the latency to an incredible **0.76 µs**, it was rejected as a vanity metric. Real exchanges (Eurex, CME) route orders via TCP/UDP over fiber optics, not local shared memory. Bypassing the network stack entirely masks the real-world network parsing characteristics we set out to benchmark.

### 4. The Final Architecture: `io_uring` SQPOLL (~2.33 µs)
To achieve the absolute physical limit of pure `AF_INET` TCP on Linux, we eliminated the system call overhead entirely while staying true to the network stack:
- **io_uring SQPOLL**: Configured with `IORING_SETUP_SQPOLL`, offloading the `send`/`recv` submissions to a dedicated kernel thread. This completely eliminates context switches.
- **SINGLE_ISSUER & Lock-Free Ring**: Configured with `IORING_SETUP_SINGLE_ISSUER` to strip atomic locks from the `io_uring` hotpath.
- **Core Isolation & IRQ Affinity**: Pinned the C++ Engine to Core 4, the Mock Exchange to Core 2, the `io_uring` SQ threads to Cores 3 & 5, and strictly isolated the Linux Network Soft-IRQ (`ksoftirqd`) to Core 6. This prevents the 100% busy-polling userspace threads from starving the kernel packet delivery mechanism.

**Final Deterministic Result (99th Percentile Jitter Eliminated):**
```text
Results (98999 samples after warmup):
--- Core Logic (Parse -> Book -> Order) ---
Median: 38 CPU Cycles (~0.0126 us)
99th %: 38 CPU Cycles (~0.0126 us)

--- Software T2T (Core Logic + SQPOLL submit) ---
Median: 6992 CPU Cycles (~2.33 us)
99th %: 7296 CPU Cycles (~2.43 us)
```
*Note: 2.3 µs represents the mathematical floor for a loopback packet to traverse the `tcp_v4_rcv` Linux kernel stack on a 5 GHz CPU. Standard OS optimizations (e.g. `nohz_full`, `rcu_nocbs`) were used to eliminate the 1000Hz timer tick overhead.*

## Usage & Execution

```bash
# 1. Compile the project
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 2. Run the Benchmark (Root required for SQPOLL and IRQ Affinity)
sudo ./run_final.sh
```

## Future Work
- **True Kernel Bypass (NIC-level)**: Integrating DPDK or Solarflare OpenOnload (via `EF_VI`) to completely bypass the kernel on physical network interfaces to break the 1µs barrier.
- **Orderbook Scaling**: Stress-testing the 38-cycle core logic by scaling the L3 limit orderbook to track 10,000+ active instruments, forcing cache contention and memory bandwith analysis.
- **UDP Multicast Ingestion**: Migrating the market data feed to UDP Multicast to mirror real-world exchange architectures (e.g., Eurex T7 / CME Globex).
