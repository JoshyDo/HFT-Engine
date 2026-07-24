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
#include <simdjson.h>

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <chrono>
#include <csr_graph.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_test.hpp>
#include <device_graph.hpp>
#include <graph_generator.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <thread>

#include "arbitrage_optimized.hpp"
#include "bellman_ford.hpp"
#include "dsl-ast.hpp"
#include "dsl-lexer.hpp"
#include "dsl-parser.hpp"
#include "dsl_compiler.hpp"
#include "dsl_vm.hpp"
#include "execution_engine.hpp"
#include "live_arbitrage.hpp"
#include "market_tick.hpp"
#include "stream_parser.hpp"
#include "spsc_ringbuffer.hpp"
#include "thread_pinning.hpp"
#include "vram_updater.hpp"
#include "windows_lean.hpp"
#include "feed_client.hpp"
#include "parsers.hpp"
#include "wal/wal_writer.hpp"
#include "transports/websocket_transport.hpp"
#include "transports/udp_multicast_transport.hpp"

using WsClient = phase7::WebSocketTransport<phase7::BinanceJsonParser>;
using UdpClient = phase7::UdpMulticastTransport<phase7::SbeParser>;

// =============================================================================
// PHASE 7 TOOLCHAIN SANITY: Both libraries must be includable and generate
// a viable code path. Once the definitive WebSocket client is active
// (Step 4), this block will be superseded.
namespace {
inline void phase7_boot_check() {
    // Beast provides version() as a constexpr int.
    // simdjson provides SIMDJSON_VERSION as a macro.
    std::cout << " Boot OK | Boost.Beast " << BOOST_BEAST_VERSION << " | simdjson " << SIMDJSON_VERSION << "\n";
}
}  // namespace

// =============================================================================
// VRAM budget parsing + graph sizing
// =============================================================================
//
// `HFT_Engine.exe`           → defaults to 12 GiB VRAM
// `HFT_Engine.exe 12G`       → explicit 12 GiB
// `HFT_Engine.exe 4096M`     → 4 GiB
// `HFT_Engine.exe 8589934592`→ raw bytes
//
// Per-vertex VRAM cost (3 CSR arrays, double-weighted):
//   row_offsets : (V + 1) * 8  bytes
//   col_indices : V * E * 4   bytes
//   edge_weights: V * E * 8   bytes
// With E = 150 ( default), that's 1808 * V bytes. At 12 GiB
// (12 * 1024^3 = 12884901888 bytes) we fit ~7.1M vertices.
//
// We leave a 5% headroom so the OS / driver / context overhead does
// not blow up the allocation.

static constexpr uint64_t DEFAULT_VRAM_BUDGET    = 12ULL * 1024 * 1024 * 1024;
static constexpr uint32_t DEFAULT_EDGES_PER_NODE = 150;
static constexpr double   VRAM_HEADROOM          = 0.95;
// : Sentinel value to toggle `vram_budget` into "dense-mode"
// (V=500, E/V=20). We select 0xD07E50 (DENSE little-endian) as a
// self-explanatory magic number.
[[maybe_unused]] static constexpr uint64_t DENSE_MODE_SENTINEL = 0xD07E50ULL;

static uint64_t parse_vram_budget(const char* s) {
    if (!s || !*s) return DEFAULT_VRAM_BUDGET;
    char*    end  = nullptr;
    uint64_t base = std::strtoull(s, &end, 10);
    if (end == s) {
        std::cerr << "[CLI] Could not parse VRAM budget '" << s << "', using default 12G.\n";
        return DEFAULT_VRAM_BUDGET;
    }
    if (*end == 'G' || *end == 'g') return base * 1024ULL * 1024ULL * 1024ULL;
    if (*end == 'M' || *end == 'm') return base * 1024ULL * 1024ULL;
    if (*end == 'K' || *end == 'k') return base * 1024ULL;
    return base;  // raw bytes
}

static uint64_t vertices_for_budget(uint64_t budget_bytes, uint32_t edges_per_node) {
    // 8 (row_off) + 4*E (col) + 8*E (w) ≈ 12*E + 8 bytes per vertex.
    uint64_t per_v  = 8ULL + 12ULL * edges_per_node;
    uint64_t usable = static_cast<uint64_t>(budget_bytes * VRAM_HEADROOM);
    return usable / per_v;
}

static int run_default_main(uint64_t vram_budget) {
    std::string_view script = "Weight < 2.5";
    Lexer            lexer(script);
    ASTArena         arena(1024);
    Parser           parser(lexer, arena);
    uint32_t         root = parser.parseExpression();
    Compiler         compiler(arena);
    compiler.compileNode(root);

    std::cout << "Bytecode generated. Instruction count: " << compiler.getBytecode().size() << "\n";

    VirtualMachine vm(compiler.getBytecode());
    double         edgeA_weight = 1.2;
    double         edgeB_weight = 3.0;

    bool resultA = vm.evaluate(edgeA_weight);
    bool resultB = vm.evaluate(edgeB_weight);

    std::cout << "Result for edge A: " << resultA << "\n";
    std::cout << "Result for edge B: " << resultB << "\n";

    // Launch CUDA test
    std::cout << "\n--- : GPU-Native Graph Generation & Benchmark ---\n";

    uint32_t EDGES_PER_NODE = DEFAULT_EDGES_PER_NODE;
    uint64_t NUM_NODES      = vertices_for_budget(vram_budget, EDGES_PER_NODE);
    if (NUM_NODES < 1024) {
        std::cerr << "[Config] VRAM budget too small (got " << NUM_NODES << " vertices). Bumping to 1024 minimum.\n";
        NUM_NODES = 1024;
    }
    if (NUM_NODES > static_cast<uint64_t>(INT32_MAX)) {
        // CSR uses int for num_vertices in many places.
        std::cerr << "[Config] Capping to INT32_MAX vertices.\n";
        NUM_NODES = INT32_MAX;
    }

    uint64_t total_edges = NUM_NODES * static_cast<uint64_t>(EDGES_PER_NODE);
    std::cout << "[Config] VRAM budget: " << (vram_budget / (1024 * 1024)) << " MiB\n";
    std::cout << "[Host] Graph config: " << NUM_NODES << " nodes, " << total_edges << " edges ("
              << (total_edges * (12ULL)) / (1024 * 1024) << " MiB CSR)\n";

    // --- STEP 1: Build row_offsets only on CPU (64-bit offsets!) ---
    auto                 t_start_offsets = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> row_offsets(NUM_NODES + 1);
    for (uint64_t i = 0; i < NUM_NODES; ++i) {
        row_offsets[i] = static_cast<int64_t>(i) * EDGES_PER_NODE;
    }
    row_offsets[NUM_NODES] = static_cast<int64_t>(total_edges);
    auto t_end_offsets     = std::chrono::high_resolution_clock::now();
    std::cout << "[CPU] Row offsets built: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_end_offsets - t_start_offsets).count()
              << " ms\n";

#ifdef HAS_CUDA
    // --- STEP 2: Allocate VRAM + copy row_offsets over PCIe ---
    auto        t_start_pci = std::chrono::high_resolution_clock::now();
    DeviceGraph d_graph =
        DeviceGraph::create_device_only(row_offsets, static_cast<int>(NUM_NODES), static_cast<int64_t>(total_edges));
    cudaDeviceSynchronize();
    auto t_end_pci = std::chrono::high_resolution_clock::now();
    std::cout << "[PCIe] VRAM alloc + row_offsets copy: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_end_pci - t_start_pci).count() << " ms\n";

    if (d_graph.d_col_indices == nullptr) {
        std::cerr << "[FATAL] VRAM allocation failed. Run with a smaller "
                  << "VRAM budget, e.g. `HFT_Engine.exe 4G`.\n";
        return 1;
    }

    // --- STEP 3: GPU generates col_indices and edge_weights directly ---
    auto t_start_gen = std::chrono::high_resolution_clock::now();
    launch_graph_generator(d_graph, EDGES_PER_NODE);
    auto t_end_gen = std::chrono::high_resolution_clock::now();
    std::cout << "[GPU] Graph generation kernel: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_end_gen - t_start_gen).count() << " ms\n";

    // --- STEP 4: Traverse kernel (same as before, now on GPU-native data) ---
    auto t_start_kernel = std::chrono::high_resolution_clock::now();
    launch_csr_test(d_graph);
    auto   t_end_kernel = std::chrono::high_resolution_clock::now();
    double kernel_ms    = std::chrono::duration<double, std::milli>(t_end_kernel - t_start_kernel).count();
    std::cout << "[GPU] Traverse kernel (read from VRAM): " << kernel_ms << " ms\n";

    // --- STEP 5: Verify integrity (check first few edges) ---
    int         h_col[10] = {0};
    double      h_w[10]   = {0.0};
    cudaError_t e1        = cudaMemcpy(h_col, d_graph.d_col_indices, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaError_t e2        = cudaMemcpy(h_w, d_graph.d_edge_weights, 10 * sizeof(double), cudaMemcpyDeviceToHost);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        std::cerr << "[Sanity] cudaMemcpy failed: " << cudaGetErrorString(e1) << " / " << cudaGetErrorString(e2)
                  << "\n";
    } else {
        std::cout << "[Host] Sanity check (first 3 edges):\n";
        for (int i = 0; i < 3; ++i) {
            std::cout << "  Edge " << i << ": target=" << h_col[i] << ", weight=" << h_w[i] << "\n";
        }
    }

    // Total GPU pipeline time
    double total_gpu_ms = std::chrono::duration<double, std::milli>(t_end_kernel - t_start_offsets).count();
    std::cout << "[Total] GPU-native pipeline: " << total_gpu_ms << " ms\n";
    std::cout << "[Savings] Old CPU pipeline took ~370,000 ms (6 min) for 2B edges.\n";
    std::cout << "[Savings] GPU pipeline avoids ~" << (total_edges * (32ULL + 8ULL)) / (1024ULL * 1024ULL)
              << " MB of host memory & PCIe traffic.\n";
#else
    std::cout << "[INFO] No CUDA hardware detected.\n";
#endif
    return 0;
}

// =============================================================================
// End-to-end pipeline : graph_generator → BF → arbitrage detection.
// Triggered by passing "e2e" as the first CLI argument. An optional second
// argument sets the VRAM budget (default 12G). For a tiny smoke test use
// "e2e 64M" or similar.
//
// : `ab_compare` executes both the V1 path AND the V2-Optimized path
// on identical graphs to benchmark latency and edge counts.
// =============================================================================
#ifdef HAS_CUDA
// : Dense topology as a dedicated default path. If `dense_mode`
// is active, V=500 and E_PER_NODE=20 are utilized — the entire graph
// fits into the L1/Shared Memory of a single SM.
struct GraphConfig {
    int         v          = 100'000;
    uint32_t    e_per_node = 8;
    const char* label      = "default";
};

static GraphConfig graph_config_default() {
    GraphConfig c;  // 100k / 8
    return c;
}

static GraphConfig graph_config_dense() {
    GraphConfig c;
    c.v          = 500;
    c.e_per_node = 20;
    c.label      = "dense-micro (V=500, E/V=20)";
    return c;
}

static void run_end_to_end_pipeline(uint64_t vram_budget, bool use_v2, bool ab_compare = false) {
    std::cout << "\n--- : End-to-End Pipeline ---\n";

    // : Default remains 100k/8 (large/sparse). Dense topology
    // is explicitly activated via the `--dense` flag (see argv parsing).
    const bool  dense_mode = (vram_budget == DENSE_MODE_SENTINEL);
    GraphConfig gc         = dense_mode ? graph_config_dense() : graph_config_default();

    // VRAM budget cap is only meaningful for the default topology.
    if (!dense_mode) {
        uint64_t V64 = vertices_for_budget(vram_budget, gc.e_per_node);
        if (V64 < 8) V64 = 8;
        // Cap so the Bellman-Ford O(V*E) work stays interactive. At V=100k,
        // E=8: 100k iters × 8 edges × 100k vertices = 8e10 ops — runs in
        // single-digit seconds on consumer GPUs. Larger graphs scale the
        // runtime quadratically with V.
        if (V64 > 100'000ULL) V64 = 100'000ULL;
        gc.v = static_cast<int>(V64);
    }

    const int      V          = gc.v;
    const uint32_t E_PER_NODE = gc.e_per_node;

    std::cout << "[Config] Topology: " << gc.label << "\n";
    std::cout << "[Config] V=" << V << ", E=" << (V * E_PER_NODE) << " (E_PER_NODE=" << E_PER_NODE << ")\n";
    if (ab_compare) {
        std::cout << "[Config] A/B comparison active\n";
    }

    // --- 1. Build row_offsets on host ---
    std::vector<int64_t> row_offsets(V + 1);
    for (int i = 0; i <= V; ++i) {
        row_offsets[i] = static_cast<int64_t>(i) * E_PER_NODE;
    }

    // --- 2. Allocate DeviceGraph and generate edges on GPU ---
    DeviceGraph dg = DeviceGraph::create_device_only(row_offsets, V, static_cast<int64_t>(V * E_PER_NODE));

    if (dg.d_col_indices == nullptr) {
        std::cerr << "[FATAL] VRAM allocation failed for e2e pipeline.\n";
        return;
    }

    auto t_gen_start = std::chrono::high_resolution_clock::now();
    launch_graph_generator(dg, E_PER_NODE);
    auto   t_gen_end = std::chrono::high_resolution_clock::now();
    double gen_ms    = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
    std::cout << "[GPU] Graph generation: " << gen_ms << " ms\n";

    // --- CSR to CSC Transpose (CPU -> GPU) ---
    std::vector<int>    h_col_indices(dg.num_edges);
    std::vector<double> h_edge_weights(dg.num_edges);
    cudaMemcpy(h_col_indices.data(), dg.d_col_indices, dg.num_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edge_weights.data(), dg.d_edge_weights, dg.num_edges * sizeof(double), cudaMemcpyDeviceToHost);
    const auto& h_row_offsets = row_offsets;

    std::vector<int64_t> h_csc_col_offsets(V + 1, 0);
    std::vector<int>     h_csc_row_indices(dg.num_edges);
    std::vector<float>   h_csc_edge_weights(dg.num_edges);
    std::vector<int64_t> h_csc_orig_idx(dg.num_edges);

    // Calculate incoming edges per node
    std::vector<int> in_degree(V, 0);
    for (int64_t i = 0; i < dg.num_edges; ++i) {
        in_degree[h_col_indices[i]]++;
    }

    // Prefix sum for offsets
    for (int i = 0; i < V; ++i) {
        h_csc_col_offsets[i + 1] = h_csc_col_offsets[i] + in_degree[i];
    }

    // Populate arrays
    std::vector<int64_t> current_offset = h_csc_col_offsets;
    for (int u = 0; u < V; ++u) {
        for (int64_t pos = h_row_offsets[u]; pos < h_row_offsets[u + 1]; ++pos) {
            int     v        = h_col_indices[pos];
            int64_t dest_pos = current_offset[v]++;

            h_csc_row_indices[dest_pos]  = u;
            h_csc_edge_weights[dest_pos] = static_cast<float>(h_edge_weights[pos]);
            h_csc_orig_idx[dest_pos]     = pos;  // Critical for flagging tracking!
        }
    }

    // GPU allocation and payload upload
    cudaMalloc(&dg.d_csc_col_offsets, (V + 1) * sizeof(int64_t));
    cudaMalloc(&dg.d_csc_row_indices, dg.num_edges * sizeof(int));
    cudaMalloc(&dg.d_csc_edge_weights, dg.num_edges * sizeof(float));
    cudaMalloc(&dg.d_csc_orig_idx, dg.num_edges * sizeof(int64_t));

    cudaMemcpy(dg.d_csc_col_offsets, h_csc_col_offsets.data(), (V + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dg.d_csc_row_indices, h_csc_row_indices.data(), dg.num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dg.d_csc_edge_weights, h_csc_edge_weights.data(), dg.num_edges * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dg.d_csc_orig_idx, h_csc_orig_idx.data(), dg.num_edges * sizeof(int64_t), cudaMemcpyHostToDevice);

    // --- 3. Run Bellman-Ford ---
    auto t_bf_start = std::chrono::high_resolution_clock::now();
    bool converged;
    if (use_v2 || ab_compare) {
        // V2 is a prerequisite for the V2-Optimized path. In A/B mode,
        // ALWAYS use V2.
        std::cout << "[GPU] Using V2 (per-block flags, single PCIe read)\n";
        converged = launch_bellman_ford_full_v2(dg, 0, V);
    } else {
        converged = launch_bellman_ford_full(dg, /*source=*/0, /*max_iters=*/V - 1);
    }
    auto   t_bf_end = std::chrono::high_resolution_clock::now();
    double bf_ms    = std::chrono::duration<double, std::milli>(t_bf_end - t_bf_start).count();
    std::cout << "[GPU] Bellman-Ford: " << bf_ms << " ms (converged=" << converged << ")\n";

    // --- 4. Run arbitrage detection ---
    if (ab_compare) {
        // A/B: V1 (original) vs. V2-Optimized (WPT=2 + Early-Term) vs.
        //      V2-Persistent (Cooperative Groups, 1 Launch)
        std::cout << "\n[A/B] V1-Arbitrage (original):\n";
        auto   t1_start = std::chrono::high_resolution_clock::now();
        auto   v1_edges = launch_arbitrage_detection(dg, /*source=*/0);
        auto   t1_end   = std::chrono::high_resolution_clock::now();
        double v1_ms    = std::chrono::duration<double, std::milli>(t1_end - t1_start).count();
        std::cout << "  V1: " << v1_ms << " ms, " << v1_edges.size() << " edges\n";

        std::cout << "[A/B] V2-Optimized (WPT=2 + Early-Term):\n";
        auto t2_start = std::chrono::high_resolution_clock::now();
        // V2-Optimized requires BF-V2 output (d_dist populated) and
        // converged=false (otherwise no cascade). Has V1 already
        // overwritten d_arbitrage_flags in dg? No, d_flags is a
        // distinct buffer per invocation. Proceed.
        bool   did_conv = converged;
        auto   v2_edges = phase7::optimized::launch_arbitrage_detection_optimized(dg, /*source=*/0, &did_conv);
        auto   t2_end   = std::chrono::high_resolution_clock::now();
        double v2_ms    = std::chrono::duration<double, std::milli>(t2_end - t2_start).count();
        std::cout << "  V2-Opt: " << v2_ms << " ms, " << v2_edges.size() << " edges\n";

        std::cout << "[A/B] V2-Persistent (Cooperative Groups):\n";
        auto   t3_start   = std::chrono::high_resolution_clock::now();
        bool   did_conv_p = converged;
        auto   v3_edges   = phase7::optimized::launch_arbitrage_detection_persistent(dg, /*source=*/0, &did_conv_p);
        auto   t3_end     = std::chrono::high_resolution_clock::now();
        double v3_ms      = std::chrono::duration<double, std::milli>(t3_end - t3_start).count();
        std::cout << "  V2-Persistent: " << v3_ms << " ms, " << v3_edges.size() << " edges\n";

        // Benchmark comparison.
        if (v1_ms > 0) {
            double speedup_v2 = v1_ms / v2_ms;
            double speedup_v3 = v1_ms / v3_ms;
            std::cout << "\n[A/B] Speedup vs V1:"
                      << "  V2-Opt=" << speedup_v2 << "x"
                      << "  V2-Persistent=" << speedup_v3 << "x\n";
        }

        // : V4 Micro-Kernel (single block, shared memory).
        // Only practical in dense mode (V<=1024). In default mode
        // (V=100k), the wrapper falls back to Persistent; we
        // bypass the path here to eliminate ambiguity.
        if (V <= 1024) {
            std::cout << "\n[A/B] V4-Micro (single block, shared memory):\n";

            // : Execution Engine
            CSRGraph host_csr(static_cast<int>(V), dg.num_edges);
            host_csr.row_offsets = row_offsets;
            host_csr.col_indices = h_col_indices;
            phase7::ExecutionEngine engine(host_csr, "api.binance.com", "443");

            // Pre-allocation for micro-kernel
            ArbitrageOpportunity* d_opp = nullptr;
            ArbitrageOpportunity* h_opp = nullptr;
            cudaMalloc(&d_opp, sizeof(ArbitrageOpportunity));
            cudaMallocHost(&h_opp, sizeof(ArbitrageOpportunity));

            auto t4_start = std::chrono::high_resolution_clock::now();
            phase7::optimized::launch_arbitrage_detection_micro(dg, d_opp);
            cudaMemcpy(h_opp, d_opp, sizeof(ArbitrageOpportunity), cudaMemcpyDeviceToHost);
            auto t4_end = std::chrono::high_resolution_clock::now();

            // : Tick-to-Trade Execution
            engine.execute_cycles(h_opp);

            cudaFree(d_opp);
            cudaFreeHost(h_opp);

            double v4_ms = std::chrono::duration<double, std::milli>(t4_end - t4_start).count();
            std::cout << "  V4-Micro: " << v4_ms << " ms, " << h_opp->path_length << " edges (profit: " << h_opp->profit
                      << "%)\n";
            if (v1_ms > 0) {
                std::cout << "[A/B] Speedup V4-Micro vs V1: " << (v1_ms / v4_ms) << "x\n";
            }
        } else {
            std::cout << "[A/B] V4-Micro skipped (V=" << V << " > 1024 micro-limit)\n";
        }

        // Sanity: identical edge count?
        // WARNING: V1 and V2 INTERNALLY employ divergent BF versions!
        //   - launch_arbitrage_detection invokes launch_bellman_ford_full (V1-BF)
        //   - V2-Opt relies on the PRE-POPULATED d_dist from V2-BF
        // Edge counts may deviate - this is NOT a bug, but rather
        // divergent algorithms operating on discrete states.
        if (v1_edges.size() != v2_edges.size()) {
            std::cout << "[A/B] INFO: Edge counts differ! V1=" << v1_edges.size() << " V2-Opt=" << v2_edges.size()
                      << " (erwartet, da V1/V2-BF intern unterschiedlich)\n";
        } else {
            std::cout << "[A/B] OK: edge count match (" << v1_edges.size() << ")\n";
        }
    } else {
        // Single path (alter Code).
        auto                 t_arb_start = std::chrono::high_resolution_clock::now();
        std::vector<int64_t> arb_edges;
        if (use_v2) {
            arb_edges = launch_arbitrage_detection_v2(dg, /*source=*/0);
        } else {
            arb_edges = launch_arbitrage_detection(dg, /*source=*/0);
        }
        auto   t_arb_end = std::chrono::high_resolution_clock::now();
        double arb_ms    = std::chrono::duration<double, std::milli>(t_arb_end - t_arb_start).count();
        std::cout << "[GPU] Arbitrage detection: " << arb_ms << " ms\n";
        std::cout << "[Result] Arbitrage edges found: " << arb_edges.size() << "\n";

        // Print first few flagged edges, if any.
        if (!arb_edges.empty()) {
            std::vector<int> h_col(V * E_PER_NODE);
            cudaMemcpy(h_col.data(), dg.d_col_indices, h_col.size() * sizeof(int), cudaMemcpyDeviceToHost);
            size_t show = std::min<size_t>(5, arb_edges.size());
            std::cout << "[Sample] First " << show << " flagged edges:\n";
            for (size_t i = 0; i < show; ++i) {
                int64_t pos = arb_edges[i];
                std::cout << "  Edge " << pos << " -> node " << h_col[pos] << "\n";
            }
        }

        double total_ms = std::chrono::duration<double, std::milli>(t_arb_end - t_gen_start).count();
        std::cout << "[Total] End-to-end pipeline: " << total_ms << " ms\n";
    }
}
#endif  // HAS_CUDA

int main(int argc, char** argv) {
    phase7_boot_check();

    // argv[1] semantics:
    //   absent            → default 12G VRAM
    //   "e2e"             → e2e pipeline with default 12G
    //   "e2e <budget>"    → e2e pipeline with explicit budget
    //   "<budget>"        → default pipeline with explicit budget
    //   "-h" / "--help"   → usage

    auto print_usage = []() {
        std::cout << "Usage:\n"
                  << "  HFT_Engine               Run  demo (12 GiB VRAM)\n"
                  << "  HFT_Engine <budget>       demo with custom VRAM\n"
                  << "  HFT_Engine e2e <budget>  End-to-end with custom VRAM\n"
                  << "  HFT_Engine phase7-ws [symbol]  WebSocket client (Binance ticker, 10s)\n"
                  << "  HFT_Engine phase7-udp-sbe [ip] [port]  UDP Multicast SBE client\n"
                  << "\nVRAM budget format:\n"
                  << "  12G    = 12 GiB  (default)\n"
                  << "  4096M  = 4 GiB\n"
                  << "  8589934592  = 8 GiB raw bytes\n";
    };

    if (argc > 1 && (std::string_view(argv[1]) == "-h" || std::string_view(argv[1]) == "--help")) {
        print_usage();
        return 0;
    }

    if (argc > 1 && std::string_view(argv[1]) == "phase7-ws") {
        //  Smoke Test: Start WebSocket client, run for 10s,
        // output telemetry. Requires active internet connection to Binance.
        std::cout << "[Phase7-WS] Initializing SPSC Ringbuffer + WebSocket Client...\n";

        // SPSC Ringbuffer allocated on HEAP (2 MB, > Windows thread stack limit).
        auto ringbuffer = std::make_shared<WsClient::RingBuffer>();

        WsClient::Config cfg;
        // Target symbol from argv[2] (default: btcusdt).
        if (argc > 2) {
            std::string sym(argv[2]);
            cfg.target = "/ws/" + sym + "@ticker";
        }

        auto wal_ringbuffer = std::make_shared<phase7::wal::WalWriter::RingBuffer>();
        phase7::wal::WalWriter wal_writer("smoke_wal.bin", wal_ringbuffer);
        wal_writer.start();

        WsClient client(cfg, ringbuffer, wal_ringbuffer);
        client.run();

        // : Launch VRAMUpdater to stimulate the PCIe API timer.
        // Without this step, the latency probe remains inactive.
        phase7::VRAMUpdater vram_updater(ringbuffer);
        vram_updater.run();

        std::cout << "[Phase7-WS] Connected. Running for 10s...\n";
        std::this_thread::sleep_for(std::chrono::seconds(10));

        client.stop();
        vram_updater.stop();
        // Brief suspension to ensure threads join gracefully.
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        std::cout << "[Phase7-WS] ticks_pushed=" << client.ticks_pushed() << " drops_full=" << client.drops_full()
                  << " parse_errors=" << client.parse_errors() << "\n";

        // Retrieve sample ticks from the buffer for validation.
        MarketTick t{};
        int        samples = 0;
        while (ringbuffer->try_pop(t) && samples < 3) {
            std::cout << "[Sample] node_id=" << t.node_id << " bid=" << t.bid << " ask=" << t.ask
                      << " weight=" << t.weight << "\n";
            ++samples;
        }
        return 0;
    }

    if (argc > 1 && std::string_view(argv[1]) == "phase7-udp-sbe") {
        std::cout << "[Phase7-UDP] Initializing SPSC Ringbuffer + UDP Multicast Client...\n";

        auto ringbuffer = std::make_shared<UdpClient::RingBuffer>();

        UdpClient::Config cfg;
        cfg.host = (argc > 2) ? argv[2] : "224.0.0.1";
        cfg.port = (argc > 3) ? argv[3] : "12345";
        
        // Ensure some symbol is mapped if needed (just for test)
        cfg.symbol_to_node_id["BTCUSDT"] = 0;
        cfg.symbol_to_node_id["ETHUSDT"] = 1;

        UdpClient client(cfg, ringbuffer);
        client.run();

        std::cout << "[Phase7-UDP] Bound to " << cfg.host << ":" << cfg.port << ". Running for 10s...\n";
        std::this_thread::sleep_for(std::chrono::seconds(10));

        client.stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        std::cout << "[Phase7-UDP] ticks_pushed=" << client.ticks_pushed() << " drops_full=" << client.drops_full()
                  << " parse_errors=" << client.parse_errors() << "\n";

        return 0;
    }

    if (argc > 1 && std::string_view(argv[1]) == "phase7-live") {
#ifdef HAS_CUDA
        // =====================================================================
        // /8.6: Live Binance → Multi-Symbol → CSR → BF → Arbitrage.
        // =====================================================================
        // Pipeline:
        //   Beast WS (Combined-Stream) → SPSC → VRAMUpdater
        //                                 ↓
        //                            d_ring → d_tick_batch (D2D memcpy)
        //                                 ↓
        //                            tick_to_edge_update (LUT Override)
        //                                 ↓ mutates d_edge_weights[node_id]
        //                            (1 Hz) bellman_ford_full_v2
        //                                 ↓
        //                            arbitrage_detection_v2
        //
        // CLI:
        //   phase7-live                          -> 4 Default Symbols, 10s, 1s BF
        //   phase7-live 30                       -> 30s, 1s BF
        //   phase7-live 30 2000                  -> 30s, 2s BF
        //   phase7-live 30 2000 btcusdt,ethusdt  -> Explicit Symbols
        // =====================================================================

        int duration_s     = 10;
        int bf_interval_ms = 1000;
        if (argc > 2) duration_s = std::atoi(argv[2]);
        if (argc > 3) bf_interval_ms = std::atoi(argv[3]);
        if (duration_s <= 0) duration_s = 10;
        if (bf_interval_ms <= 0) bf_interval_ms = 1000;

        // Default: 8 high-frequency USDT cross pairs .
        // These are the most liquid spot pairs on Binance.
        // With 8 symbols we yield ~8-16 ticks/s aggregate, each carrying
        // a dedicated node_id override in the LUT.
        std::vector<std::string> symbols;
        if (argc > 4) {
            // Comma-separated list: "btcusdt,ethusdt,solusdt,xrpusdt,dogeusdt,adausdt,avaxusdt,linkusdt"
            std::string list(argv[4]);
            size_t      pos = 0;
            while (true) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) {
                    symbols.push_back(list.substr(pos));
                    break;
                }
                symbols.push_back(list.substr(pos, comma - pos));
                pos = comma + 1;
            }
        } else {
            symbols = {"btcusdt", "ethusdt", "solusdt", "xrpusdt", "dogeusdt", "ethbtc", "solbtc", "xrpbtc", "dogebtc"};
        }

        // : --persistent enforces HW barrier path (cg::this_grid().sync()).
        // Default: persistent (faster + correct on Blackwell sm_120).
        bool use_persistent = true;
        for (int i = 1; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--persistent") use_persistent = true;
            if (std::string_view(argv[i]) == "--v2-opt") use_persistent = false;
        }

        std::cout << "[Phase7-Live] Duration=" << duration_s << "s, BF-Interval=" << bf_interval_ms
                  << "ms, arb=" << (use_persistent ? "persistent-HW" : "v2-opt") << ", Symbole=[";
        for (size_t i = 0; i < symbols.size(); ++i) {
            std::cout << symbols[i] << (i + 1 < symbols.size() ? "," : "");
        }
        std::cout << "]\n";

        // : Real Order Book Topology
        constexpr uint32_t V         = 6;
        constexpr int64_t  NUM_EDGES = 18;  // 9 pairs * 2 directed edges

        // 0: USDT, 1: BTC, 2: ETH, 3: SOL, 4: XRP, 5: DOGE

        // Define edges: pairs of (source, target)
        std::vector<std::pair<int, int>> edge_list = {
            {0, 1},
            {1, 0},  // BTCUSDT
            {0, 2},
            {2, 0},  // ETHUSDT
            {0, 3},
            {3, 0},  // SOLUSDT
            {0, 4},
            {4, 0},  // XRPUSDT
            {0, 5},
            {5, 0},  // DOGEUSDT
            {1, 2},
            {2, 1},  // ETHBTC
            {1, 3},
            {3, 1},  // SOLBTC
            {1, 4},
            {4, 1},  // XRPBTC
            {1, 5},
            {5, 1}  // DOGEBTC
        };

        // --- 1. Construct CSR graph ---
        std::vector<int64_t> row_offsets(V + 1, 0);
        std::vector<int>     col_indices;
        col_indices.reserve(NUM_EDGES);

        // CSR format requires sorted edges by source
        // Let's build adj list first
        std::vector<std::vector<int>> adj(V);
        for (int i = 0; i < V; ++i) {
            for (const auto& e : edge_list) {
                if (e.first == i) adj[i].push_back(e.second);
            }
        }

        for (int i = 0; i < V; ++i) {
            row_offsets[i] = col_indices.size();
            for (int target : adj[i]) {
                col_indices.push_back(target);
            }
        }
        row_offsets[V] = col_indices.size();

        DeviceGraph dg = DeviceGraph::create_device_only(row_offsets, static_cast<int>(V), NUM_EDGES);
        if (dg.d_col_indices == nullptr) {
            std::cerr << "[FATAL] VRAM allocation failed for phase7-live.\n";
            return 1;
        }
        cudaMemcpy(dg.d_col_indices, col_indices.data(), NUM_EDGES * sizeof(int), cudaMemcpyHostToDevice);

        std::cout << "[Phase7-Live] Graph allocated: V=" << V << ", edges=" << NUM_EDGES << "\n";
        std::cout << "[Phase7-Live] Graph generated (real topology)\n";

        // --- 1.1 CSC Construction for V4-Micro ---
        std::vector<int64_t> h_csc_col_offsets(V + 1, 0);
        std::vector<int>     h_csc_row_indices(NUM_EDGES, 0);
        std::vector<float>   h_csc_edge_weights(NUM_EDGES, 0.0f);
        std::vector<int64_t> h_csc_orig_idx(NUM_EDGES, 0);
        std::vector<int64_t> h_orig_to_csc_idx(NUM_EDGES, 0);

        std::vector<int> h_col = col_indices;

        std::vector<int> in_degree(V, 0);
        for (int64_t i = 0; i < NUM_EDGES; ++i) {
            in_degree[h_col[i]]++;
        }
        for (uint32_t v = 0; v < V; ++v) {
            h_csc_col_offsets[v + 1] = h_csc_col_offsets[v] + in_degree[v];
        }
        std::vector<int64_t> current_offset = h_csc_col_offsets;
        for (uint32_t u = 0; u < V; ++u) {
            int64_t start = row_offsets[u];
            int64_t end   = row_offsets[u + 1];
            for (int64_t i = start; i < end; ++i) {
                int     v                   = h_col[i];
                int64_t dest_pos            = current_offset[v]++;
                h_csc_row_indices[dest_pos] = u;
                // Find original edge index
                int orig_idx = -1;
                for (int k = 0; k < static_cast<int>(edge_list.size()); ++k) {
                    if (edge_list[k].first == static_cast<int>(u) && edge_list[k].second == v) {
                        orig_idx = k;
                        break;
                    }
                }
                h_csc_orig_idx[dest_pos]    = orig_idx;
                h_orig_to_csc_idx[orig_idx] = dest_pos;
            }
        }

        cudaMalloc(&dg.d_csc_col_offsets, (V + 1) * sizeof(int64_t));
        cudaMalloc(&dg.d_csc_row_indices, NUM_EDGES * sizeof(int));
        cudaMalloc(&dg.d_csc_edge_weights, NUM_EDGES * sizeof(float));
        cudaMalloc(&dg.d_csc_orig_idx, NUM_EDGES * sizeof(int64_t));
        cudaMalloc(&dg.d_orig_to_csc_idx, NUM_EDGES * sizeof(int64_t));

        cudaMemcpy(dg.d_csc_col_offsets, h_csc_col_offsets.data(), (V + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dg.d_csc_row_indices, h_csc_row_indices.data(), NUM_EDGES * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dg.d_csc_edge_weights, h_csc_edge_weights.data(), NUM_EDGES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dg.d_csc_orig_idx, h_csc_orig_idx.data(), NUM_EDGES * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dg.d_orig_to_csc_idx, h_orig_to_csc_idx.data(), NUM_EDGES * sizeof(int64_t), cudaMemcpyHostToDevice);

        // --- 1.2 Execution Engine Setup ---
        ArbitrageOpportunity* d_opp = nullptr;
        cudaMalloc(&d_opp, sizeof(ArbitrageOpportunity));
        ArbitrageOpportunity* h_opp = nullptr;
        cudaMallocHost(&h_opp, sizeof(ArbitrageOpportunity));

        CSRGraph host_csr;
        host_csr.num_vertices = V;
        host_csr.num_edges    = NUM_EDGES;
        host_csr.row_offsets  = row_offsets;
        host_csr.col_indices  = h_col;
        host_csr.edge_weights.resize(NUM_EDGES, 0.0);

        phase7::ExecutionEngine engine(host_csr, "api.binance.com", "443");

        // --- 2. SPSC + VRAMUpdater + WS-Client (Multi-Symbol + LUT) ---
        auto ringbuffer = std::make_shared<WsClient::RingBuffer>();
        WsClient::Config ws_cfg;

        // Symbol -> node_id (Here: node_id is the base edge index of the pair)
        // btcusdt -> Edge 0 (and 1)
        // ethusdt -> Edge 2 (and 3)
        // solusdt -> Edge 4 (and 5)
        // ...
        std::string target = "/stream?streams=";
        for (size_t i = 0; i < symbols.size(); ++i) {
            ws_cfg.symbol_to_node_id[symbols[i]] = static_cast<std::uint32_t>(i * 2);
            target += symbols[i] + "@ticker";
            if (i + 1 < symbols.size()) target += "/";
        }
        ws_cfg.target = target;
        std::cout << "[Phase7-Live] WS-Endpoint: " << target << "\n";
        // : Pin WS client to logical core 2.
        ws_cfg.io_core_id = 2;
        std::cout << "[Phase7-Live] LUT: ";
        for (const auto& [sym, nid] : ws_cfg.symbol_to_node_id) {
            std::cout << sym << "->" << nid << " ";
        }
        std::cout << "\n";

        auto wal_ringbuffer = std::make_shared<phase7::wal::WalWriter::RingBuffer>();
        phase7::wal::WalWriter wal_writer("live_wal.bin", wal_ringbuffer);
        wal_writer.start(3); // Pin WAL to core 3 for I/O

        WsClient ws_client(ws_cfg, ringbuffer, wal_ringbuffer);
        ws_client.run();
        std::cout << "[Phase7-Live] WS client started\n";

        // : Pinning VRAMUpdater worker to logical core 4
        // (separate CCD on 16-core systems like RTX 5070 Ti hosts,
        // prevents L1/L2 contention with WS client on Core 2 and
        // main thread on Core 1).
        phase7::VRAMRingConfig vram_cfg{};
        vram_cfg.worker_core_id = 4;
        // PCIe coalescing defaults remain: hiw=128 ticks (4 KB),
        // timeout=1 ms. Both documented in the header.
        phase7::VRAMUpdater vram_updater(ringbuffer, vram_cfg);
        vram_updater.run();
        std::cout << "[Phase7-Live] VRAMUpdater gestartet (pinned core=4, "
                     "hiw=128, timeout=1ms)\n";

        // : Pinning the main scheduler to Core 1 (sibling of
        // Core 0 = OS thread). Prevents cudaStream/cudaEvent synchronizations
        // of the BF launches from causing cache contention with the WS-path.
        ::phase7::pin_current_thread_to_core(1);

        // --- 3. d_tick_batch Buffer ---
        constexpr std::size_t BATCH_CAP    = 256;
        MarketTick*           d_tick_batch = nullptr;
        cudaError_t           ce           = cudaMalloc(&d_tick_batch, BATCH_CAP * sizeof(MarketTick));
        if (ce != cudaSuccess) {
            std::cerr << "[FATAL] d_tick_batch alloc failed: " << cudaGetErrorString(ce) << "\n";
            return 1;
        }

        // --- 4. Pipeline Execution Loop with BF Cooldown ---
        using clock          = std::chrono::steady_clock;
        const auto t_start   = clock::now();
        const auto t_end     = t_start + std::chrono::seconds(duration_s);
        auto       t_next_bf = t_start + std::chrono::milliseconds(bf_interval_ms);

        std::uint64_t total_ticks_processed = 0;
        int           bf_runs               = 0;
        int           total_opps            = 0;
        std::uint64_t last_seen_head        = 0;

        // : cudaEvents for deterministic hardware-level timing.
        // e_update_* measures the accumulated Update kernel execution time per
        // cooldown window (may encompass multiple launches). e_bf_* measures
        // the complete bellman_ford_full_v2 run. e_arb_* records
        // arbitrage_detection_v2 execution (only if divergence is detected).
        cudaEvent_t e_update_start, e_update_end, e_bf_start, e_bf_end, e_arb_start, e_arb_end;
        cudaEventCreate(&e_update_start);
        cudaEventCreate(&e_update_end);
        cudaEventCreate(&e_bf_start);
        cudaEventCreate(&e_bf_end);
        cudaEventCreate(&e_arb_start);
        cudaEventCreate(&e_arb_end);

        // Accumulated execution latencies across all BF dispatches (in milliseconds).
        double total_update_ms = 0.0;
        double total_bf_ms     = 0.0;
        double total_arb_ms    = 0.0;
        bool   update_started_ = false;  // Reinitialized per BF-cycle

        std::cout << "[Phase7-Live] Pipeline laeuft " << duration_s << "s...\n";

        while (clock::now() < t_end) {
            // Continuous polling — Binance transmits ~1 tick/s, necessitating
            // instantaneous reaction latencies.
            _mm_pause();
            std::uint64_t new_count = vram_updater.ticks_pushed_to_vram();
            if (new_count == last_seen_head) {
                continue;
            }
            std::uint64_t n = new_count - last_seen_head;
            // Exhaustively process ALL queued ticks rather than clamping
            // to BATCH_CAP. If > BATCH_CAP, chunk into sequential updates.
            while (n > 0) {
                std::uint64_t batch_n = (n > BATCH_CAP) ? BATCH_CAP : n;

                // (b) VRAM-Ring → d_tick_batch via DeviceToDevice-Memcpy.
                //     Inherently ROBUST against data races with the Updater-thread:
                //     the counter increments monotonically and the memory copy
                //     targets strictly up to the latest commited position.
                //     Ring-buffer wrap-around logic: if the pending 'n' ticks
                //     exceed the contiguous tail capacity, split into two Memcpys.
                MarketTick*         d_ring    = vram_updater.device_ring_ptr();
                const std::size_t   cap       = vram_updater.device_ring_capacity();
                const std::uint64_t start_pos = (last_seen_head + (new_count - last_seen_head - n)) % cap;

                // Synchronize to guarantee the Updater-cudaMemcpyAsync completes
                // BEFORE initiating ring extraction.
                //
                //  final: cudaDeviceSynchronize (blocking,
                // but executed strictly once per BF-cycle if n>0). Consistent
                // with  specifications (5.37 ms max PCIe variance).
                // Prior iterations utilizing per-Memcpy Event-Syncs induced
                // 30-80 ms latency spikes due to Lock-Contention with the Updater stream.
                cudaDeviceSynchronize();

                const std::size_t chunk_a =
                    (std::min)(static_cast<std::size_t>(batch_n), cap - static_cast<std::size_t>(start_pos));
                const std::size_t chunk_b = static_cast<std::size_t>(batch_n) - chunk_a;

                ce = cudaMemcpy(
                    d_tick_batch, d_ring + start_pos, chunk_a * sizeof(MarketTick), cudaMemcpyDeviceToDevice);
                if (ce != cudaSuccess) {
                    std::cerr << "[Phase7-Live] d_tick_batch memcpy A failed: " << cudaGetErrorString(ce) << "\n";
                    break;
                }
                if (chunk_b > 0) {
                    ce = cudaMemcpy(
                        d_tick_batch + chunk_a, d_ring, chunk_b * sizeof(MarketTick), cudaMemcpyDeviceToDevice);
                    if (ce != cudaSuccess) {
                        std::cerr << "[Phase7-Live] d_tick_batch memcpy B failed: " << cudaGetErrorString(ce) << "\n";
                        break;
                    }
                }

                // (c) Tick → CSC edge_weight dispatch
                // : Record the start-Event strictly BEFORE the FIRST
                // Update-Kernel within the active cooldown window. update_started_
                // resets systematically every BF cycle.
                if (!update_started_) {
                    cudaEventRecord(e_update_start, 0);
                    update_started_ = true;
                }
                ce = phase7::launch_tick_to_edge_update(
                    d_tick_batch, static_cast<int>(batch_n), dg.d_csc_edge_weights, dg.d_orig_to_csc_idx, NUM_EDGES);
                if (ce != cudaSuccess) {
                    std::cerr << "[Phase7-Live] tick_to_edge_update failed: " << cudaGetErrorString(ce) << "\n";
                    break;
                }
                // Record the end-Event DIRECTLY subsequent to the final Update-Kernel,
                // ensuring the recorded duration exclusively measures kernel execution
                // and perfectly excludes polling wait times.
                cudaEventRecord(e_update_end, 0);
                total_ticks_processed += batch_n;
                n -= batch_n;
            }
            last_seen_head = new_count;

            // (d) BF-Cooldown: Dispatch a full run exactly once per bf_interval_ms.
            auto t_now = clock::now();
            if (t_now >= t_next_bf) {
                // : update_end event already successfully recorded post-
                // Update-Kernel. Transition directly into BF execution.

                // BF-Start (V4-Micro)
                cudaEventRecord(e_bf_start, 0);
                phase7::optimized::launch_arbitrage_detection_micro(dg, d_opp);
                cudaEventRecord(e_bf_end, 0);

                // Fetch result
                cudaMemcpy(h_opp, d_opp, sizeof(ArbitrageOpportunity), cudaMemcpyDeviceToHost);

                bool converged = (h_opp->path_length == 0);

                // Arbitrage-Dispatch
                if (!converged) {
                    cudaEventRecord(e_arb_start, 0);
                    engine.execute_cycles(h_opp);
                    cudaEventRecord(e_arb_end, 0);
                }

                // Synchronize (1x per BF-cycle) and extract elapsed timings.
                cudaEventSynchronize(e_bf_end);
                if (update_started_) cudaEventSynchronize(e_update_end);
                if (!converged) cudaEventSynchronize(e_arb_end);

                float      update_ms_f = 0.0f, bf_ms_f = 0.0f, arb_ms_f = 0.0f;
                const bool had_updates = update_started_;  // cache status prior to reset
                if (had_updates) {
                    cudaEventElapsedTime(&update_ms_f, e_update_start, e_update_end);
                    total_update_ms += update_ms_f;
                }
                cudaEventElapsedTime(&bf_ms_f, e_bf_start, e_bf_end);
                total_bf_ms += bf_ms_f;
                if (!converged) {
                    cudaEventElapsedTime(&arb_ms_f, e_arb_start, e_arb_end);
                    total_arb_ms += arb_ms_f;
                }

                // Reinitialize BF-Cooldown parameters for the upcoming cycle.
                update_started_ = false;

                ++bf_runs;
                total_opps += (converged ? 0 : 1);

                // : Stdout BF metrics containing all 3 GPU execution timings.
                // : Integrating ingestion & PCIe snapshots per dispatch.
                std::cout << "[BF " << bf_runs << "]"
                          << " upd=" << (had_updates ? update_ms_f : 0.0f) << "ms"
                          << " micro=" << bf_ms_f << "ms"
                          << " dispatch=" << (converged ? 0.0f : arb_ms_f) << "ms"
                          << " conv=" << (converged ? "Y" : "N") << " | CPU: ing_mean="
                          << (ws_client.ingest_count() > 0 ? static_cast<int>(ws_client.ingest_mean_ns()) : 0)
                          << "ns ing_max=" << ws_client.ingest_max_ns() << "ns"
                          << " pcie_mean="
                          << (vram_updater.pcie_count() > 0 ? static_cast<int>(vram_updater.pcie_mean_ns()) : 0)
                          << "ns pcie_max=" << vram_updater.pcie_max_ns() << "ns";

                if (!converged) {
                    std::cout << "\n  -> CYCLE FOUND! Path len: " << h_opp->path_length << ", Profit: " << h_opp->profit
                              << "%";
                }
                std::cout << "\n";

                t_next_bf = t_now + std::chrono::milliseconds(bf_interval_ms);
            }
        }

        // --- 5. Graceful Shutdown ---
        cudaDeviceSynchronize();
        ws_client.stop();
        vram_updater.stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        std::cout << "\n[Phase7-Live] === Shutdown ===\n";
        std::cout << "Ticks processed (Update-Kernel): " << total_ticks_processed << "\n";
        std::cout << "BF runs: " << bf_runs << "\n";
        std::cout << "Total opportunities found: " << total_opps << "\n";

        // : Aggregated GPU execution times across all BF cycles.
        // WARNING: total_update_ms encompasses total latency between e_update_start
        // (prior to the first update) and e_update_end (post final update) on
        // the default stream. This intrinsically incorporates cudaDeviceSync + D2D
        // Memcpys. For PURE Update-Kernel profiling, employ nsys/ncu traces.
        // total_bf_ms / total_arb_ms provide higher fidelity (isolated events).
        std::cout << "GPU times (accumulated over " << bf_runs << " BF runs):\n";
        std::cout << "  Update-Kernel (incl. Sync+D2D, total): " << total_update_ms << " ms\n";
        std::cout << "  Bellman-Ford-Micro (total): " << total_bf_ms << " ms\n";
        std::cout << "  Arbitrage-Dispatch (total): " << total_arb_ms << " ms\n";
        if (bf_runs > 0) {
            std::cout << "  Per BF-run: update=" << total_update_ms / bf_runs << "ms  bf=" << total_bf_ms / bf_runs
                      << "ms  arb=" << total_arb_ms / bf_runs << "ms\n";
        }

        std::cout << "WS client: pushed=" << ws_client.ticks_pushed() << " parse_errors=" << ws_client.parse_errors()
                  << " drops_full=" << ws_client.drops_full() << " unknown_sym=" << ws_client.unknown_symbol_drops()
                  << "\n";
        std::cout << "VRAMUpdater: pushed=" << vram_updater.ticks_pushed_to_vram()
                  << " batches=" << vram_updater.batches_pushed()
                  << " coalesce_early=" << vram_updater.coalesce_early_flushes()
                  << " coalesce_timeout=" << vram_updater.coalesce_timeout_flushes()
                  << " coalesce_total=" << vram_updater.coalesce_total_flushes() << "\n";

        // =====================================================================
        // : Comprehensive Latency Distribution Report (all 3 probes).
        // Time buckets: 0=<1us 1=<10us 2=<100us 3=<1ms 4=<10ms 5=>=10ms
        // =====================================================================
        std::cout << "\n[Latency Report]\n";

        // (1) Ingestion telemetry probe
        std::cout << "  Ingestion (parse+LUT+push, WS-Thread):\n";
        if (ws_client.ingest_count() > 0) {
            std::cout << "    n=" << ws_client.ingest_count()
                      << " mean=" << static_cast<int>(ws_client.ingest_mean_ns()) << "ns"
                      << " max=" << ws_client.ingest_max_ns() << "ns"
                      << " hist=[";
            for (std::size_t i = 0; i < WsClient::kIngestBuckets; ++i) {
                std::cout << ws_client.ingest_bucket(i)
                          << ((i + 1 < WsClient::kIngestBuckets) ? "," : "");
            }
            std::cout << "] (<1us,<10us,<100us,<1ms,<10ms,>=10ms)\n";
        } else {
            std::cout << "    no samples\n";
        }

        // (2) PCIe API telemetry probe
        std::cout << "  PCIe API (cudaMemcpyAsync submission, Updater-Thread):\n";
        if (vram_updater.pcie_count() > 0) {
            std::cout << "    n=" << vram_updater.pcie_count()
                      << " mean=" << static_cast<int>(vram_updater.pcie_mean_ns()) << "ns"
                      << " max=" << vram_updater.pcie_max_ns() << "ns"
                      << " bytes=" << vram_updater.pcie_bytes_total() << " hist=[";
            for (std::size_t i = 0; i < phase7::VRAMUpdater::kPcieBuckets; ++i) {
                std::cout << vram_updater.pcie_bucket(i) << ((i + 1 < phase7::VRAMUpdater::kPcieBuckets) ? "," : "");
            }
            std::cout << "] (<1us,<10us,<100us,<1ms,<10ms,>=10ms)\n";
        } else {
            std::cout << "    no samples\n";
        }

        // (3) GPU telemetry probe (cudaEvent)
        std::cout << "  GPU (cudaEvent, hard-wall on default stream):\n";
        std::cout << "    update (incl. Sync+D2D): " << total_update_ms << " ms total"
                  << " / " << bf_runs << " runs = " << (bf_runs > 0 ? total_update_ms / bf_runs : 0.0) << " ms/run\n";
        std::cout << "    bellman_ford_full_v2:    " << total_bf_ms << " ms total"
                  << " / " << bf_runs << " runs = " << (bf_runs > 0 ? total_bf_ms / bf_runs : 0.0) << " ms/run\n";
        std::cout << "    arbitrage_detection_v2:  " << total_arb_ms << " ms total"
                  << " / " << bf_runs << " runs = " << (bf_runs > 0 ? total_arb_ms / bf_runs : 0.0) << " ms/run\n";

        // (4) End-to-End Wall-Clock Pipeline Latency
        if (bf_runs > 0) {
            const double per_run = total_update_ms / bf_runs + total_bf_ms / bf_runs + total_arb_ms / bf_runs;
            std::cout << "  GPU per-Run (sum): " << per_run << " ms\n";
        }

        // : Deallocate cudaEvents.
        cudaEventDestroy(e_update_start);
        cudaEventDestroy(e_update_end);
        cudaEventDestroy(e_bf_start);
        cudaEventDestroy(e_bf_end);
        cudaEventDestroy(e_arb_start);
        cudaEventDestroy(e_arb_end);

        cudaFree(d_tick_batch);
        // dg dtor inherently dispatches cudaFree across all device buffers
        return 0;
#else
        std::cout << "[INFO] phase7-live requires CUDA. Skipping.\n";
        return 0;
#endif
    }

    if (argc > 1 && std::string_view(argv[1]) == "e2e") {
#ifdef HAS_CUDA
        // Parse optional --v2, --ab, --dense arguments identically decoupled
        // from positional order.
        //   --v2     = enforces V2-BF and V2-Arbitrage executions (default V1)
        //   --ab     = A/B-Benchmark evaluating V1-Arbitrage vs. V2-Optimized vs.
        //              V2-Persistent vs. Micro mapping over identical graph topologies.
        //              Implicitly overrides --v2.
        //   --dense  = Initializes dense Micro-Topology (V=500, E/V=20). Overrides
        //              VRAM-Budget explicitly to DENSE_MODE_SENTINEL.
        bool     use_v2_e2e = false;
        bool     ab_compare = false;
        bool     dense_mode = false;
        uint64_t budget     = DEFAULT_VRAM_BUDGET;
        for (int i = 2; i < argc; ++i) {
            std::string_view a(argv[i]);
            if (a == "--v2") {
                use_v2_e2e = true;
            } else if (a == "--ab") {
                ab_compare = true;
            } else if (a == "--dense") {
                dense_mode = true;
                budget     = DENSE_MODE_SENTINEL;
            } else {
                budget = parse_vram_budget(argv[i]);
            }
        }
        run_end_to_end_pipeline(budget, use_v2_e2e, ab_compare);
#else
        std::cout << "[INFO] e2e mode requires CUDA. Skipping.\n";
#endif
        return 0;
    }

    uint64_t budget = (argc > 1) ? parse_vram_budget(argv[1]) : DEFAULT_VRAM_BUDGET;
    return run_default_main(budget);
}
