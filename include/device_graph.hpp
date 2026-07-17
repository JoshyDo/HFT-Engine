#pragma once
#include "csr_graph.hpp"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

struct DeviceGraph {
    int64_t* d_row_offsets;
    int*     d_col_indices;
    double*  d_edge_weights;
    int      num_vertices;
    int64_t  num_edges;

    // Bellman-Ford state (allocated on demand by launch_bellman_ford_*).
    double* d_dist     = nullptr;  // size = num_vertices, edge weights of distances
    int*    d_modified = nullptr;  // size = 1, the global converged flag

    // Phase 6.1 per-block state (allocated on demand by V2 wrappers).
    int* d_block_modified = nullptr;  // size = num_blocks_v2, per-block flag
    int  num_blocks_v2    = 0;        // number of CUDA blocks in the V2 launch

    // CSC (Compressed Sparse Column) - Incoming edges for pull-model architecture
    int64_t* d_csc_col_offsets  = nullptr;
    int*     d_csc_row_indices  = nullptr;
    float*   d_csc_edge_weights = nullptr;
    int64_t* d_csc_orig_idx     = nullptr;  // Maps directly to d_arbitrage_flags position
    int64_t* d_orig_to_csc_idx  = nullptr;  // Maps original CSR Edge Index to CSC Index

    /// Full constructor: copies all CSR data (row_offsets, col_indices, edge_weights)
    /// from host to device. Use for CPU-generated graphs.
    DeviceGraph(const CSRGraph& host_graph) {
        num_vertices = host_graph.num_vertices;
        num_edges    = host_graph.num_edges;

#ifdef HAS_CUDA
        size_t bytes_offsets = (num_vertices + 1) * sizeof(int64_t);
        size_t bytes_indices = static_cast<size_t>(num_edges) * sizeof(int);
        size_t bytes_weights = static_cast<size_t>(num_edges) * sizeof(double);

        cudaMalloc((void**)&d_row_offsets, bytes_offsets);
        cudaMalloc((void**)&d_col_indices, bytes_indices);
        cudaMalloc((void**)&d_edge_weights, bytes_weights);

        cudaMemcpy(d_row_offsets, host_graph.row_offsets.data(), bytes_offsets, cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, host_graph.col_indices.data(), bytes_indices, cudaMemcpyHostToDevice);
        cudaMemcpy(d_edge_weights, host_graph.edge_weights.data(), bytes_weights, cudaMemcpyHostToDevice);
#endif
    }

    /// GPU-only constructor: allocates all buffers on device.
    /// row_offsets_host is copied to device; col_indices and edge_weights
    /// are allocated but left for a GPU kernel to fill.
    /// Use after calling launch_graph_generator().
    static DeviceGraph create_device_only(const std::vector<int64_t>& row_offsets_host, int vertices, int64_t edges) {
        (void)row_offsets_host;
        DeviceGraph dg;
        dg.num_vertices = vertices;
        dg.num_edges    = edges;

#ifdef HAS_CUDA
        size_t bytes_offsets = (vertices + 1) * sizeof(int64_t);
        size_t bytes_indices = static_cast<size_t>(edges) * sizeof(int);
        size_t bytes_weights = static_cast<size_t>(edges) * sizeof(double);

        // We use cudaMalloc directly (not the CUDA_OR_DIE macro) because this
        // header has no <cstdio> dependency. Callers that hit an OOM here will
        // see the cudaError_t printed to stderr from the gpu_check macro below.
        cudaError_t e1 = cudaMalloc((void**)&dg.d_row_offsets, bytes_offsets);
        cudaError_t e2 = cudaMalloc((void**)&dg.d_col_indices, bytes_indices);
        cudaError_t e3 = cudaMalloc((void**)&dg.d_edge_weights, bytes_weights);
        if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
            fprintf(stderr,
                    "[DeviceGraph] cudaMalloc failed: row_offsets=%s, "
                    "col_indices=%s, edge_weights=%s. Requested "
                    "%zu + %zu + %zu bytes (%zu MiB) for V=%d E=%lld.\n",
                    cudaGetErrorString(e1),
                    cudaGetErrorString(e2),
                    cudaGetErrorString(e3),
                    bytes_offsets,
                    bytes_indices,
                    bytes_weights,
                    (bytes_offsets + bytes_indices + bytes_weights) / (1024 * 1024),
                    vertices,
                    static_cast<long long>(edges));
            // Free whatever did succeed, then return a half-baked DeviceGraph
            // whose d_* pointers are nullptr. Downstream cudaMemcpy/launch_*
            // calls will then fail loudly (kernel launch error) instead of
            // silently reading 0xCDCDCDCD from uninitialised host memory.
            if (e1 == cudaSuccess) cudaFree(dg.d_row_offsets);
            if (e2 == cudaSuccess) cudaFree(dg.d_col_indices);
            if (e3 == cudaSuccess) cudaFree(dg.d_edge_weights);
            dg.d_row_offsets  = nullptr;
            dg.d_col_indices  = nullptr;
            dg.d_edge_weights = nullptr;
            return dg;
        }

        cudaMemcpy(dg.d_row_offsets, row_offsets_host.data(), bytes_offsets, cudaMemcpyHostToDevice);
// col_indices and edge_weights are uninitialized — kernel fills them
#endif

        return dg;
    }

    ~DeviceGraph() {
#ifdef HAS_CUDA
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_edge_weights);
        cudaFree(d_dist);
        cudaFree(d_modified);
        cudaFree(d_block_modified);
#endif
    }

    // Delete copy constructor and assignment operator to prevent accidental copies
    DeviceGraph(const DeviceGraph&)            = delete;
    DeviceGraph& operator=(const DeviceGraph&) = delete;

    // Allow move construction (needed for create_device_only return)
    DeviceGraph(DeviceGraph&& other) noexcept
        : d_row_offsets(other.d_row_offsets),
          d_col_indices(other.d_col_indices),
          d_edge_weights(other.d_edge_weights),
          num_vertices(other.num_vertices),
          num_edges(other.num_edges),
          d_dist(other.d_dist),
          d_modified(other.d_modified),
          d_block_modified(other.d_block_modified),
          num_blocks_v2(other.num_blocks_v2) {
        other.d_row_offsets    = nullptr;
        other.d_col_indices    = nullptr;
        other.d_edge_weights   = nullptr;
        other.d_dist           = nullptr;
        other.d_modified       = nullptr;
        other.d_block_modified = nullptr;
        other.num_blocks_v2    = 0;
        other.num_vertices     = 0;
        other.num_edges        = 0;
    }
    // Default constructor (zero-init all device pointers). Useful for
    // member declarations where the value will be overwritten via
    // assignment or move. MSVC requires this to be accessible.
    DeviceGraph()
        : d_row_offsets(nullptr),
          d_col_indices(nullptr),
          d_edge_weights(nullptr),
          num_vertices(0),
          num_edges(0),
          d_dist(nullptr),
          d_modified(nullptr),
          d_block_modified(nullptr),
          num_blocks_v2(0) {}

    DeviceGraph& operator=(DeviceGraph&& other) noexcept {
        if (this != &other) {
// Free our current resources.
#ifdef HAS_CUDA
            cudaFree(d_row_offsets);
            cudaFree(d_col_indices);
            cudaFree(d_edge_weights);
            cudaFree(d_dist);
            cudaFree(d_modified);
            cudaFree(d_block_modified);
#endif
            // Steal the other's resources.
            d_row_offsets    = other.d_row_offsets;
            d_col_indices    = other.d_col_indices;
            d_edge_weights   = other.d_edge_weights;
            d_dist           = other.d_dist;
            d_modified       = other.d_modified;
            d_block_modified = other.d_block_modified;
            num_blocks_v2    = other.num_blocks_v2;
            num_vertices     = other.num_vertices;
            num_edges        = other.num_edges;
            // Nullify the other.
            other.d_row_offsets    = nullptr;
            other.d_col_indices    = nullptr;
            other.d_edge_weights   = nullptr;
            other.d_dist           = nullptr;
            other.d_modified       = nullptr;
            other.d_block_modified = nullptr;
            other.num_blocks_v2    = 0;
            other.num_vertices     = 0;
            other.num_edges        = 0;
        }
        return *this;
    }
};

struct ArbitrageOpportunity {
    int     path_length;
    int64_t edge_indices[10];
    float   profit;
};