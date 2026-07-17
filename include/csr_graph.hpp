#pragma once
#include <cstdint>
#include <vector>

#include "arena.hpp"

struct CSRGraph {
    int                  num_vertices;
    int64_t              num_edges;
    std::vector<int64_t> row_offsets;   // size = num_vertices + 1, 64-bit for >2B edges
    std::vector<int>     col_indices;   // size = num_edges
    std::vector<double>  edge_weights;  // size = num_edges

    CSRGraph() : num_vertices(0), num_edges(0) {}

    CSRGraph(int vertices, int64_t edges) : num_vertices(vertices), num_edges(edges) {
        row_offsets.resize(num_vertices + 1, 0);
        col_indices.resize(static_cast<size_t>(num_edges));
        edge_weights.resize(static_cast<size_t>(num_edges));
    }

    /// Converts the GraphArena-DOD (Phase 2) into the CSR format.
    /// Returns a flat, GPU-compatible graph.
    static CSRGraph convert_to_csr(const GraphArena& arena) {
        uint32_t V = arena.getActiveNodeCount();

        // --- 1. Count total edges ---
        uint64_t total_edges = 0;
        for (uint32_t i = 0; i < V; ++i) {
            total_edges += arena.getNodes()[i].edgeCount;
        }

        CSRGraph csr(static_cast<int>(V), static_cast<int64_t>(total_edges));

        // --- 2. Write per-vertex edge count into row_offsets ---
        for (uint32_t i = 0; i < V; ++i) {
            csr.row_offsets[i] = static_cast<int64_t>(arena.getNodes()[i].edgeCount);
        }

        // --- 3. Exclusive prefix sum (64-bit) ---
        int64_t running_sum = 0;
        for (uint32_t i = 0; i < V; ++i) {
            int64_t count      = csr.row_offsets[i];
            csr.row_offsets[i] = running_sum;
            running_sum += count;
        }
        csr.row_offsets[V] = running_sum;  // sentinel = total_edges

        // --- 4. Fill column indices and edge weights ---
        for (uint32_t i = 0; i < V; ++i) {
            const auto& node = arena.getNodes()[i];

            for (uint32_t e = 0; e < node.edgeCount; ++e) {
                const auto& edge                           = arena.getEdges()[node.startEdgeIndex + e];
                int64_t     pos                            = csr.row_offsets[i] + e;
                csr.col_indices[static_cast<size_t>(pos)]  = static_cast<int>(edge.to);
                csr.edge_weights[static_cast<size_t>(pos)] = edge.log_ask;
            }
        }

        return csr;
    }
};