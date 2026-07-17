#pragma once
#include <cmath>
#include <limits>
#include <vector>

#include "arena.hpp"

struct SPFABuffer {
    std::vector<double>   distances;
    std::vector<uint32_t> queue;
    std::vector<bool>     in_queue;
    std::vector<uint32_t> count;

    SPFABuffer(uint32_t V) {
        distances.resize(V, std::numeric_limits<double>::infinity());
        queue.resize(V + 1);
        in_queue.resize(V, false);
        count.resize(V, 0);
    }
};

class BellmanFord {
public:
    static bool detectArbitrage(const GraphArena& arena, uint32_t source_id, SPFABuffer& buffer) {
        uint32_t V = arena.getActiveNodeCount();

        // 1. Reset buffer (recycle the old heap memory)
        std::fill(buffer.distances.begin(), buffer.distances.end(), std::numeric_limits<double>::infinity());
        std::fill(buffer.in_queue.begin(), buffer.in_queue.end(), false);
        std::fill(buffer.count.begin(), buffer.count.end(), 0);

        // 2. Local stack variables for pointers (free, 0 allocations)
        uint32_t head = 0;
        uint32_t tail = 0;

        // 3. Setup for the source node
        buffer.distances[source_id] = 0.0;
        buffer.queue[tail++]        = source_id;
        buffer.in_queue[source_id]  = true;
        buffer.count[source_id]     = 1;

        // --- SPFA (Shortest Path Faster Algorithm) ---

        while (head != tail) {
            uint32_t u         = buffer.queue[head];
            head               = (head + 1) % (V + 1);
            buffer.in_queue[u] = false;

            const auto& node = arena.getNodes()[u];

            for (uint32_t e = 0; e < node.edgeCount; ++e) {
                const auto& edge       = arena.getEdges()[node.startEdgeIndex + e];
                double      log_weight = edge.log_ask;
                uint32_t    v          = edge.to;

                if (buffer.distances[u] + log_weight < buffer.distances[v]) {
                    buffer.distances[v] = buffer.distances[u] + log_weight;

                    if (!buffer.in_queue[v]) {
                        buffer.queue[tail] = v;
                        tail               = (tail + 1) % (V + 1);
                        buffer.in_queue[v] = true;

                        buffer.count[v]++;
                        if (buffer.count[v] >= V) {
                            return true;  // Negative cycle detected
                        }
                    }
                }
            }
        }
        return false;  // No negative cycle detected
    }
};