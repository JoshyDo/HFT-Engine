#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "types_dod.hpp"

class GraphArena {
private:
    std::vector<NodeDOD> nodes;
    std::vector<EdgeDOD> edges;
    uint32_t             active_nodes = 0;
    uint32_t             active_edges = 0;

public:
    GraphArena(uint32_t max_nodes, uint32_t max_edges) {
        nodes.resize(max_nodes);
        edges.resize(max_edges);
    }

    uint32_t addNode(uint32_t id) {
        if (active_nodes >= nodes.size()) {
            throw std::runtime_error("Max nodes exceeded");
        }
        nodes[active_nodes] = {id, active_edges, 0};
        return active_nodes++;
    }

    uint32_t addEdge(uint32_t from_node_index, uint32_t to_node_id, double bid, double ask) {
        if (active_edges >= edges.size()) {
            throw std::runtime_error("Max edges exceeded");
        }
        if (nodes[from_node_index].edgeCount == 0) {
            nodes[from_node_index].startEdgeIndex = active_edges;
        }
        edges[active_edges] = {bid, ask, -std::log(ask), to_node_id};
        nodes[from_node_index].edgeCount++;
        return active_edges++;
    }

    const std::vector<NodeDOD>& getNodes() const { return nodes; }
    const std::vector<EdgeDOD>& getEdges() const { return edges; }
    uint32_t                    getActiveNodeCount() const { return active_nodes; }
};