#pragma once
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

struct Node;

struct Edge {
    std::shared_ptr<Node> to;
    double                weight;
};

struct Node {
    uint32_t          id;
    std::vector<Edge> edges;
};

class Graph {
private:
    std::unordered_map<uint32_t, std::shared_ptr<Node>> nodes;

public:
    void addNode(uint32_t id) {
        auto new_node = std::make_shared<Node>();
        new_node->id  = id;
        nodes[id]     = new_node;
    }

    void addEdge(uint32_t from_id, uint32_t to_id, double weight) {
        auto from_node = getNode(from_id);
        auto to_node   = getNode(to_id);
        if (from_node == nullptr || to_node == nullptr) {
            return;
        }

        Edge edge;
        edge.to     = to_node;
        edge.weight = weight;

        from_node->edges.push_back(edge);
    }

    std::shared_ptr<Node> getNode(uint32_t id) {
        if (nodes.find(id) != nodes.end()) {
            return nodes[id];
        }
        return nullptr;
    }

    const std::unordered_map<uint32_t, std::shared_ptr<Node>>& getNodes() const { return nodes; }
};