#pragma once
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "dsl-lexer.hpp"

enum class ASTNodeType { NUMBER, IDENTIFIER, BINARY_OP };

struct ASTNode {
    ASTNodeType type;
    Token       token;
    uint32_t    left;
    uint32_t    right;
};

class ASTArena {
private:
    std::vector<ASTNode> nodes;
    uint32_t             active_nodes = 0;

public:
    ASTArena(uint32_t max_nodes) { nodes.resize(max_nodes); }

    uint32_t allocateNode(ASTNodeType type, Token token, uint32_t left = UINT32_MAX, uint32_t right = UINT32_MAX) {
        if (active_nodes >= nodes.size()) {
            throw std::runtime_error("Max nodes exceeded");
        }
        nodes[active_nodes] = {type, token, left, right};
        return active_nodes++;
    }
    const std::vector<ASTNode>& getNodes() const { return nodes; }
};