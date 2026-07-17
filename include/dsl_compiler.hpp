#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dsl-ast.hpp"
#include "dsl-lexer.hpp"
#include "dsl-parser.hpp"

enum class Opcode { PUSH_NUMBER, PUSH_VAR, OP_GREATER, OP_LESS };

struct Instruction {
    Opcode op;
    double operand;
};

class Compiler {
    const ASTArena&          arena;
    std::vector<Instruction> bytecode;

public:
    Compiler(const ASTArena& ar) : arena(ar) {}

    void compileNode(uint32_t nodeIndex) {
        // load the node from the arena
        const ASTNode& node = arena.getNodes()[nodeIndex];

        if (node.type == ASTNodeType::NUMBER) {
            // convert Lexer to double
            double value = std::stod(std::string(node.token.lexeme));
            bytecode.push_back({Opcode::PUSH_NUMBER, value});
        } else if (node.type == ASTNodeType::IDENTIFIER) {
            // push: push_var and set to 0.0
            bytecode.push_back({Opcode::PUSH_VAR, 0.0});
        } else if (node.type == ASTNodeType::BINARY_OP) {
            // compile left and right nodes
            compileNode(node.left);
            compileNode(node.right);
            if (node.token.lexeme == ">") {
                bytecode.push_back({Opcode::OP_GREATER, 0.0});
            } else if (node.token.lexeme == "<") {
                bytecode.push_back({Opcode::OP_LESS, 0.0});
            } else {
                throw std::runtime_error("Unknown binary operator");
            }
        }
    }
    const std::vector<Instruction>& getBytecode() const { return bytecode; }
};