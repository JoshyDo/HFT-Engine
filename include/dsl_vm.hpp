#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "dsl-ast.hpp"
#include "dsl-lexer.hpp"
#include "dsl-parser.hpp"
#include "dsl_compiler.hpp"

class VirtualMachine {
    const std::vector<Instruction>& bytecode;

public:
    VirtualMachine(const std::vector<Instruction>& bc) : bytecode(bc) {}

    bool evaluate(double edge_weight) const {
        std::array<double, 32> stack;
        int                    sp = 0;
        for (const auto& instr : bytecode) {
            switch (instr.op) {
                case Opcode::PUSH_NUMBER:
                    stack[sp++] = instr.operand;
                    break;
                case Opcode::PUSH_VAR:
                    stack[sp++] = edge_weight;
                    break;
                case Opcode::OP_GREATER: {
                    if (sp < 2) return false;
                    double right = stack[--sp];
                    double left  = stack[--sp];
                    stack[sp++]  = (left > right) ? 1.0 : 0.0;
                    break;
                }
                case Opcode::OP_LESS: {
                    if (sp < 2) return false;
                    double right = stack[--sp];
                    double left  = stack[--sp];
                    stack[sp++]  = (left < right) ? 1.0 : 0.0;
                    break;
                }
            }
        }
        return sp == 0 ? true : stack[sp - 1] > 0.0;
    }
};