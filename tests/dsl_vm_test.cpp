// Tests for dsl_vm.hpp — bytecode evaluation.
//
// The VM is a stack machine. The result is whatever the stack top is
// when bytecode runs out. Edge cases:
//   - empty bytecode → returns true (stack empty, sp == 0)
//   - underflow (OP without two operands) → returns false
//   - truthiness: only nonzero stack-top counts as true
//   - operand order: VM pops right then left, so left OP right works

#include <gtest/gtest.h>
#include <string>
#include "dsl-lexer.hpp"
#include "dsl-parser.hpp"
#include "dsl-ast.hpp"
#include "dsl_compiler.hpp"
#include "dsl_vm.hpp"

namespace {

bool eval(std::string_view src, double edge_weight) {
    ASTArena arena(16);
    Lexer lex(src);
    Parser p(lex, arena);
    uint32_t root = p.parseExpression();
    Compiler c(arena);
    c.compileNode(root);
    VirtualMachine vm(c.getBytecode());
    return vm.evaluate(edge_weight);
}

}  // namespace

TEST(VMTest, EmptyBytecodeReturnsTrue) {
    ASTArena arena(8);
    Compiler c(arena);
    // No compileNode call → empty bytecode
    VirtualMachine vm(c.getBytecode());
    EXPECT_TRUE(vm.evaluate(0.0));
}

TEST(VMTest, LessThanTrue) {
    EXPECT_TRUE(eval("Weight < 2.5", 1.0));
    EXPECT_TRUE(eval("Weight < 2.5", 2.4));
    EXPECT_FALSE(eval("Weight < 2.5", 2.5));  // boundary
    EXPECT_FALSE(eval("Weight < 2.5", 3.0));
}

TEST(VMTest, GreaterThanTrue) {
    EXPECT_FALSE(eval("Weight > 2.5", 1.0));
    EXPECT_FALSE(eval("Weight > 2.5", 2.5));  // boundary
    EXPECT_TRUE(eval("Weight > 2.5", 2.6));
    EXPECT_TRUE(eval("Weight > 2.5", 100.0));
}

TEST(VMTest, ConstantNumber) {
    // 42 is always truthy.
    EXPECT_TRUE(eval("42", 0.0));
    // 0.0 is falsy (sp > 0 at end, top == 0.0, so the result `top > 0.0` is false).
    EXPECT_FALSE(eval("0", 0.0));
}

TEST(VMTest, ChainedComparisonLeftAssociative) {
    // "a < b < c" parses as (a < b) < c. With a=Weight, b=2, c=3:
    //   - (Weight < 2) → 0.0 or 1.0
    //   - that result < 3 → if (Weight<2) is 0: 0<3 is true. If 1: 1<3 is true.
    // So with Weight < 2, the result is always true here.
    EXPECT_TRUE(eval("a < 2 < 3", 1.0));
    EXPECT_TRUE(eval("a < 2 < 3", 5.0));
}
