// Tests for dsl_compiler.hpp.
//
// The compiler walks an AST and emits bytecode for the VM. Bugs hide in:
//   - operand order: PUSH left, PUSH right, OP — for any binary op
//   - numeric conversion: std::stod edge cases
//   - unknown binary operator (would throw)

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>
#include "dsl-lexer.hpp"
#include "dsl-parser.hpp"
#include "dsl-ast.hpp"
#include "dsl_compiler.hpp"

namespace {

struct Compiled {
    ASTArena arena;
    std::vector<Instruction> bytecode;

    explicit Compiled(std::string_view src, uint32_t max_nodes = 16)
        : arena(max_nodes) {
        Lexer lex(src);
        Parser p(lex, arena);
        uint32_t root = p.parseExpression();
        Compiler c(arena);
        c.compileNode(root);
        bytecode = c.getBytecode();
    }
};

}  // namespace

TEST(CompilerTest, NumberEmitsPushNumber) {
    Compiled c("42");
    ASSERT_EQ(c.bytecode.size(), 1u);
    EXPECT_EQ(c.bytecode[0].op, Opcode::PUSH_NUMBER);
    EXPECT_DOUBLE_EQ(c.bytecode[0].operand, 42.0);
}

TEST(CompilerTest, IdentifierEmitsPushVar) {
    Compiled c("Weight");
    ASSERT_EQ(c.bytecode.size(), 1u);
    EXPECT_EQ(c.bytecode[0].op, Opcode::PUSH_VAR);
}

TEST(CompilerTest, DecimalNumber) {
    Compiled c("3.14");
    ASSERT_EQ(c.bytecode.size(), 1u);
    EXPECT_EQ(c.bytecode[0].op, Opcode::PUSH_NUMBER);
    EXPECT_DOUBLE_EQ(c.bytecode[0].operand, 3.14);
}

TEST(CompilerTest, LessThanEmitsPushPushOpLess) {
    Compiled c("Weight < 2.5");
    // Expected: PUSH_VAR, PUSH_NUMBER(2.5), OP_LESS
    ASSERT_EQ(c.bytecode.size(), 3u);
    EXPECT_EQ(c.bytecode[0].op, Opcode::PUSH_VAR);
    EXPECT_EQ(c.bytecode[1].op, Opcode::PUSH_NUMBER);
    EXPECT_DOUBLE_EQ(c.bytecode[1].operand, 2.5);
    EXPECT_EQ(c.bytecode[2].op, Opcode::OP_LESS);
}

TEST(CompilerTest, GreaterThanEmitsOpGreater) {
    Compiled c("x > 10");
    ASSERT_EQ(c.bytecode.size(), 3u);
    EXPECT_EQ(c.bytecode[0].op, Opcode::PUSH_VAR);
    EXPECT_EQ(c.bytecode[1].op, Opcode::PUSH_NUMBER);
    EXPECT_DOUBLE_EQ(c.bytecode[1].operand, 10.0);
    EXPECT_EQ(c.bytecode[2].op, Opcode::OP_GREATER);
}

TEST(CompilerTest, LeftAssociativeChainEmitsNestedOps) {
    // a < b < c  =>  (a < b) < c
    // Tree:
    //        <
    //       / \
    //      <   c
    //     / \
    //    a   b
    // Postorder traversal compiles left-subtree fully, then right-subtree,
    // then the operator at the current node. So bytecode for the OUTER `<`:
    //   [a, b, inner_<] then [c, outer_<]
    // = PUSH_VAR, PUSH_VAR, OP_LESS, PUSH_VAR, OP_LESS
    Compiled c("a < b < c");
    ASSERT_EQ(c.bytecode.size(), 5u);
    EXPECT_EQ(c.bytecode[0].op, Opcode::PUSH_VAR);
    EXPECT_EQ(c.bytecode[1].op, Opcode::PUSH_VAR);
    EXPECT_EQ(c.bytecode[2].op, Opcode::OP_LESS);
    EXPECT_EQ(c.bytecode[3].op, Opcode::PUSH_VAR);
    EXPECT_EQ(c.bytecode[4].op, Opcode::OP_LESS);
}
