// Tests for dsl-parser.hpp + dsl-ast.hpp.
//
// The parser is a recursive-descent over the lexer. Bugs hide in:
//   - left-associativity: "a < b < c" must parse as (a < b) < c
//   - error path: unexpected token (e.g. operator at start)
//   - arena overflow: more nodes than max_nodes
//
// Grammar (informal):
//   expression := primary ( ( "<" | ">" ) primary )*
//   primary    := NUMBER | IDENTIFIER

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>
#include "dsl-lexer.hpp"
#include "dsl-parser.hpp"
#include "dsl-ast.hpp"

namespace {

// Helper: parse a string and return the root index.
uint32_t parseString(std::string_view src, ASTArena& arena) {
    Lexer lex(src);
    Parser p(lex, arena);
    return p.parseExpression();
}

const ASTNode& nodeAt(const ASTArena& arena, uint32_t i) {
    return arena.getNodes()[i];
}

}  // namespace

TEST(ParserTest, SingleNumber) {
    ASTArena arena(8);
    uint32_t root = parseString("42", arena);
    EXPECT_EQ(nodeAt(arena, root).type, ASTNodeType::NUMBER);
    EXPECT_EQ(nodeAt(arena, root).token.lexeme, "42");
}

TEST(ParserTest, SingleIdentifier) {
    ASTArena arena(8);
    uint32_t root = parseString("Weight", arena);
    EXPECT_EQ(nodeAt(arena, root).type, ASTNodeType::IDENTIFIER);
    EXPECT_EQ(nodeAt(arena, root).token.lexeme, "Weight");
}

TEST(ParserTest, BinaryLessThan) {
    ASTArena arena(8);
    uint32_t root = parseString("Weight < 2.5", arena);

    // Root is a BINARY_OP for `<`, with left=Weight, right=2.5
    EXPECT_EQ(nodeAt(arena, root).type, ASTNodeType::BINARY_OP);
    EXPECT_EQ(nodeAt(arena, root).token.lexeme, "<");

    uint32_t left = nodeAt(arena, root).left;
    uint32_t right = nodeAt(arena, root).right;
    EXPECT_EQ(nodeAt(arena, left).type, ASTNodeType::IDENTIFIER);
    EXPECT_EQ(nodeAt(arena, left).token.lexeme, "Weight");
    EXPECT_EQ(nodeAt(arena, right).type, ASTNodeType::NUMBER);
    EXPECT_EQ(nodeAt(arena, right).token.lexeme, "2.5");
}

TEST(ParserTest, LeftAssociativeChain) {
    // a < b < c  must parse as  (a < b) < c
    // In the AST: root is the OUTER `<`, left is the inner `<(a,b)`, right is c.
    ASTArena arena(16);
    uint32_t root = parseString("a < b < c", arena);

    ASSERT_EQ(nodeAt(arena, root).type, ASTNodeType::BINARY_OP);
    EXPECT_EQ(nodeAt(arena, root).token.lexeme, "<");

    uint32_t outer_left = nodeAt(arena, root).left;
    uint32_t outer_right = nodeAt(arena, root).right;
    EXPECT_EQ(nodeAt(arena, outer_right).type, ASTNodeType::IDENTIFIER);
    EXPECT_EQ(nodeAt(arena, outer_right).token.lexeme, "c");

    ASSERT_EQ(nodeAt(arena, outer_left).type, ASTNodeType::BINARY_OP);
    EXPECT_EQ(nodeAt(arena, outer_left).token.lexeme, "<");

    uint32_t inner_left = nodeAt(arena, outer_left).left;
    uint32_t inner_right = nodeAt(arena, outer_left).right;
    EXPECT_EQ(nodeAt(arena, inner_left).token.lexeme, "a");
    EXPECT_EQ(nodeAt(arena, inner_right).token.lexeme, "b");
}

TEST(ParserTest, BinaryGreaterThan) {
    ASTArena arena(8);
    uint32_t root = parseString("x > 10", arena);
    EXPECT_EQ(nodeAt(arena, root).token.lexeme, ">");
}

TEST(ParserTest, TrailingOperatorThrows) {
    // "Weight <" — primary `<` primary fails because there's no right operand.
    // The current parser will call parsePrimary() which sees EOF, throws.
    ASTArena arena(8);
    EXPECT_THROW(parseString("Weight <", arena), std::runtime_error);
}

TEST(ParserTest, LeadingOperatorThrows) {
    // "< 5" — parsePrimary() sees `<` which is not a NUMBER or IDENTIFIER.
    ASTArena arena(8);
    EXPECT_THROW(parseString("< 5", arena), std::runtime_error);
}

TEST(ParserTest, ArenaOverflowThrows) {
    // 4 nodes fit (a, b, c, root) — but we demand a 3-node arena.
    ASTArena arena(3);
    EXPECT_THROW(parseString("a < b < c", arena), std::runtime_error);
}
