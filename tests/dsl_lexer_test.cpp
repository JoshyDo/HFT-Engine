// Tests for dsl-lexer.hpp.
//
// The lexer is a string-to-token stream. Bugs hide in:
//   - empty input
//   - whitespace handling
//   - reserved words vs identifiers (IF/THEN)
//   - numbers: integer, decimal, multi-dot (e.g. "1.2.3")
//   - unknown characters
//   - stream position: consecutive nextToken() calls

#include <gtest/gtest.h>
#include <string>
#include "dsl-lexer.hpp"

namespace {

// Helper: pull all tokens from a string until EOF.
std::vector<Token> lexAll(std::string_view src) {
    Lexer l(src);
    std::vector<Token> out;
    while (true) {
        Token t = l.nextToken();
        out.push_back(t);
        if (t.type == TokenType::EOF_TOKEN) break;
    }
    return out;
}

}  // namespace

TEST(LexerTest, EmptyInputReturnsEOF) {
    Lexer l("");
    Token t = l.nextToken();
    EXPECT_EQ(t.type, TokenType::EOF_TOKEN);
    EXPECT_EQ(t.lexeme, "");
}

TEST(LexerTest, WhitespaceOnlyReturnsEOF) {
    Lexer l("   \t\n  ");
    Token t = l.nextToken();
    EXPECT_EQ(t.type, TokenType::EOF_TOKEN);
}

TEST(LexerTest, ReservedWordsAreRecognized) {
    auto toks = lexAll("IF THEN");
    ASSERT_EQ(toks.size(), 3u);  // IF, THEN, EOF
    EXPECT_EQ(toks[0].type, TokenType::IF);
    EXPECT_EQ(toks[0].lexeme, "IF");
    EXPECT_EQ(toks[1].type, TokenType::THEN);
    EXPECT_EQ(toks[1].lexeme, "THEN");
    EXPECT_EQ(toks[2].type, TokenType::EOF_TOKEN);
}

TEST(LexerTest, IdentifierIsNotReserved) {
    // Lexer is case-sensitive: "IfThen", "If" and "Then" are all
    // identifiers, NOT reserved words. Only the exact strings "IF" and
    // "THEN" are reserved (see ReservedWordsAreRecognized).
    auto toks = lexAll("IfThen If Then");
    ASSERT_EQ(toks.size(), 4u);  // 3 idents + EOF
    EXPECT_EQ(toks[0].type, TokenType::IDENTIFIER);
    EXPECT_EQ(toks[0].lexeme, "IfThen");
    EXPECT_EQ(toks[1].type, TokenType::IDENTIFIER);
    EXPECT_EQ(toks[1].lexeme, "If");
    EXPECT_EQ(toks[2].type, TokenType::IDENTIFIER);
    EXPECT_EQ(toks[2].lexeme, "Then");
    EXPECT_EQ(toks[3].type, TokenType::EOF_TOKEN);
}

TEST(LexerTest, NumbersIntegerAndDecimal) {
    auto toks = lexAll("42 3.14");
    ASSERT_EQ(toks.size(), 3u);
    EXPECT_EQ(toks[0].type, TokenType::NUMBER);
    EXPECT_EQ(toks[0].lexeme, "42");
    EXPECT_EQ(toks[1].type, TokenType::NUMBER);
    EXPECT_EQ(toks[1].lexeme, "3.14");
}

TEST(LexerTest, NumberWithTrailingDotStopsAtDot) {
    // "1.2.3" — lexer should produce 1.2 then UNKNOWN "." then 3.
    // This documents the (possibly intentional) behavior.
    auto toks = lexAll("1.2.3");
    ASSERT_EQ(toks.size(), 4u);
    EXPECT_EQ(toks[0].type, TokenType::NUMBER);
    EXPECT_EQ(toks[0].lexeme, "1.2");
    EXPECT_EQ(toks[1].type, TokenType::UNKNOWN);
    EXPECT_EQ(toks[1].lexeme, ".");
    EXPECT_EQ(toks[2].type, TokenType::NUMBER);
    EXPECT_EQ(toks[2].lexeme, "3");
}

TEST(LexerTest, OperatorsAreUnknownTokens) {
    // Current implementation: < and > are not assigned a specific token
    // type. The parser matches them by lexeme. This test documents that
    // contract: operators come out as UNKNOWN with the operator string.
    auto toks = lexAll("< >");
    ASSERT_EQ(toks.size(), 3u);
    EXPECT_EQ(toks[0].type, TokenType::UNKNOWN);
    EXPECT_EQ(toks[0].lexeme, "<");
    EXPECT_EQ(toks[1].type, TokenType::UNKNOWN);
    EXPECT_EQ(toks[1].lexeme, ">");
}

TEST(LexerTest, MixedExpression) {
    // The canonical "Weight < 2.5" from main.cpp.
    auto toks = lexAll("Weight < 2.5");
    ASSERT_EQ(toks.size(), 4u);
    EXPECT_EQ(toks[0].type, TokenType::IDENTIFIER);
    EXPECT_EQ(toks[0].lexeme, "Weight");
    EXPECT_EQ(toks[1].type, TokenType::UNKNOWN);
    EXPECT_EQ(toks[1].lexeme, "<");
    EXPECT_EQ(toks[2].type, TokenType::NUMBER);
    EXPECT_EQ(toks[2].lexeme, "2.5");
}

TEST(LexerTest, LeadingWhitespaceSkipped) {
    auto toks = lexAll("   foo");
    ASSERT_EQ(toks.size(), 2u);
    EXPECT_EQ(toks[0].type, TokenType::IDENTIFIER);
    EXPECT_EQ(toks[0].lexeme, "foo");
}

TEST(LexerTest, UnknownCharacter) {
    auto toks = lexAll("@");
    ASSERT_EQ(toks.size(), 2u);
    EXPECT_EQ(toks[0].type, TokenType::UNKNOWN);
    EXPECT_EQ(toks[0].lexeme, "@");
}
