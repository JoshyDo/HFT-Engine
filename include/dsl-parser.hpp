#pragma once
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "dsl-ast.hpp"
#include "dsl-lexer.hpp"

class Parser {
    Lexer&    lexer;
    ASTArena& arena;
    Token     currentToken;

public:
    Parser(Lexer& lex, ASTArena& ar) : lexer(lex), arena(ar) { currentToken = lexer.nextToken(); }

    void advance() { currentToken = lexer.nextToken(); }

    uint32_t parsePrimary() {
        if (currentToken.type == TokenType::NUMBER) {
            uint32_t nodeIndex = arena.allocateNode(ASTNodeType::NUMBER, currentToken);
            advance();
            return nodeIndex;
        } else if (currentToken.type == TokenType::IDENTIFIER) {
            uint32_t nodeIndex = arena.allocateNode(ASTNodeType::IDENTIFIER, currentToken);
            advance();
            return nodeIndex;
        } else {
            throw std::runtime_error("Unexpected token in primary expression");
        }
    }

    uint32_t parseExpression() {
        uint32_t leftNode = parsePrimary();
        while (currentToken.lexeme == ">" || currentToken.lexeme == "<") {
            Token opToken = currentToken;
            advance();
            uint32_t rightNode = parsePrimary();
            leftNode           = arena.allocateNode(ASTNodeType::BINARY_OP, opToken, leftNode, rightNode);
        }
        return leftNode;
    }
};