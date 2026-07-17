#pragma once
#include <cctype>
#include <string_view>

enum class TokenType { IF, THEN, NUMBER, IDENTIFIER, OPERATOR, EOF_TOKEN, UNKNOWN };

struct Token {
    TokenType        type;
    std::string_view lexeme;
};

class Lexer {
private:
    std::string_view source;
    size_t           pos = 0;

public:
    Lexer(std::string_view src) : source(src) {}

    Token nextToken() {
        while (pos < source.length() && std::isspace(source[pos])) {
            ++pos;
        }
        if (pos >= source.length()) {
            return {TokenType::EOF_TOKEN, ""};
        }
        if (std::isalpha(source[pos])) {
            size_t start = pos;
            while (pos < source.length() && std::isalnum(source[pos])) {
                ++pos;
            }
            std::string_view lexeme = source.substr(start, pos - start);
            if (lexeme == "IF") {
                return {TokenType::IF, lexeme};
            } else if (lexeme == "THEN") {
                return {TokenType::THEN, lexeme};
            } else {
                return {TokenType::IDENTIFIER, lexeme};
            }
        } else if (std::isdigit(source[pos])) {
            size_t start = pos;
            while (pos < source.length() && std::isdigit(source[pos])) {
                ++pos;
            }
            if (pos < source.length() && source[pos] == '.') {
                ++pos;
                while (pos < source.length() && std::isdigit(source[pos])) {
                    ++pos;
                }
            }
            return {TokenType::NUMBER, source.substr(start, pos - start)};

        } else {
            return {TokenType::UNKNOWN, source.substr(pos++, 1)};
        }
    }
};