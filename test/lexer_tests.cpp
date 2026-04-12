#include "toy/Lexer.h"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

using namespace toy;

TEST_CASE("Lexer identifies basic tokens", "[lexer]") {
  const char *Input = "def foo(a) { return a + 1; }";
  LexerBuffer Lexer(Input, Input + strlen(Input), "test.toy");

  REQUIRE(Lexer.getNextToken() == Token::Def);
  REQUIRE(Lexer.getNextToken() == Token::Identifier);
  REQUIRE(Lexer.getId() == "foo");
  REQUIRE(Lexer.getNextToken() == Token::ParenthesesOpen);
  REQUIRE(Lexer.getNextToken() == Token::Identifier);
  REQUIRE(Lexer.getId() == "a");
  REQUIRE(Lexer.getNextToken() == Token::ParenthesesClose);
  REQUIRE(Lexer.getNextToken() == Token::BracketOpen);
  REQUIRE(Lexer.getNextToken() == Token::Return);
  REQUIRE(Lexer.getNextToken() == Token::Identifier);
  REQUIRE(Lexer.getId() == "a");
  REQUIRE(Lexer.getNextToken() == Token::Plus);
  REQUIRE(Lexer.getNextToken() == Token::Number);
  REQUIRE(Lexer.getNumber() == 1.0);
  REQUIRE(Lexer.getNextToken() == Token::Semicolon);
  REQUIRE(Lexer.getNextToken() == Token::BracketClose);
  REQUIRE(Lexer.getNextToken() == Token::Eof);
}

TEST_CASE("Lexer identifies numbers and keywords", "[lexer]") {
  const char *Input = "var b = 42.0\n print b";
  LexerBuffer Lexer(Input, Input + strlen(Input), "test.toy");

  REQUIRE(Lexer.getNextToken() == Token::Var);
  REQUIRE(Lexer.getNextToken() == Token::Identifier);
  REQUIRE(Lexer.getId() == "b");
  REQUIRE(Lexer.getNextToken() == Token::Equal);
  REQUIRE(Lexer.getNextToken() == Token::Number);
  REQUIRE(Lexer.getNumber() == 42.0);
  REQUIRE(Lexer.getNextToken() == Token::Print);
  REQUIRE(Lexer.getNextToken() == Token::Identifier);
  REQUIRE(Lexer.getId() == "b");
  REQUIRE(Lexer.getNextToken() == Token::Eof);
}

TEST_CASE("Lexer identifies tensors", "[lexer]") {
  const char *Input = "var tensor<2, 3> = [1, 2, 3, 4, 5, 6];";
  LexerBuffer Lexer(Input, Input + strlen(Input), "test.toy");

  REQUIRE(Lexer.getNextToken() == Token::Var);
  REQUIRE(Lexer.getNextToken() == Token::Identifier);
  REQUIRE(Lexer.getId() == "tensor");
  REQUIRE(Lexer.getNextToken() == Token::Less);
  REQUIRE(Lexer.getNextToken() == Token::Number);
  REQUIRE(Lexer.getNumber() == 2.0);
  REQUIRE(Lexer.getNextToken() == Token::Colon);
  REQUIRE(Lexer.getNextToken() == Token::Number);
  REQUIRE(Lexer.getNumber() == 3.0);
  REQUIRE(Lexer.getNextToken() == Token::Greater);
  REQUIRE(Lexer.getNextToken() == Token::Equal);
  REQUIRE(Lexer.getNextToken() == Token::SbracketOpen);
  REQUIRE(Lexer.getNextToken() == Token::Number);
  REQUIRE(Lexer.getNumber() == 1.0);
  REQUIRE(Lexer.getNextToken() == Token::Colon);
  REQUIRE(Lexer.getNextToken() == Token::Number);
  REQUIRE(Lexer.getNumber() == 2.0);
  REQUIRE(Lexer.getNextToken() == Token::Colon);
  REQUIRE(Lexer.getNextToken() == Token::Number);
  REQUIRE(Lexer.getNumber() == 3.0);
  REQUIRE(Lexer.getNextToken() == Token::Colon);
  REQUIRE(Lexer.getNextToken() == Token::Number);
  REQUIRE(Lexer.getNumber() == 4.0);
  REQUIRE(Lexer.getNextToken() == Token::Colon);
  REQUIRE(Lexer.getNextToken() == Token::Number);
  REQUIRE(Lexer.getNumber() == 5.0);
  REQUIRE(Lexer.getNextToken() == Token::Colon);
  REQUIRE(Lexer.getNextToken() == Token::Number);
  REQUIRE(Lexer.getNumber() == 6.0);
  REQUIRE(Lexer.getNextToken() == Token::SbracketClose);
  REQUIRE(Lexer.getNextToken() == Token::Semicolon);
  REQUIRE(Lexer.getNextToken() == Token::Eof);
}

TEST_CASE("Lexer identifies built-in functions", "[lexer]") {
  const char *Input = "print(transpose(a) * transpose(b));";
  LexerBuffer Lexer(Input, Input + strlen(Input), "test.toy");

  REQUIRE(Lexer.getNextToken() == Token::Print);
  REQUIRE(Lexer.getNextToken() == Token::ParenthesesOpen);
  REQUIRE(Lexer.getNextToken() == Token::Transpose);
  REQUIRE(Lexer.getNextToken() == Token::ParenthesesOpen);
  REQUIRE(Lexer.getNextToken() == Token::Identifier);
  REQUIRE(Lexer.getId() == "a");
  REQUIRE(Lexer.getNextToken() == Token::ParenthesesClose);
  REQUIRE(Lexer.getNextToken() == Token::Asterisk);
  REQUIRE(Lexer.getNextToken() == Token::Transpose);
  REQUIRE(Lexer.getNextToken() == Token::ParenthesesOpen);
  REQUIRE(Lexer.getNextToken() == Token::Identifier);
  REQUIRE(Lexer.getId() == "b");
  REQUIRE(Lexer.getNextToken() == Token::ParenthesesClose);
  REQUIRE(Lexer.getNextToken() == Token::ParenthesesClose);
  REQUIRE(Lexer.getNextToken() == Token::Semicolon);
  REQUIRE(Lexer.getNextToken() == Token::Eof);
}
