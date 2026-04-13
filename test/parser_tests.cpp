#include "toy/Parser.h"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

using namespace toy;

TEST_CASE("Parser handles the multiply_transpose example", "[parser]") {
  const char *Input = R"(
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);

  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return <3, 2>.
  var d = multiply_transpose(b, a);

  # A new call with <3, 2> (instead of <2, 3>) for both dimensions will
  # trigger another specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);

  # Finally, calling into `multiply_transpose` with incompatible shapes
  # (<2, 3> and <3, 2>) will trigger a shape inference error.
  var f = multiply_transpose(a, c);
}
)";

  LexerBuffer Lexer(Input, Input + strlen(Input), "test.toy");
  Parser Parser(Lexer);

  auto Module = Parser.parseModule();
  REQUIRE(Module != nullptr);

  int FnCount = 0;
  for (auto it = Module->begin(); it != Module->end(); ++it) {
    FnCount++;
  }
  REQUIRE(FnCount == 2);

  auto it = Module->begin();
  auto &MTFunc = *it;
  REQUIRE(MTFunc.getProto()->getName() == "multiply_transpose");
  REQUIRE(MTFunc.getProto()->getArgs().size() == 2);
  REQUIRE(MTFunc.getBody()->size() == 1);

  ++it;
  auto &MainFunc = *it;
  REQUIRE(MainFunc.getProto()->getName() == "main");
  REQUIRE(MainFunc.getProto()->getArgs().size() == 0);
  REQUIRE(MainFunc.getBody()->size() == 6);
}

TEST_CASE("Parser handles empty functions", "[parser]") {
  const char *Input = "def foo() {}";
  LexerBuffer Lexer(Input, Input + strlen(Input), "test.toy");
  Parser Parser(Lexer);

  auto Module = Parser.parseModule();
  REQUIRE(Module != nullptr);
  
  int FnCount = 0;
  for (auto it = Module->begin(); it != Module->end(); ++it) {
    FnCount++;
  }
  REQUIRE(FnCount == 1);
  auto &Func = *(Module->begin());
  REQUIRE(Func.getProto()->getName() == "foo");
  REQUIRE(Func.getBody()->size() == 0);
}

TEST_CASE("Parser handles variable declarations with types", "[parser]") {
  const char *Input = "def foo() { var a<2, 3> = [1, 2, 3, 4, 5, 6]; }";
  LexerBuffer Lexer(Input, Input + strlen(Input), "test.toy");
  Parser Parser(Lexer);

  auto Module = Parser.parseModule();
  REQUIRE(Module != nullptr);
  auto &Func = *(Module->begin());
  REQUIRE(Func.getBody()->size() == 1);
  auto *VarDecl = llvm::dyn_cast<VarDeclExprAST>(Func.getBody()->front().get());
  REQUIRE(VarDecl != nullptr);
  REQUIRE(VarDecl->getName() == "a");
  REQUIRE(VarDecl->getType().Shape.size() == 2);
  REQUIRE(VarDecl->getType().Shape[0] == 2);
  REQUIRE(VarDecl->getType().Shape[1] == 3);

  auto *InitVal = llvm::dyn_cast<LiteralExprAST>(VarDecl->getInitVal());
  REQUIRE(InitVal != nullptr);
  auto Values = InitVal->getValues();
  REQUIRE(Values.size() == 6);
  for (size_t i = 0; i < 6; ++i) {
    auto *Num = llvm::dyn_cast<NumberExprAST>(Values[i].get());
    REQUIRE(Num != nullptr);
    REQUIRE(Num->getValue() == static_cast<double>(i + 1));
  }
}
