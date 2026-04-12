#include "toy/Lexer.h"

#include <iostream>
#include <string>

using namespace toy;

const std::string Archive =
    "def main() {\n"
    "  # Define a variable `a` with shape <2, 3>, initialized with the literal "
    "value.\n"
    "  # The shape is inferred from the supplied literal.\n"
    "  var a = [[1, 2, 3], [4, 5, 6]];\n"
    "\n"
    "  # b is identical to a, the literal tensor is implicitly reshaped: "
    "defining new\n"
    "  # variables is the way to reshape tensors (element count must match).\n"
    "  var b<2, 3> = [1, 2, 3, 4, 5, 6];\n"
    "\n"
    "  # transpose() and print() are the only builtin, the following will "
    "transpose\n"
    "  # a and b and perform an element-wise multiplication before printing"
    "the result.\n"
    "  print(transpose(a) * transpose(b));\n"
    "}\n";

int main() {
  LexerBuffer Lexer(Archive.c_str(), Archive.c_str() + Archive.size(), "test");
  Lexer.getNextToken();

  int Count = 1;
  Token Tok = Lexer.getCurToken();
  while (true) {
    std::cout << Count << ": " << tokenToWord(Tok).data() << "\n";
    Lexer.getNextToken();
    Tok = Lexer.getCurToken();
    if (Tok == Token::Eof) {
      std::cout << tokenToWord(Tok).data() << "\n";
      break;
    }

    Count++;
  }

  return 0;
}
