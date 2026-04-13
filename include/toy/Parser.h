//===--- Parser.h - Parser for the Toy language ---------------------------===//
//
// This file implements the parser for the Toy language. It processes the Token
// provided by the Lexer and returns an AST.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_PARSER_H
#define TOY_PARSER_H

#include "toy/AST.h"
#include "toy/Lexer.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

namespace toy {

/// This is a simple recursive parser for the Toy language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &LexInstance) : Lex(LexInstance) {}

  /// Parse a full Module. A module is a list of function definitions
  std::unique_ptr<ModuleAST> parseModule() {
    // Init the lexer
    Lex.getNextToken();

    // Parse function one at a time and accumulate in this fector
    std::vector<FunctionAST> Functions;
    while (auto F = parseDefinition()) {
      Functions.push_back(std::move(*F));
      if (Lex.getCurToken() == Token::Eof) {
        break;
      }
    }

    // If we didn't reach EOF, there was a error during parsing
    if (Lex.getCurToken() != Token::Eof) {
      return parseError<ModuleAST>("nothing", "at end of module");
    }

    return std::make_unique<ModuleAST>(std::move(Functions));
  }

private:
  Lexer &Lex;

  /// return := 'return' ';'
  ///         | 'return' expr ';'
  std::unique_ptr<ReturnExprAST> parseReturn() {
    auto Loc = Lex.getLastLocation();
    Lex.consume(Token::Return);

    // "return" task an optional argument
    std::optional<std::unique_ptr<ExprAST>> Expr;
    if (Lex.getCurToken() != Token::Semicolon) {
      Expr = parseExpression();
      if (!Expr) {
        return nullptr;
      }
    }

    return std::make_unique<ReturnExprAST>(std::move(Loc), std::move(Expr));
  }

  /// numberExpr := number
  std::unique_ptr<ExprAST> parseNumberExpr() {
    auto Loc = Lex.getLastLocation();
    auto Res = std::make_unique<NumberExprAST>(std::move(Loc), Lex.getNumber());
    Lex.consume(Token::Number);
    return Res;
  }

  /// Parse a literal array expression
  /// tensorLiteral := '[' literalList ']'
  //                |  number
  /// literalList   := tensorLiteral
  //                |  tensorLiteral, literalList
  std::unique_ptr<ExprAST> parseTensorLiteralExpr() {
    auto Loc = Lex.getLastLocation();
    Lex.consume(Token::SbracketOpen);

    // Hold the list of values at this nesting level
    std::vector<std::unique_ptr<ExprAST>> Values;
    // Hold the dimensions for all the nesting inside this level
    std::vector<int64_t> Dims;

    do {
      // We can have either another nested array or a number literal
      if (Lex.getCurToken() == Token::SbracketOpen) {
        Values.push_back(parseTensorLiteralExpr());
        if (!Values.back()) {
          return nullptr; // Parse error in the nested array
        }
      } else {
        if (Lex.getCurToken() != Token::Number) {
          return parseError<ExprAST>("<num> or [", "in literal expression");
        }

        Values.push_back(parseNumberExpr());
      }

      // End of this list on ']'
      if (Lex.getCurToken() == Token::SbracketClose) {
        break;
      }

      // Elements are separeted by a comma.
      if (Lex.getCurToken() != Token::Comma) {
        return parseError<ExprAST>("] or ,", "in literal expression");
      }

      Lex.consume(Token::Comma); // Consume ','
    } while (true);

    if (Values.empty()) {
      return parseError<ExprAST>("<something>", "to fill literal expression");
    }
    Lex.getNextToken(); // Consume ']'

    // Fill in the dimensions now. First hte current nesting level:
    Dims.push_back(static_cast<long>(Values.size()));

    // If there is any nested array, process all of them and ensure that
    // dimensios are uniform.
    if (llvm::any_of(Values, [](std::unique_ptr<ExprAST> &E) {
          return llvm::isa<LiteralExprAST>(E.get());
        })) {
      auto *FirstLiteral = llvm::dyn_cast<LiteralExprAST>(Values.front().get());
      if (!FirstLiteral) {
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");
      }

      // Append the nested dimensions to the current level
      auto FirstDims = FirstLiteral->getDims();
      Dims.insert(Dims.end(), FirstDims.begin(), FirstDims.end());

      // Sanity check that shape is uniform across all elements of the list
      for (auto &Expr : Values) {
        auto *ExprLit = llvm::cast<LiteralExprAST>(Expr.get());
        if (!ExprLit) {
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
        }

        if (ExprLit->getDims() != FirstDims) {
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
        }
      }
    }

    return std::make_unique<LiteralExprAST>(std::move(Loc), std::move(Values),
                                            std::move(Dims));
  }

  /// parenExpr := '(' expression ')'
  std::unique_ptr<ExprAST> parseParenExpr() {
    Lex.getNextToken(); // consume '('

    auto V = parseExpression();
    if (!V) {
      return nullptr;
    }

    if (Lex.getCurToken() != Token::ParenthesesClose) {
      return parseError<ExprAST>(")", "to close expression with parentheses");
    }

    Lex.consume(Token::ParenthesesClose);
    return V;
  }

  /// identifierexpr := identifier
  ///                 | identifier '(' expression ')'
  std::unique_ptr<ExprAST> parseIdentifierExpr() {
    std::string Name(Lex.getId());

    auto Loc = Lex.getLastLocation();
    Lex.getNextToken(); // consume identifier

    // Simple variable ref.
    if (Lex.getCurToken() != Token::ParenthesesOpen) {
      return std::make_unique<VariableExprAST>(std::move(Loc), Name);
    }

    // This is a function call.
    Lex.consume(Token::ParenthesesOpen);
    std::vector<std::unique_ptr<ExprAST>> Args;
    if (Lex.getCurToken() != Token::ParenthesesClose) {
      while (true) {
        if (auto Arg = parseExpression()) {
          Args.push_back(std::move(Arg));
        } else {
          return nullptr;
        }

        if (Lex.getCurToken() == Token::ParenthesesClose) {
          break;
        }

        if (Lex.getCurToken() != Token::Comma) {
          return parseError<ExprAST>(", or )", "in argument list");
        }

        Lex.getNextToken();
      }
    }

    Lex.consume(Token::ParenthesesClose);

    // It can be a builint call to 'print' or 'transpose'
    if (Name == "print") {
      if (Args.size() != 1) {
        return parseError<ExprAST>("<single arg>",
                                   "as argument to 'print' function.");
      }

      return std::make_unique<PrintExprAST>(std::move(Loc), std::move(Args[0]));
    }

    if (Name == "transpose") {
      if (Args.size() != 1) {
        return parseError<ExprAST>("<single arg>",
                                   "as argument to 'transpose' function.");
      }

      return std::make_unique<TransposeExprAST>(std::move(Loc),
                                                std::move(Args[0]));
    }

    // Call to a user-defined function
    return std::make_unique<CallExprAST>(std::move(Loc), Name, std::move(Args));
  }

  /// primary := identifierExpr
  ///          | numberExpr
  ///          | parenExpr
  ///          | tensorLiteral
  std::unique_ptr<ExprAST> parsePrimary() {
    switch (Lex.getCurToken()) {
    case Token::Identifier:
      return parseIdentifierExpr();
    case Token::Number:
      return parseNumberExpr();
    case Token::ParenthesesOpen:
      return parseParenExpr();
    case Token::SbracketOpen:
      return parseTensorLiteralExpr();
    case Token::BracketClose:
    case Token::Semicolon:
      return nullptr;
    default:
      llvm::errs() << "unknown token when expecting an expression\n";
      return nullptr;
    }
  }

  /// Recursively parse the right hand side of a binary epxression, the ExprPrec
  /// argument indicates the precedence of the current binary operator.
  ///
  /// binOpRhs := ('+' primary)*
  std::unique_ptr<ExprAST> parseBinOpRHS(int ExprPrec,
                                         std::unique_ptr<ExprAST> LHS) {
    // If this is a binop, find its precedence
    while (true) {
      int TokPrec = getTokPrecedence();

      // If this is a binop that binds at least as tightly as the current binop,
      // consume it, otherwise we are done.
      if (TokPrec < ExprPrec) {
        return LHS;
      }

      Token BinOp = Lex.getCurToken();
      Lex.consume(BinOp);
      auto Loc = Lex.getLastLocation();

      // Parse the primary expression after the binary operator.
      auto RHS = parsePrimary();
      if (!RHS) {
        return parseError<ExprAST>("expression", "to complete binary operator");
      }

      // If BinOp binds less tightly with rhs thatn the operator after rhs, let
      // the pending operator take rhs as its lhs.
      int NextPrec = getTokPrecedence();
      if (TokPrec < NextPrec) {
        RHS = parseBinOpRHS(TokPrec + 1, std::move(RHS));
        if (!RHS) {
          return nullptr;
        }
      }

      // Merge LHS/RHS
      LHS = std::make_unique<BinaryExprAST>(std::move(Loc), tokenToChar(BinOp),
                                            std::move(LHS), std::move(RHS));
    }
  }

  /// expression := primary binop rhs
  std::unique_ptr<ExprAST> parseExpression() {
    auto LHS = parsePrimary();
    if (!LHS) {
      return nullptr;
    }

    return parseBinOpRHS(0, std::move(LHS));
  }

  /// type       := '<' shape_list '>'
  /// shape_list := num
  ///             | num , shape_list
  std::unique_ptr<VarType> parseType() {
    if (Lex.getCurToken() != Token::Less) {
      return parseError<VarType>("<", "to begin type");
    }

    Lex.getNextToken(); // Consume '<'

    auto Type = std::make_unique<VarType>();

    while (Lex.getCurToken() == Token::Number) {
      Type->Shape.push_back(static_cast<long>(Lex.getNumber()));
      Lex.getNextToken();
      if (Lex.getCurToken() == Token::Comma) {
        Lex.getNextToken();
      }
    }

    if (Lex.getCurToken() != Token::Greater) {
      return parseError<VarType>(">", "to end type");
    }

    Lex.getNextToken(); // consume '>'
    return Type;
  }

  /// Parse a variable declaration, it starts with a `var` keyword followed by
  /// and identifier and an optional type (shape specification) before the
  /// initializer.
  /// decl := 'var' identifier [ type ] = expr
  std::unique_ptr<VarDeclExprAST> parseDeclaration() {
    if (Lex.getCurToken() != Token::Var) {
      return parseError<VarDeclExprAST>("var", "to begin declaration");
    }

    auto Loc = Lex.getLastLocation();
    Lex.getNextToken(); // consume 'var'

    if (Lex.getCurToken() != Token::Identifier) {
      return parseError<VarDeclExprAST>("identifier",
                                        "after 'var' declaration");
    }

    std::string ID(Lex.getId());
    Lex.getNextToken(); // consume identifier

    std::unique_ptr<VarType> Type; // Type is optional, it can be inferred
    if (Lex.getCurToken() == Token::Less) {
      Type = parseType();
      if (!Type) {
        return nullptr;
      }
    }

    if (!Type) {
      Type = std::make_unique<VarType>();
    }

    Lex.consume(Token::Equal);
    auto Expr = parseExpression();
    return std::make_unique<VarDeclExprAST>(std::move(Loc), std::move(ID),
                                            std::move(*Type), std::move(Expr));
  }

  /// Parse a block: a list of expressions spearated by semicolons and wrapped
  /// in curly braces.
  ///
  /// block           := '{' expression_list '}'
  /// expression_list := block_expr ';' expression_list
  /// block_expr      := decl
  ///                  | "return"
  ///                  | expr
  std::unique_ptr<ExprASTList> parseBlock() {
    if (Lex.getCurToken() != Token::BracketOpen) {
      return parseError<ExprASTList>("{", "to begin block");
    }

    Lex.consume(Token::BracketOpen);

    auto ExprList = std::make_unique<ExprASTList>();

    // Ignore empty expressions: swallow sequences of semicolons.
    while (Lex.getCurToken() == Token::Semicolon) {
      Lex.consume(Token::Semicolon);
    }

    while (Lex.getCurToken() != Token::BracketClose &&
           Lex.getCurToken() != Token::Eof) {
      if (Lex.getCurToken() == Token::Var) {
        // Variable declaration
        auto VarDecl = parseDeclaration();
        if (!VarDecl) {
          return nullptr;
        }

        ExprList->push_back(std::move(VarDecl));
      } else if (Lex.getCurToken() == Token::Return) {
        // Return statement
        auto Ret = parseReturn();
        if (!Ret) {
          return nullptr;
        }

        ExprList->push_back(std::move(Ret));
      } else {
        // General expression
        auto Expr = parseExpression();
        if (!Expr) {
          return nullptr;
        }

        ExprList->push_back(std::move(Expr));
      }

      // Ensure that elements are separated by a semicolon
      if (Lex.getCurToken() != Token::Semicolon) {
        return parseError<ExprASTList>(";", "after expression");
      }

      // Ignore empty expressions: swallow sequences of semicolons.
      while (Lex.getCurToken() == Token::Semicolon) {
        Lex.consume(Token::Semicolon);
      }
    }

    if (Lex.getCurToken() != Token::BracketClose) {
      return parseError<ExprASTList>("}", "to close block");
    }

    Lex.consume(Token::BracketClose);
    return ExprList;
  }

  /// prototype := 'def' id '(' decl_list ')'
  /// decl_list := identifier
  ///            | identifier ',' decl_list
  std::unique_ptr<PrototypeAST> parsePrototype() {
    auto Loc = Lex.getLastLocation();

    if (Lex.getCurToken() != Token::Def) {
      return parseError<PrototypeAST>("def", "in prototype");
    }
    Lex.consume(Token::Def);

    if (Lex.getCurToken() != Token::Identifier) {
      return parseError<PrototypeAST>("function name", "in prototype");
    }
    std::string FnName(Lex.getId());
    Lex.consume(Token::Identifier);

    if (Lex.getCurToken() != Token::ParenthesesOpen) {
      return parseError<PrototypeAST>("(", "in prototype");
    }
    Lex.consume(Token::ParenthesesOpen);

    std::vector<std::unique_ptr<VariableExprAST>> Args;
    if (Lex.getCurToken() != Token::ParenthesesClose) {
      do {

        std::string Name(Lex.getId());
        auto IdLoc = Lex.getLastLocation();
        Lex.consume(Token::Identifier);
        auto Decl = std::make_unique<VariableExprAST>(std::move(IdLoc), Name);
        Args.push_back(std::move(Decl));

        if (Lex.getCurToken() != Token::Comma) {
          break;
        }

        Lex.consume(Token::Comma);
        if (Lex.getCurToken() != Token::Identifier) {
          return parseError<PrototypeAST>(
              "identifier", "after ',' in function parameter list");
        }
      } while (true);
    }

    if (Lex.getCurToken() != Token::ParenthesesClose) {
      return parseError<PrototypeAST>(")", "to end function prototype");
    }

    // Success
    Lex.consume(Token::ParenthesesClose);
    return std::make_unique<PrototypeAST>(std::move(Loc), FnName,
                                          std::move(Args));
  }

  /// Parse a function definition, we expect a prototype initiated with the
  /// 'def' keyword, followed by a block containing a list of expressions.
  ///
  /// definition := prototype block
  std::unique_ptr<FunctionAST> parseDefinition() {
    auto Proto = parsePrototype();
    if (!Proto) {
      return nullptr;
    }

    if (auto Block = parseBlock()) {
      return std::make_unique<FunctionAST>(std::move(Proto), std::move(Block));
    }

    return nullptr;
  }

  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    // 1 is the lowest precedence
    switch (Lex.getCurToken()) {
    case Token::Minus:
    case Token::Plus:
      return 20;
    case Token::Asterisk:
      return 40;
    default:
      return -1;
    }
  }

  /// Helper functiont o signal errors while parsing, it takes an argument
  /// indicanting the expected token and another argument giving more context.
  /// Location is retrieve from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&Expt, U &&Ctx = "") {
    auto CurToken = tokenToWord(Lex.getCurToken());
    llvm::errs() << "Parse error (" << Lex.getLastLocation().Line << ", "
                 << Lex.getLastLocation().Col << "): expected '" << Expt << "' "
                 << Ctx << " but has Token " << CurToken << '\n';
    return nullptr;
  }
};

} // namespace toy

#endif // TOY_PARSER_H
