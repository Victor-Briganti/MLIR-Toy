//===--- AST.cpp - Helper for printing the AST ----------------------------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST dump for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "toy/AST.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/STLExtras.h>

using namespace toy;

namespace {
// RAII helper to mange increasing/decreasing the identation as we traverse the
// AST
struct Indent {
  Indent(int &levelToIncrement) : level(levelToIncrement) { ++level; }
  ~Indent() { --level; }
  int &level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
public:
  void dump(ModuleAST *node);

private:
  int curIndent = 0;

  void dump(const VarType &type);
  void dump(VarDeclExprAST *varDecl);
  void dump(ExprAST *expr);
  void dump(ExprASTList *exprList);
  void dump(NumberExprAST *num);
  void dump(LiteralExprAST *node);
  void dump(VariableExprAST *node);
  void dump(ReturnExprAST *node);
  void dump(BinaryExprAST *node);
  void dump(CallExprAST *node);
  void dump(PrintExprAST *node);
  void dump(TransposeExprAST *node);
  void dump(PrototypeAST *node);
  void dump(FunctionAST *node);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < curIndent; i++) {
      llvm::errs() << " ";
    }
  }
};

} // namespace

/// Return a formatted string for the location of any node
template <typename T>
static std::string loc(T *node) {
  const auto &location = node->loc();
  return (llvm::Twine("@") + *location.file + ":" + llvm::Twine(location.line) +
          ":" + llvm::Twine(location.col))
      .str();
}

// Helper macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_raii_(curIndent);                                               \
  indent();

/// Dispatch the generic expression to the appropriate subclass using RTTI
void ASTDumper::dump(ExprAST *expr) {
  llvm::TypeSwitch<ExprAST *>(expr)
      .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
            PrintExprAST, ReturnExprAST, TransposeExprAST, VarDeclExprAST,
            VariableExprAST>([&](auto *node) { this->dump(node); })
      .Default([&](ExprAST *) {
        // No match fallback to generic message
        INDENT();
        llvm::errs() << "<unknow Expr, kind " << expr->getStrKind() << ">\n";
      });
}

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(VarDeclExprAST *varDecl) {
  INDENT();
  llvm::errs() << "VarDecl " << varDecl->getName();
  dump(varDecl->getType());
  llvm::errs() << " " << loc(varDecl) << "\n";
  dump(varDecl->getInitVal());
}

/// A "block", or a list of expression.
void ASTDumper::dump(ExprASTList *exprList) {
  INDENT();
  llvm::errs() << "Block {\n";
  for (auto &E : *exprList) {
    dump(E.get());
  }
  llvm::errs() << "} // Block\n";
}

/// A literal number, just print the value
void ASTDumper::dump(NumberExprAST *num) {
  INDENT();
  llvm::errs() << num->getValue() << " " << loc(num) << "\n";
}

/// Helper to print recursively a literal. This handles nested array like:
/// [ [ 1, 2 ], [ 3, 4 ] ]
/// We print out such array wiht the dimensions spelled out at every level:
/// <2,2>[<2>[ 1, 2 ], <2>[ 3, 4 ] ]
static void printLitHelper(ExprAST *litOrNum) {
  // Inside a literal expression we can have either a number or another literal
  if (auto *numExpr = llvm::dyn_cast<NumberExprAST>(litOrNum)) {
    llvm::errs() << numExpr->getValue();
    return;
  }

  auto *lit = llvm::cast<LiteralExprAST>(litOrNum);

  // Print the dimensions for the literal first
  llvm::errs() << "<";
  llvm::interleaveComma(lit->getDims(), llvm::errs());
  llvm::errs() << ">";

  // Now print the content, recursing on every element of the list
  llvm::errs() << "[ ";
  llvm::interleaveComma(lit->getValues(), llvm::errs(),
                        [&](const auto &E) { printLitHelper(E.get()); });
  llvm::errs() << "]";
}

/// Print a literal
void ASTDumper::dump(LiteralExprAST *node) {
  INDENT();
  llvm::errs() << "Literal: ";
  printLitHelper(node);
  llvm::errs() << " " << loc(node) << "\n";
}

/// Print a variable reference (just a name)
void ASTDumper::dump(VariableExprAST *node) {
  INDENT();
  llvm::errs() << "Var: " << node->getName() << " " << loc(node) << "\n";
}

/// Return statement print the return and its (optional) arguments.
void ASTDumper::dump(ReturnExprAST *node) {
  INDENT();
  llvm::errs() << "Return\n";
  if (node->getExpr().has_value()) {
    return dump(*node->getExpr());
  }

  {
    INDENT();
    llvm::errs() << "(void)\n";
  }
}

/// Print a binary operation, first the operator, then recurse into LHS and RHS
void ASTDumper::dump(BinaryExprAST *node) {
  INDENT();
  llvm::errs() << "BinOps: " << node->getOp() << " " << loc(node) << '\n';
  dump(node->getLhs());
  dump(node->getRhs());
}

/// Print a call expression, first the callee name and the list of args by
/// recursing into each individual argument.
void ASTDumper::dump(CallExprAST *node) {
  INDENT();
  llvm::errs() << "Call '" << node->getCallee() << "' [" << loc(node) << '\n';
  for (auto &arg : node->getArgs()) {
    dump(arg.get());
  }
  indent();
  llvm::errs() << "]\n";
}

/// Print a builtin "print" call, first the builtin name and then the argument.
void ASTDumper::dump(PrintExprAST *node) {
  INDENT();
  llvm::errs() << "Print [" << loc(node) << '\n';
  dump(node->getArg());
  indent();
  llvm::errs() << "]\n";
}

/// Print a builtin "transpose" call, first the builtin name and then the
/// argument.
void ASTDumper::dump(TransposeExprAST *node) {
  INDENT();
  llvm::errs() << "Transpose [" << loc(node) << '\n';
  dump(node->getArg());
  indent();
  llvm::errs() << "]\n";
}

/// Print type: only the shape is printed in between '<' and '>'
void ASTDumper::dump(const VarType &type) {
  llvm::errs() << "<";
  llvm::interleaveComma(type.shape, llvm::errs());
  llvm::errs() << ">";
}

/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(PrototypeAST *node) {
  INDENT();
  llvm::errs() << "Proto '" << node->getName() << "' " << loc(node) << '\n';
  indent();
  llvm::errs() << "Params: [";
  llvm::interleaveComma(node->getArgs(), llvm::errs(), [&](const auto &arg) {
    llvm::errs() << arg->getName();
  });
  llvm::errs() << "]\n";
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(FunctionAST *node) {
  INDENT();
  llvm::errs() << "Function\n";
  dump(node->getProto());
  dump(node->getBody());
}

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(ModuleAST *node) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto &F : *node) {
    dump(&F);
  }
}

namespace toy {
// Public API
void dump(ModuleAST &module) { ASTDumper().dump(&module); }
} // namespace toy