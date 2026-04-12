//===--- AST.cpp - Helper for printing the AST ----------------------------===//
//
// This file implements the AST dump for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "toy/AST.h"

#include <llvm/ADT/STLExtras.h>
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace toy;

namespace {
// RAII helper to mange increasing/decreasing the identation as we traverse the
// AST
struct Indent {
  Indent(int &Level) : Level(Level) { ++Level; }
  ~Indent() { --Level; }
  int &Level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
public:
  void dump(ModuleAST *Node);

private:
  int CurIdent = 0;

  void dump(const VarType &Type);
  void dump(VarDeclExprAST *VarDecl);
  void dump(ExprAST *Expr);
  void dump(ExprASTList *ExprList);
  void dump(NumberExprAST *Num);
  void dump(LiteralExprAST *Node);
  void dump(VariableExprAST *Node);
  void dump(ReturnExprAST *Node);
  void dump(BinaryExprAST *Node);
  void dump(CallExprAST *Node);
  void dump(PrintExprAST *Node);
  void dump(TransposeExprAST *Node);
  void dump(PrototypeAST *Node);
  void dump(FunctionAST *Node);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < CurIdent; i++) {
      llvm::errs() << " ";
    }
  }
};

} // namespace

/// Return a formatted string for the location of any node
template <typename T> static std::string loc(T *Node) {
  const auto &Loc = Node->loc();
  return (llvm::Twine("@") + *Loc.File + ":" + llvm::Twine(Loc.Line) + ":" +
          llvm::Twine(Loc.Col))
      .str();
}

// Helper macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(CurIdent);                                                     \
  indent();

/// Dispatch the generic expression to the appropriate subclass using RTTI
void ASTDumper::dump(ExprAST *Expr) {
  llvm::TypeSwitch<ExprAST *>(Expr)
      .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
            PrintExprAST, ReturnExprAST, TransposeExprAST, VarDeclExprAST,
            VariableExprAST>([&](auto *node) { this->dump(node); })
      .Default([&](ExprAST *) {
        // No match fallback to generic message
        INDENT();
        llvm::errs() << "<unknow Expr, kind " << Expr->getStrKind() << ">\n";
      });
}

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(VarDeclExprAST *VarDecl) {
  INDENT();
  llvm::errs() << "VarDecl " << VarDecl->getName();
  dump(VarDecl->getType());
  llvm::errs() << " " << loc(VarDecl) << "\n";
  dump(VarDecl->getInitVal());
}

/// A "block", or a list of expression.
void ASTDumper::dump(ExprASTList *ExprList) {
  INDENT();
  llvm::errs() << "Block {\n";
  for (auto &E : *ExprList) {
    dump(E.get());
  }
  llvm::errs() << "} // Block\n";
}

/// A literal number, just print the value
void ASTDumper::dump(NumberExprAST *Num) {
  INDENT();
  llvm::errs() << Num->getValue() << " " << loc(Num) << "\n";
}

/// Helper to print recursively a literal. This handles nested array like:
/// [ [ 1, 2 ], [ 3, 4 ] ]
/// We print out such array wiht the dimensions spelled out at every level:
/// <2,2>[<2>[ 1, 2 ], <2>[ 3, 4 ] ]
static void printLitHelper(ExprAST *LitOrNum) {
  // Inside a literal expression we can have either a number or another literal
  if (auto *Num = llvm::dyn_cast<NumberExprAST>(LitOrNum)) {
    llvm::errs() << Num->getValue();
    return;
  }

  auto *Lit = llvm::cast<LiteralExprAST>(LitOrNum);

  // Print the dimensions for the literal first
  llvm::errs() << "<";
  llvm::interleaveComma(Lit->getDims(), llvm::errs());
  llvm::errs() << ">";

  // Now print the content, recursing on every element of the list
  llvm::errs() << "[ ";
  llvm::interleaveComma(Lit->getValues(), llvm::errs(),
                        [&](const auto &E) { printLitHelper(E.get()); });
  llvm::errs() << "]";
}

/// Print a literal
void ASTDumper::dump(LiteralExprAST *Node) {
  INDENT();
  llvm::errs() << "Literal: ";
  printLitHelper(Node);
  llvm::errs() << " " << loc(Node) << "\n";
}

/// Print a variable reference (just a name)
void ASTDumper::dump(VariableExprAST *Node) {
  INDENT();
  llvm::errs() << "Var: " << Node->getName() << " " << loc(Node) << "\n";
}

/// Return statement print the return and its (optional) arguments.
void ASTDumper::dump(ReturnExprAST *Node) {
  INDENT();
  llvm::errs() << "Return\n";
  if (Node->getExpr().has_value()) {
    return dump(*Node->getExpr());
  }

  {
    INDENT();
    llvm::errs() << "(void)\n";
  }
}

/// Print a binary operation, first the operator, then recurse into LHS and RHS
void ASTDumper::dump(BinaryExprAST *Node) {
  INDENT();
  llvm::errs() << "BinOps: " << Node->getOp() << " " << loc(Node) << '\n';
  dump(Node->getLhs());
  dump(Node->getRhs());
}

/// Print a call expression, first the callee name and the list of args by
/// recursing into each individual argument.
void ASTDumper::dump(CallExprAST *Node) {
  INDENT();
  llvm::errs() << "Call '" << Node->getCalle() << "' [" << loc(Node) << '\n';
  for (auto &Arg : Node->getArgs()) {
    dump(Arg.get());
  }
  indent();
  llvm::errs() << "]\n";
}

/// Print a builtin "print" call, first the builtin name and then the argument.
void ASTDumper::dump(PrintExprAST *Node) {
  INDENT();
  llvm::errs() << "Print [" << loc(Node) << '\n';
  dump(Node->getArg());
  indent();
  llvm::errs() << "]\n";
}

/// Print a builtin "transpose" call, first the builtin name and then the
/// argument.
void ASTDumper::dump(TransposeExprAST *Node) {
  INDENT();
  llvm::errs() << "Transpose [" << loc(Node) << '\n';
  dump(Node->getArg());
  indent();
  llvm::errs() << "]\n";
}

/// Print type: only the shape is printed in between '<' and '>'
void ASTDumper::dump(const VarType &Type) {
  llvm::errs() << "<";
  llvm::interleaveComma(Type.Shape, llvm::errs());
  llvm::errs() << ">";
}

/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(PrototypeAST *Node) {
  INDENT();
  llvm::errs() << "Proto '" << Node->getName() << "' " << loc(Node) << '\n';
  indent();
  llvm::errs() << "Params: [";
  llvm::interleaveComma(Node->getArgs(), llvm::errs(), [&](const auto &Arg) {
    llvm::errs() << Arg->getName();
  });
  llvm::errs() << "]\n";
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(FunctionAST *Node) {
  INDENT();
  llvm::errs() << "Function\n";
  dump(Node->getProto());
  dump(Node->getBody());
}

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(ModuleAST *Node) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto &F : *Node) {
    dump(&F);
  }
}

namespace toy {
// Public API
void dump(ModuleAST &Module) { ASTDumper().dump(&Module); }
} // namespace toy