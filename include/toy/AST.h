//===--- AST.h - AST for the Toy language ---------------------------------===//
//
// This file implements the AST for the Toy language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_AST_H
#define TOY_AST_H

#include "toy/Lexer.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include <llvm/ADT/StringRef.h>

namespace toy {
/// A variable type with shape information
struct VarType {
  std::vector<int64_t> shape;
};

/// Base class for all expression nodes.
class ExprAST {
public:
  enum class ExprASTKind {
    VarDecl,
    Return,
    Num,
    Literal,
    Var,
    BinOp,
    Call,
    Print,
    Transpose,
  };

  ExprAST(ExprASTKind astKind, SourceLocation location)
      : kind(astKind), location(std::move(location)) {}

  ExprASTKind getKind() const { return kind; }

  constexpr llvm::StringRef getStrKind() const { return kindToString(kind); }

  const SourceLocation &loc() { return location; }

private:
  const ExprASTKind kind;
  SourceLocation location;

  constexpr llvm::StringRef kindToString(const ExprASTKind &kind) const {
    switch (kind) {
    case ExprASTKind::VarDecl:
      return "VarDecl";
    case ExprASTKind::Return:
      return "Return";
    case ExprASTKind::Num:
      return "Num";
    case ExprASTKind::Literal:
      return "Literal";
    case ExprASTKind::Var:
      return "Var";
    case ExprASTKind::BinOp:
      return "BinOp";
    case ExprASTKind::Call:
      return "Call";
    case ExprASTKind::Print:
      return "Print";
    case ExprASTKind::Transpose:
      return "Transpose";
    default:
      return "Unknown";
    }
  }
};

/// Block-list of expressions
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0"
class NumberExprAST : public ExprAST {
  double val;

public:
  NumberExprAST(SourceLocation loc, double value)
      : ExprAST(ExprAST::ExprASTKind::Num, std::move(loc)), val(value) {}

  double getValue() { return val; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Num;
  }
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;

public:
  LiteralExprAST(SourceLocation location,
                 std::vector<std::unique_ptr<ExprAST>> valueList,
                 std::vector<int64_t> dimensions)
      : ExprAST(ExprAST::ExprASTKind::Literal, std::move(location)),
        values(std::move(valueList)), dims(std::move(dimensions)) {}

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }

  llvm::ArrayRef<int64_t> getDims() { return dims; }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Literal;
  }
};

/// Expression class for referencing a variable, like "a"
class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(SourceLocation loc, std::string varName)
      : ExprAST(ExprAST::ExprASTKind::Var, std::move(loc)),
        name(std::move(varName)) {}

  llvm::StringRef getName() { return name; }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Var;
  }
};

/// Expression class for declaring a variable
class VarDeclExprAST : public ExprAST {
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;

public:
  VarDeclExprAST(SourceLocation loc, std::string varName, VarType varType,
                 std::unique_ptr<ExprAST> initializer)
      : ExprAST(ExprAST::ExprASTKind::VarDecl, std::move(loc)),
        name(std::move(varName)), type(std::move(varType)),
        initVal(std::move(initializer)) {}

  llvm::StringRef getName() { return name; }

  VarType getType() { return type; }

  ExprAST *getInitVal() { return initVal.get(); }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::VarDecl;
  }
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
  std::optional<std::unique_ptr<ExprAST>> expr;

public:
  ReturnExprAST(SourceLocation loc,
                std::optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(ExprAST::ExprASTKind::Return, std::move(loc)),
        expr(std::move(expr)) {}

  std::optional<ExprAST *> getExpr() {
    if (expr.has_value()) {
      return expr->get();
    }

    return std::nullopt;
  }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Return;
  }
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;

public:
  BinaryExprAST(SourceLocation loc, char opChar, std::unique_ptr<ExprAST> left,
                std::unique_ptr<ExprAST> right)
      : ExprAST(ExprAST::ExprASTKind::BinOp, std::move(loc)), op(opChar),
        lhs(std::move(left)), rhs(std::move(right)) {}

  char getOp() { return op; }

  ExprAST *getLhs() { return lhs.get(); }

  ExprAST *getRhs() { return rhs.get(); }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::BinOp;
  }
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string callee;
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  CallExprAST(SourceLocation loc, std::string callee,
              std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(ExprAST::ExprASTKind::Call, std::move(loc)),
        callee(std::move(callee)), args(std::move(args)) {}

  llvm::StringRef getCallee() { return callee; }

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return args; }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Call;
  }
};

/// Expression class for builint print calls.
class PrintExprAST : public ExprAST {
  std::unique_ptr<ExprAST> arg;

public:
  PrintExprAST(SourceLocation loc, std::unique_ptr<ExprAST> arg)
      : ExprAST(ExprAST::ExprASTKind::Print, std::move(loc)),
        arg(std::move(arg)) {}

  ExprAST *getArg() { return arg.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Print;
  }
};

/// Expression class for builint transpose calls.
class TransposeExprAST : public ExprAST {
  std::unique_ptr<ExprAST> arg;

public:
  TransposeExprAST(SourceLocation loc, std::unique_ptr<ExprAST> arg)
      : ExprAST(ExprAST::ExprASTKind::Transpose, std::move(loc)),
        arg(std::move(arg)) {}

  ExprAST *getArg() { return arg.get(); }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Transpose;
  }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
  SourceLocation location;
  std::string name;
  std::vector<std::unique_ptr<VariableExprAST>> args;

public:
  PrototypeAST(SourceLocation loc, const std::string &nameStr,
               std::vector<std::unique_ptr<VariableExprAST>> args)
      : location(std::move(loc)), name(nameStr), args(std::move(args)) {}

  const SourceLocation &loc() { return location; }

  llvm::StringRef getName() { return name; }

  llvm::ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() { return args; }
};

/// This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprASTList> body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> prototype,
              std::unique_ptr<ExprASTList> functionBody)
      : proto(std::move(prototype)), body(std::move(functionBody)) {}

  PrototypeAST *getProto() { return proto.get(); }

  ExprASTList *getBody() { return body.get(); }
};

/// This class represents a list of functions to be processed together.
class ModuleAST {
  std::vector<FunctionAST> functions;

public:
  ModuleAST(std::vector<FunctionAST> functionList)
      : functions(std::move(functionList)) {}

  auto begin() { return functions.begin(); }

  auto end() { return functions.end(); }
};

void dump(ModuleAST &);

} // namespace toy

#endif // TOY_AST_H
