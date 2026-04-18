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
  std::vector<int64_t> Shape;
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

  ExprAST(ExprASTKind ASTKind, SourceLocation Location)
      : Kind(ASTKind), Loc(std::move(Location)) {}

  ExprASTKind getKind() const { return Kind; }

  constexpr llvm::StringRef getStrKind() const { return kindToString(Kind); }

  const SourceLocation &loc() { return Loc; }

private:
  const ExprASTKind Kind;
  SourceLocation Loc;

  constexpr llvm::StringRef kindToString(const ExprASTKind &K) const {
    switch (K) {
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
  double Val;

public:
  NumberExprAST(SourceLocation Location, double Value)
      : ExprAST(ExprAST::ExprASTKind::Num, std::move(Location)), Val(Value) {}

  double getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Num;
  }
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> Values;
  std::vector<int64_t> Dims;

public:
  LiteralExprAST(SourceLocation Location,
                 std::vector<std::unique_ptr<ExprAST>> ValueList,
                 std::vector<int64_t> Dimensions)
      : ExprAST(ExprAST::ExprASTKind::Literal, std::move(Location)),
        Values(std::move(ValueList)), Dims(std::move(Dimensions)) {}

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return Values; }

  llvm::ArrayRef<int64_t> getDims() { return Dims; }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Literal;
  }
};

/// Expression class for referencing a variable, like "a"
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  VariableExprAST(SourceLocation Location, std::string VariableName)
      : ExprAST(ExprAST::ExprASTKind::Var, std::move(Location)),
        Name(VariableName) {}

  llvm::StringRef getName() { return Name; }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Var;
  }
};

/// Expression class for declaring a variable
class VarDeclExprAST : public ExprAST {
  std::string Name;
  VarType Type;
  std::unique_ptr<ExprAST> InitVal;

public:
  VarDeclExprAST(SourceLocation Location, std::string VariableName,
                 VarType VariableType, std::unique_ptr<ExprAST> Initializer)
      : ExprAST(ExprAST::ExprASTKind::VarDecl, std::move(Location)),
        Name(VariableName), Type(VariableType),
        InitVal(std::move(Initializer)) {}

  llvm::StringRef getName() { return Name; }

  VarType getType() { return Type; }

  ExprAST *getInitVal() { return InitVal.get(); }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::VarDecl;
  }
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
  std::optional<std::unique_ptr<ExprAST>> Expr;

public:
  ReturnExprAST(SourceLocation Location,
                std::optional<std::unique_ptr<ExprAST>> Expression)
      : ExprAST(ExprAST::ExprASTKind::Return, std::move(Location)),
        Expr(std::move(Expression)) {}

  std::optional<ExprAST *> getExpr() {
    if (Expr.has_value()) {
      return Expr->get();
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
  char Op;
  std::unique_ptr<ExprAST> Lhs, Rhs;

public:
  BinaryExprAST(SourceLocation Location, char OpChar,
                std::unique_ptr<ExprAST> Left, std::unique_ptr<ExprAST> Right)
      : ExprAST(ExprAST::ExprASTKind::BinOp, std::move(Location)), Op(OpChar),
        Lhs(std::move(Left)), Rhs(std::move(Right)) {}

  char getOp() { return Op; }

  ExprAST *getLhs() { return Lhs.get(); }

  ExprAST *getRhs() { return Rhs.get(); }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::BinOp;
  }
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Calle;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(SourceLocation Location, std::string Callee,
              std::vector<std::unique_ptr<ExprAST>> Arguments)
      : ExprAST(ExprAST::ExprASTKind::Call, std::move(Location)), Calle(Callee),
        Args(std::move(Arguments)) {}

  llvm::StringRef getCalle() { return Calle; }

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return Args; }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Call;
  }
};

/// Expression class for builint print calls.
class PrintExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Arg;

public:
  PrintExprAST(SourceLocation Location, std::unique_ptr<ExprAST> Argument)
      : ExprAST(ExprAST::ExprASTKind::Print, std::move(Location)),
        Arg(std::move(Argument)) {}

  ExprAST *getArg() { return Arg.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Print;
  }
};

/// Expression class for builint transpose calls.
class TransposeExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Arg;

public:
  TransposeExprAST(SourceLocation Location, std::unique_ptr<ExprAST> Argument)
      : ExprAST(ExprAST::ExprASTKind::Transpose, std::move(Location)),
        Arg(std::move(Argument)) {}

  ExprAST *getArg() { return Arg.get(); }

  /// LLVM stype RTTI
  static bool classof(const ExprAST *C) {
    return C->getKind() == ExprAST::ExprASTKind::Transpose;
  }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
  SourceLocation Loc;
  std::string Name;
  std::vector<std::unique_ptr<VariableExprAST>> Args;

public:
  PrototypeAST(SourceLocation Location, const std::string &NameStr,
               std::vector<std::unique_ptr<VariableExprAST>> Arguments)
      : Loc(std::move(Location)), Name(NameStr), Args(std::move(Arguments)) {}

  const SourceLocation &loc() { return Loc; }

  llvm::StringRef getName() { return Name; }

  llvm::ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() { return Args; }
};

/// This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprASTList> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Prototype,
              std::unique_ptr<ExprASTList> FunctionBody)
      : Proto(std::move(Prototype)), Body(std::move(FunctionBody)) {}

  PrototypeAST *getProto() { return Proto.get(); }

  ExprASTList *getBody() { return Body.get(); }
};

/// This class represents a list of functions to be processed together.
class ModuleAST {
  std::vector<FunctionAST> Functions;

public:
  ModuleAST(std::vector<FunctionAST> FunctionList)
      : Functions(std::move(FunctionList)) {}

  auto begin() { return Functions.begin(); }

  auto end() { return Functions.end(); }
};

void dump(ModuleAST &);

} // namespace toy

#endif // TOY_AST_H
